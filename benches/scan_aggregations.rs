use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use mini_columnar_engine::column::dictionary::DictionaryColumn;
use mini_columnar_engine::column::numeric::Int64Column;
use mini_columnar_engine::column::rle::RleColumn;
use mini_columnar_engine::column::string::StringColumn;
use mini_columnar_engine::execution::aggregate::{
    group_by_borrowed_str_sum_i64, group_by_dict_sum_i64, group_by_raw_string_sum_i64, sum_i64,
};
use mini_columnar_engine::execution::filter::{
    scan_chunk_filter_i64, scan_filter_dict, scan_filter_i64, scan_filter_i64_into, ComparisonOp,
    SelectionVectorPool,
};

/// Row-oriented structure representing a wide OLTP tuple (64 bytes).
#[derive(Clone)]
#[allow(dead_code)]
struct EventRecord {
    id: i64,
    timestamp: i64,
    status: i64,
    val_a: i64,
    val_b: i64,
    val_c: i64,
    val_d: i64,
    payload_metric: i64,
}

// -----------------------------------------------------------------------------
// Benchmark 1: Row-Oriented vs Columnar Scan & Aggregation (1,000,000 rows)
// -----------------------------------------------------------------------------
fn bench_row_vs_columnar_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("Row_vs_Columnar_Scan_Sum");
    const N: usize = 1_000_000;
    group.throughput(Throughput::Elements(N as u64));

    // Prepare Row-Oriented Data (Array of Structs)
    let rows: Vec<EventRecord> = (0..N)
        .map(|i| EventRecord {
            id: i as i64,
            timestamp: 1700000000 + i as i64,
            status: (i % 5) as i64,
            val_a: (i % 100) as i64,
            val_b: (i % 200) as i64,
            val_c: (i % 300) as i64,
            val_d: (i % 400) as i64,
            payload_metric: (i % 1000) as i64,
        })
        .collect();

    // Prepare Contiguous Columnar Buffer
    let col = Int64Column::from_iter("payload_metric", (0..N).map(|i| (i % 1000) as i64));

    // Row-oriented baseline: iterates over 64-byte structs, thrashing cache lines.
    // black_box is applied symmetrically to the slice reference.
    group.bench_function(BenchmarkId::new("RowOriented", N), |b| {
        b.iter(|| {
            let r_slice = black_box(&rows);
            let mut sum = 0i64;
            for r in r_slice {
                sum += r.payload_metric;
            }
            sum
        });
    });

    // Columnar engine: scans contiguous 8-byte integers with unrolled hardware prefetching
    group.bench_function(BenchmarkId::new("ColumnarContiguous", N), |b| {
        b.iter(|| sum_i64(black_box(&col), None));
    });

    group.finish();
}

// -----------------------------------------------------------------------------
// Benchmark 2: Raw String Scan vs Dictionary Integer Filter (1,000,000 rows)
// -----------------------------------------------------------------------------
fn bench_raw_string_vs_dictionary_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("String_Raw_vs_Dictionary_Filter");
    const N: usize = 1_000_000;
    group.throughput(Throughput::Elements(N as u64));

    let regions = ["EMEA", "APAC", "AMER", "LATAM"];
    let raw_strings: Vec<String> = (0..N)
        .map(|i| regions[i % regions.len()].to_string())
        .collect();

    let mut dict_col = DictionaryColumn::with_capacity("region", N, 4);
    for r in &raw_strings {
        dict_col.push(r);
    }

    let string_col = StringColumn::from_iter("region_raw", &raw_strings);

    // Raw String Scan: string slice comparison per row
    group.bench_function(BenchmarkId::new("RawStringScan", N), |b| {
        b.iter(|| string_col.scan_equals_raw(black_box("EMEA")));
    });

    // Dictionary Filter: single O(1) dictionary key lookup + contiguous u32 scan
    group.bench_function(BenchmarkId::new("DictionarySurrogateScan", N), |b| {
        b.iter(|| scan_filter_dict(black_box(&dict_col), ComparisonOp::Equal, black_box("EMEA")));
    });

    group.finish();
}

// -----------------------------------------------------------------------------
// Benchmark 3: Scalar Volcano Iterator vs Vectorized Filter Pushdown
// -----------------------------------------------------------------------------
fn bench_scalar_vs_vectorized_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scalar_vs_Vectorized_Filter");
    const N: usize = 1_000_000;
    group.throughput(Throughput::Elements(N as u64));

    let col = Int64Column::from_iter("duration", (0..N).map(|i| (i % 10_000) as i64));
    let threshold = 5_000i64;

    // Scalar row-by-row Volcano iterator
    group.bench_function(BenchmarkId::new("ScalarVolcanoNext", N), |b| {
        b.iter(|| {
            let mut matches = Vec::new();
            for (i, &val) in col.as_slice().iter().enumerate() {
                if val > threshold {
                    matches.push(i);
                }
            }
            matches
        });
    });

    // Vectorized Filter Pushdown producing SelectionVector
    group.bench_function(BenchmarkId::new("VectorizedFilterPushdown", N), |b| {
        b.iter(|| {
            scan_filter_i64(
                black_box(&col),
                ComparisonOp::GreaterThan,
                black_box(threshold),
            )
        });
    });

    // Zero-allocation Vector Filter into pooled SelectionVector
    let mut pool = SelectionVectorPool::new();
    group.bench_function(BenchmarkId::new("ZeroAllocVectorFilter", N), |b| {
        let mut out = pool.acquire(N / 2);
        b.iter(|| {
            scan_filter_i64_into(
                black_box(&col),
                ComparisonOp::GreaterThan,
                black_box(threshold),
                black_box(&mut out),
            );
        });
    });

    // Chunked Vectorized Processing (1024-element vector chunks)
    group.bench_function(BenchmarkId::new("ChunkedVectorFilter_1024", N), |b| {
        b.iter(|| {
            let mut total_matches = 0usize;
            for chunk in mini_columnar_engine::vector_chunks(col.len(), 1024) {
                let sel = scan_chunk_filter_i64(
                    black_box(&col),
                    &chunk,
                    ComparisonOp::GreaterThan,
                    black_box(threshold),
                );
                total_matches += sel.len();
            }
            total_matches
        });
    });

    group.finish();
}

// -----------------------------------------------------------------------------
// Benchmark 4: Direct-Indexed Group By vs HashMap Group By
// -----------------------------------------------------------------------------
fn bench_group_by_direct_vs_hashmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("GroupBy_Direct_vs_HashMap");
    const N: usize = 1_000_000;
    group.throughput(Throughput::Elements(N as u64));

    let categories = [
        "Procurement",
        "Production",
        "Logistics",
        "Sales",
        "Finance",
        "HR",
        "CustomerService",
        "Compliance",
    ];

    let group_strings: Vec<String> = (0..N)
        .map(|i| categories[i % categories.len()].to_string())
        .collect();
    let values: Vec<i64> = (0..N).map(|i| (i % 500) as i64).collect();

    let mut dict_col = DictionaryColumn::with_capacity("category", N, categories.len());
    for s in &group_strings {
        dict_col.push(s);
    }
    let val_col = Int64Column::from_iter("amount", values.clone());

    // Traditional HashMap Group By with owned Strings
    group.bench_function(BenchmarkId::new("HashMapOwnedString", N), |b| {
        b.iter(|| group_by_raw_string_sum_i64(black_box(&group_strings), black_box(&values), None));
    });

    // Optimized HashMap Group By with borrowed &str (zero string cloning)
    group.bench_function(BenchmarkId::new("HashMapBorrowedStr", N), |b| {
        b.iter(|| {
            group_by_borrowed_str_sum_i64(black_box(&group_strings), black_box(&values), None)
        });
    });

    // Columnar Direct-Indexed Group By (SAP HANA style)
    group.bench_function(BenchmarkId::new("DirectIndexedGroupBy", N), |b| {
        b.iter(|| group_by_dict_sum_i64(black_box(&dict_col), black_box(&val_col), None));
    });

    group.finish();
}

// -----------------------------------------------------------------------------
// Benchmark 5: RLE Algebraic Sum vs Dense Columnar Scan (1,000,000 rows)
// -----------------------------------------------------------------------------
fn bench_rle_algebraic_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("Rle_Algebraic_vs_Dense_Scan");
    const N: usize = 1_000_000;
    group.throughput(Throughput::Elements(N as u64));

    let mut rle = RleColumn::new("status_code");
    for _ in 0..400_000 {
        rle.push(100i64);
    }
    for _ in 0..300_000 {
        rle.push(200i64);
    }
    for _ in 0..200_000 {
        rle.push(300i64);
    }
    for _ in 0..100_000 {
        rle.push(400i64);
    }

    let dense_vals = rle.decompress();
    let dense_col = Int64Column::from_slice("dense_status", &dense_vals);

    // Dense contiguous scan: iterates through 1,000,000 elements
    group.bench_function(BenchmarkId::new("DenseColumnarScan", N), |b| {
        b.iter(|| sum_i64(black_box(&dense_col), None));
    });

    // RLE algebraic evaluation: 4 multiplications and 3 additions (O(runs))
    group.bench_function(BenchmarkId::new("RleAlgebraicSum", N), |b| {
        b.iter(|| black_box(&rle).sum());
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_row_vs_columnar_scan,
    bench_raw_string_vs_dictionary_filter,
    bench_scalar_vs_vectorized_filter,
    bench_group_by_direct_vs_hashmap,
    bench_rle_algebraic_sum,
);
criterion_main!(benches);
