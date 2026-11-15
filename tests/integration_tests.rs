use mini_columnar_engine::column::bitmap::NullBitmap;
use mini_columnar_engine::column::dictionary::DictionaryColumn;
use mini_columnar_engine::column::numeric::{Float64Column, Int64Column};
use mini_columnar_engine::column::rle::RleColumn;
use mini_columnar_engine::column::string::StringColumn;
use mini_columnar_engine::column::{Column, ColumnType};
use mini_columnar_engine::execution::aggregate::{
    avg_f64, avg_i64, group_by_dict_count, group_by_dict_sum_i64, group_by_raw_string_sum_i64,
    max_f64, max_i64, min_f64, min_i64, sum_f64, sum_i64,
};
use mini_columnar_engine::execution::filter::{
    scan_filter_dict, scan_filter_i64, scan_filter_i64_with_selection, ComparisonOp,
    SelectionVector,
};
use mini_columnar_engine::table::{AnyColumn, ColumnTable};
use mini_columnar_engine::{AggregateOp, AggregateValue};

// =============================================================================
// Pillar 1: Contiguous Columnar Memory & NullBitmap Tests
// =============================================================================

#[test]
fn test_null_bitmap_basic_and_boundaries() {
    // Test across 64-bit word boundaries (0, 1, 63, 64, 65, 130)
    let sizes = [0, 1, 63, 64, 65, 128, 130];
    for &size in &sizes {
        let bm = NullBitmap::new(size, true);
        assert_eq!(bm.len(), size);
        assert_eq!(bm.null_count(), 0);
        for i in 0..size {
            assert!(bm.is_valid(i));
            assert!(!bm.is_null(i));
        }

        let bm_null = NullBitmap::new(size, false);
        assert_eq!(bm_null.len(), size);
        assert_eq!(bm_null.null_count(), size);
        for i in 0..size {
            assert!(!bm_null.is_valid(i));
            assert!(bm_null.is_null(i));
        }
    }
}

#[test]
fn test_null_bitmap_mutations_and_append() {
    let mut bm = NullBitmap::new(100, true);
    bm.set_null(0);
    bm.set_null(63);
    bm.set_null(64);
    bm.set_null(99);

    assert_eq!(bm.null_count(), 4);
    assert!(bm.is_null(0));
    assert!(bm.is_null(63));
    assert!(bm.is_null(64));
    assert!(bm.is_null(99));
    assert!(bm.is_valid(1));
    assert!(bm.is_valid(62));

    bm.set_valid(0);
    assert_eq!(bm.null_count(), 3);
    assert!(bm.is_valid(0));

    bm.append_valid();
    assert_eq!(bm.len(), 101);
    assert!(bm.is_valid(100));

    bm.append_null();
    assert_eq!(bm.len(), 102);
    assert!(bm.is_null(101));
    assert_eq!(bm.null_count(), 4);
}

#[test]
fn test_null_bitmap_bitwise_logic() {
    let a = NullBitmap::from_bools(&[true, false, true, false, true]);
    let b = NullBitmap::from_bools(&[true, true, false, false, true]);

    let and_res = a.bit_and(&b);
    assert_eq!(
        (0..5).map(|i| and_res.is_valid(i)).collect::<Vec<_>>(),
        vec![true, false, false, false, true]
    );

    let or_res = a.bit_or(&b);
    assert_eq!(
        (0..5).map(|i| or_res.is_valid(i)).collect::<Vec<_>>(),
        vec![true, true, true, false, true]
    );
}

#[test]
fn test_int64_column_with_and_without_nulls() {
    let col_valid = Int64Column::from_slice("c1", &[10, 20, 30, 40]);
    assert_eq!(col_valid.len(), 4);
    assert!(!col_valid.has_nulls());
    assert_eq!(col_valid.get(2), Some(30));
    assert_eq!(col_valid.get(10), None);

    let col_opt = Int64Column::from_optional_iter("c2", vec![Some(1), None, Some(3), None]);
    assert_eq!(col_opt.len(), 4);
    assert!(col_opt.has_nulls());
    assert_eq!(col_opt.get(0), Some(1));
    assert_eq!(col_opt.get(1), None);
    assert_eq!(col_opt.get(2), Some(3));
    assert_eq!(col_opt.get(3), None);

    // Push into column with nulls
    let mut col_dyn = Int64Column::new("c3");
    col_dyn.push(100);
    col_dyn.push_null();
    col_dyn.push(200);
    assert_eq!(col_dyn.len(), 3);
    assert_eq!(col_dyn.get(0), Some(100));
    assert_eq!(col_dyn.get(1), None);
    assert_eq!(col_dyn.get(2), Some(200));
}

#[test]
fn test_float64_column() {
    let mut col = Float64Column::from_slice("f1", &[1.5, 2.5, 3.5]);
    assert_eq!(col.column_type(), ColumnType::Float64);
    col.push(4.5);
    col.push_null();
    assert_eq!(col.len(), 5);
    assert_eq!(col.get(3), Some(4.5));
    assert_eq!(col.get(4), None);
    assert!(col.has_nulls());
}

// =============================================================================
// Pillar 2: Lightweight Database Compression (Dictionary & RLE)
// =============================================================================

#[test]
fn test_dictionary_column_encoding_and_surrogates() {
    let mut dict_col = DictionaryColumn::new("status");
    let id_pending = dict_col.push("PENDING");
    let id_approved = dict_col.push("APPROVED");
    let id_pending2 = dict_col.push("PENDING");
    let id_rejected = dict_col.push("REJECTED");
    dict_col.push_null();

    assert_eq!(id_pending, 0);
    assert_eq!(id_approved, 1);
    assert_eq!(id_pending2, 0); // Reused surrogate ID!
    assert_eq!(id_rejected, 2);

    assert_eq!(dict_col.cardinality(), 3);
    assert_eq!(dict_col.len(), 5);
    assert_eq!(dict_col.lookup_id("APPROVED"), Some(1));
    assert_eq!(dict_col.lookup_id("UNKNOWN"), None);

    assert_eq!(dict_col.get(0), Some("PENDING"));
    assert_eq!(dict_col.get(1), Some("APPROVED"));
    assert_eq!(dict_col.get(4), None); // Null row
    assert!(dict_col.memory_bytes() > 0);
}

#[test]
fn test_rle_column_compression_and_algebraic_sum() {
    let mut rle = RleColumn::new("run_int");
    for _ in 0..10_000 {
        rle.push(5i64);
    }
    for _ in 0..5_000 {
        rle.push(10i64);
    }
    for _ in 0..2_000 {
        rle.push(20i64);
    }

    assert_eq!(rle.len(), 17_000);
    assert_eq!(rle.num_runs(), 3);

    // O(runs) evaluation: (10,000 * 5) + (5,000 * 10) + (2,000 * 20) = 50,000 + 50,000 + 40,000 = 140,000
    let sum = rle.sum();
    assert_eq!(sum, 140_000);

    // Random access binary search test
    assert_eq!(rle.get(0), Some(&5));
    assert_eq!(rle.get(9_999), Some(&5));
    assert_eq!(rle.get(10_000), Some(&10));
    assert_eq!(rle.get(14_999), Some(&10));
    assert_eq!(rle.get(15_000), Some(&20));
    assert_eq!(rle.get(16_999), Some(&20));
    assert_eq!(rle.get(17_000), None);

    // Decompress test
    let flat = rle.decompress();
    assert_eq!(flat.len(), 17_000);
    assert_eq!(flat.iter().sum::<i64>(), 140_000);
}

#[test]
fn test_string_column_raw() {
    let mut raw = StringColumn::new("city");
    raw.push("Budapest");
    raw.push("Munich");
    raw.push("Walldorf");
    raw.push_null();

    assert_eq!(raw.len(), 4);
    assert_eq!(raw.get(0), Some("Budapest"));
    assert_eq!(raw.get(1), Some("Munich"));
    assert_eq!(raw.get(3), None);

    let matches = raw.scan_equals_raw("Munich");
    assert_eq!(matches, vec![1]);
}

// =============================================================================
// Pillar 3: Vectorized Execution & Predicate Pushdown Tests
// =============================================================================

#[test]
fn test_selection_vector_logic() {
    let sv1 = SelectionVector::from_indices(vec![1, 3, 5, 7, 9]);
    let sv2 = SelectionVector::from_indices(vec![3, 4, 5, 8, 9]);

    let and_sv = sv1.and(&sv2);
    assert_eq!(and_sv.as_slice(), &[3, 5, 9]);

    let or_sv = sv1.or(&sv2);
    assert_eq!(or_sv.as_slice(), &[1, 3, 4, 5, 7, 8, 9]);

    let all_sv = SelectionVector::all(5);
    assert_eq!(all_sv.as_slice(), &[0, 1, 2, 3, 4]);
    assert_eq!(all_sv.selectivity(10), 0.5);
}

#[test]
fn test_vectorized_scan_i64_ops_and_boundaries() {
    // Test across unrolled chunk boundaries: 0, 7, 8, 9, 16, 25 elements
    let vals = vec![
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
        210, 220, 230, 240, 250,
    ];
    let col = Int64Column::from_slice("val", &vals);

    // Equal
    let sel_eq = scan_filter_i64(&col, ComparisonOp::Equal, 50);
    assert_eq!(sel_eq.as_slice(), &[4]);

    // GreaterThan
    let sel_gt = scan_filter_i64(&col, ComparisonOp::GreaterThan, 200);
    assert_eq!(sel_gt.as_slice(), &[20, 21, 22, 23, 24]);

    // GreaterThanOrEqual
    let sel_gte = scan_filter_i64(&col, ComparisonOp::GreaterThanOrEqual, 200);
    assert_eq!(sel_gte.as_slice(), &[19, 20, 21, 22, 23, 24]);

    // LessThan
    let sel_lt = scan_filter_i64(&col, ComparisonOp::LessThan, 40);
    assert_eq!(sel_lt.as_slice(), &[0, 1, 2]);

    // LessThanOrEqual
    let sel_lte = scan_filter_i64(&col, ComparisonOp::LessThanOrEqual, 40);
    assert_eq!(sel_lte.as_slice(), &[0, 1, 2, 3]);

    // NotEqual
    let sel_ne = scan_filter_i64(&col, ComparisonOp::NotEqual, 10);
    assert_eq!(sel_ne.len(), 24);
    assert!(!sel_ne.as_slice().contains(&0));

    // Empty col edge case
    let empty_col = Int64Column::new("empty");
    let sel_empty = scan_filter_i64(&empty_col, ComparisonOp::Equal, 10);
    assert!(sel_empty.is_empty());
}

#[test]
fn test_vectorized_scan_with_compound_selection() {
    let col1 = Int64Column::from_slice("c1", &[10, 20, 30, 40, 50, 60]);
    let col2 = Int64Column::from_slice("c2", &[100, 200, 300, 400, 500, 600]);

    // c1 > 20 -> indices [2, 3, 4, 5]
    let sel1 = scan_filter_i64(&col1, ComparisonOp::GreaterThan, 20);
    // Compound filter: c2 < 500 on sel1 -> indices [2, 3]
    let sel_compound = scan_filter_i64_with_selection(&col2, &sel1, ComparisonOp::LessThan, 500);

    assert_eq!(sel_compound.as_slice(), &[2, 3]);
}

#[test]
fn test_vectorized_scan_dictionary_pruning() {
    let dict_col = DictionaryColumn::from_iter(
        "region",
        vec!["EMEA", "APAC", "EMEA", "AMER", "APAC", "EMEA"],
    );

    // Present string
    let sel_emea = scan_filter_dict(&dict_col, ComparisonOp::Equal, "EMEA");
    assert_eq!(sel_emea.as_slice(), &[0, 2, 5]);

    // Absent string -> Instant pruning without scanning!
    let sel_absent = scan_filter_dict(&dict_col, ComparisonOp::Equal, "LATAM");
    assert!(sel_absent.is_empty());

    // Not equal
    let sel_not_emea = scan_filter_dict(&dict_col, ComparisonOp::NotEqual, "EMEA");
    assert_eq!(sel_not_emea.as_slice(), &[1, 3, 4]);

    // Absent string not equal -> matches all rows
    let sel_not_absent = scan_filter_dict(&dict_col, ComparisonOp::NotEqual, "LATAM");
    assert_eq!(sel_not_absent.as_slice(), &[0, 1, 2, 3, 4, 5]);
}

// =============================================================================
// Pillar 4: Analytical Aggregation Engine & Group By Tests
// =============================================================================

#[test]
fn test_numeric_aggregations_i64_and_f64() {
    let col_i = Int64Column::from_slice("v", &[10, 20, 30, 40, 50]);
    assert_eq!(sum_i64(&col_i, None), 150);
    assert_eq!(avg_i64(&col_i, None), Some(30.0));
    assert_eq!(min_i64(&col_i, None), Some(10));
    assert_eq!(max_i64(&col_i, None), Some(50));

    // With selection vector
    let sel = SelectionVector::from_indices(vec![1, 3]); // values 20 and 40
    assert_eq!(sum_i64(&col_i, Some(&sel)), 60);
    assert_eq!(avg_i64(&col_i, Some(&sel)), Some(30.0));
    assert_eq!(min_i64(&col_i, Some(&sel)), Some(20));
    assert_eq!(max_i64(&col_i, Some(&sel)), Some(40));

    // Float64
    let col_f = Float64Column::from_slice("f", &[1.5, 2.5, 3.5]);
    assert!((sum_f64(&col_f, None) - 7.5).abs() < 1e-6);
    assert!((avg_f64(&col_f, None).unwrap() - 2.5).abs() < 1e-6);
    assert_eq!(min_f64(&col_f, None), Some(1.5));
    assert_eq!(max_f64(&col_f, None), Some(3.5));

    // Empty col edge case
    let empty_i = Int64Column::new("e");
    assert_eq!(sum_i64(&empty_i, None), 0);
    assert_eq!(avg_i64(&empty_i, None), None);
    assert_eq!(min_i64(&empty_i, None), None);
    assert_eq!(max_i64(&empty_i, None), None);
}

#[test]
fn test_vectorized_direct_indexed_group_by_parity() {
    let categories = vec!["A", "B", "A", "C", "B", "A", "C", "A"];
    let amounts = vec![10i64, 20, 30, 40, 50, 60, 70, 80];

    let dict_col = DictionaryColumn::from_iter("cat", &categories);
    let val_col = Int64Column::from_slice("amt", &amounts);

    // 1. Columnar direct-indexed group-by
    let direct_results = group_by_dict_sum_i64(&dict_col, &val_col, None);

    // 2. Row baseline hash-map group-by
    let string_cats: Vec<String> = categories.iter().map(|s| s.to_string()).collect();
    let hash_results = group_by_raw_string_sum_i64(&string_cats, &amounts, None);

    let map_direct: std::collections::HashMap<String, i64> = direct_results.into_iter().collect();
    let map_hash: std::collections::HashMap<String, i64> = hash_results.into_iter().collect();

    assert_eq!(map_direct, map_hash);

    // Verify expected exact sums:
    // A: 10 + 30 + 60 + 80 = 180
    // B: 20 + 50 = 70
    // C: 40 + 70 = 110
    assert_eq!(map_direct.get("A"), Some(&180));
    assert_eq!(map_direct.get("B"), Some(&70));
    assert_eq!(map_direct.get("C"), Some(&110));

    // Group By Count
    let count_res = group_by_dict_count(&dict_col, None);
    let count_map: std::collections::HashMap<String, usize> = count_res.into_iter().collect();
    assert_eq!(count_map.get("A"), Some(&4));
    assert_eq!(count_map.get("B"), Some(&2));
    assert_eq!(count_map.get("C"), Some(&2));
}

// =============================================================================
// Table and End-to-End OLAP Pipeline Tests
// =============================================================================

#[test]
fn test_column_table_pipeline() {
    let mut table = ColumnTable::new("orders");

    let order_id = Int64Column::from_slice("order_id", &[1, 2, 3, 4, 5]);
    let amount = Float64Column::from_slice("amount", &[100.0, 250.0, 50.0, 400.0, 150.0]);
    let country = DictionaryColumn::from_iter("country", ["DE", "HU", "DE", "US", "HU"]);

    table.add_column(AnyColumn::Int64(order_id)).unwrap();
    table.add_column(AnyColumn::Float64(amount)).unwrap();
    table.add_column(AnyColumn::Dictionary(country)).unwrap();

    assert_eq!(table.row_count(), 5);
    assert_eq!(table.column_count(), 3);

    // Query: Filter country == "DE"
    let sel_de = table
        .filter_dict("country", ComparisonOp::Equal, "DE")
        .unwrap();
    assert_eq!(sel_de.as_slice(), &[0, 2]);

    // Aggregate: SUM(amount) where country == "DE" (100.0 + 50.0 = 150.0)
    let sum_de = table
        .aggregate("amount", AggregateOp::Sum, Some(&sel_de))
        .unwrap();
    assert_eq!(sum_de, AggregateValue::Float(150.0));

    // Group By: country, SUM(amount)
    let gb = table
        .group_by("country", "amount", AggregateOp::Sum, None)
        .unwrap();
    let gb_map: std::collections::HashMap<String, AggregateValue> = gb.into_iter().collect();
    assert_eq!(gb_map.get("DE"), Some(&AggregateValue::Float(150.0)));
    assert_eq!(gb_map.get("HU"), Some(&AggregateValue::Float(400.0)));
    assert_eq!(gb_map.get("US"), Some(&AggregateValue::Float(400.0)));
}

#[test]
fn test_column_table_mismatched_row_counts_error() {
    let mut table = ColumnTable::new("mismatched");
    let col1 = Int64Column::from_slice("c1", &[1, 2, 3]);
    let col2 = Int64Column::from_slice("c2", &[1, 2]); // only 2 rows

    assert!(table.add_column(AnyColumn::Int64(col1)).is_ok());
    let err = table.add_column(AnyColumn::Int64(col2));
    assert!(err.is_err());
}
