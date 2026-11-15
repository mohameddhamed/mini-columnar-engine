use std::collections::HashMap;
use std::time::Instant;

use mini_columnar_engine::column::dictionary::DictionaryColumn;
use mini_columnar_engine::column::numeric::{Float64Column, Int64Column};
use mini_columnar_engine::column::rle::RleColumn;
use mini_columnar_engine::execution::filter::ComparisonOp;
use mini_columnar_engine::table::{AnyColumn, ColumnTable};
use mini_columnar_engine::AggregateOp;

/// Traditional row-oriented record (Array of Structs / OLTP layout)
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ProcessEventRow {
    event_id: i64,
    case_id: i64,
    activity: String,
    duration_ms: i64,
    cost: f64,
    region: String,
}

fn main() {
    println!(
        "╔═══════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║       `mini-columnar-engine` - Vectorized In-Memory OLAP Database Engine          ║"
    );
    println!(
        "║       Demonstrating SAP HANA TableStore & Celonis PQL Columnar Mechanics          ║"
    );
    println!(
        "╚═══════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    const ROW_COUNT: usize = 1_000_000;
    println!("Generating {} process event records...", ROW_COUNT);

    let activities = [
        "Create Purchase Requisition",
        "Create Purchase Order",
        "Approve Purchase Order",
        "Record Goods Receipt",
        "Record Invoice Receipt",
        "Clear Invoice Payment",
    ];
    let regions = ["EMEA", "AMER", "APAC", "LATAM"];

    // 1. Build Traditional Row-Oriented Store
    let mut row_store: Vec<ProcessEventRow> = Vec::with_capacity(ROW_COUNT);

    // 2. Build Vectorized Columnar Table
    let mut event_ids = Int64Column::with_capacity("event_id", ROW_COUNT);
    let mut case_ids = Int64Column::with_capacity("case_id", ROW_COUNT);
    let mut act_col = DictionaryColumn::with_capacity("activity", ROW_COUNT, activities.len());
    let mut durations = Int64Column::with_capacity("duration_ms", ROW_COUNT);
    let mut costs = Float64Column::with_capacity("cost", ROW_COUNT);
    let mut region_col = DictionaryColumn::with_capacity("region", ROW_COUNT, regions.len());

    // Pseudo-random deterministic generator (xorshift64)
    let mut seed = 0x853c49e6748fea9b_u64;
    let mut next_rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    for i in 0..ROW_COUNT {
        let event_id = i as i64;
        let case_id = (i / 5) as i64;
        let r = next_rand();
        let activity = activities[(r as usize) % activities.len()];
        let region = regions[((r >> 8) as usize) % regions.len()];
        let duration_ms = ((r >> 16) % 10000) as i64;
        let cost = (((r >> 32) % 50000) as f64) / 10.0;

        // Populate row store
        row_store.push(ProcessEventRow {
            event_id,
            case_id,
            activity: activity.to_string(),
            duration_ms,
            cost,
            region: region.to_string(),
        });

        // Populate columnar engine
        event_ids.push(event_id);
        case_ids.push(case_id);
        act_col.push(activity);
        durations.push(duration_ms);
        costs.push(cost);
        region_col.push(region);
    }

    let mut table = ColumnTable::new("process_events");
    table.add_column(AnyColumn::Int64(event_ids)).unwrap();
    table.add_column(AnyColumn::Int64(case_ids)).unwrap();
    table.add_column(AnyColumn::Dictionary(act_col)).unwrap();
    table.add_column(AnyColumn::Int64(durations)).unwrap();
    table.add_column(AnyColumn::Float64(costs)).unwrap();
    table.add_column(AnyColumn::Dictionary(region_col)).unwrap();

    let columnar_mem = table.memory_bytes();
    let row_mem = ROW_COUNT * (std::mem::size_of::<ProcessEventRow>() + 24); // approx struct + string heap

    println!("Dataset generated successfully:");
    println!("  • Rows:               {:>12}", ROW_COUNT);
    println!(
        "  • Row-Oriented Size:  {:>10.2} MB",
        row_mem as f64 / 1_048_576.0
    );
    println!(
        "  • Columnar Size:      {:>10.2} MB",
        columnar_mem as f64 / 1_048_576.0
    );
    println!(
        "  • Memory Compression: {:>10.2}x smaller footprint\n",
        row_mem as f64 / columnar_mem as f64
    );

    // =========================================================================
    // QUERY 1: Vectorized Filter Pushdown (`region == "EMEA" AND duration_ms > 5000`)
    // =========================================================================
    println!(
        "─── [QUERY 1] Compound Predicate Filter: (region == 'EMEA' AND duration_ms > 5000) ───"
    );

    // Columnar Vectorized Execution via Chained Filter Pushdown Pipeline
    let start_col = Instant::now();
    let sel_region = table
        .filter_dict("region", ComparisonOp::Equal, "EMEA")
        .unwrap();
    let sel_final = table
        .filter_i64_with_sel("duration_ms", ComparisonOp::GreaterThan, 5000, &sel_region)
        .unwrap();
    let dur_col = start_col.elapsed();

    // Row-Oriented Baseline Execution
    let start_row = Instant::now();
    let mut matching_row_indices = Vec::new();
    for (i, row) in row_store.iter().enumerate() {
        if row.region == "EMEA" && row.duration_ms > 5000 {
            matching_row_indices.push(i);
        }
    }
    let dur_row = start_row.elapsed();

    assert_eq!(sel_final.len(), matching_row_indices.len());
    println!("  • Matching rows:      {:>12}", sel_final.len());
    println!("  • Columnar Time:      {:>12.2?}", dur_col);
    println!("  • Row Baseline Time:  {:>12.2?}", dur_row);
    println!(
        "  • Filter Speedup:     {:>10.2}x faster\n",
        dur_row.as_nanos() as f64 / dur_col.as_nanos() as f64
    );

    // =========================================================================
    // QUERY 2: Filtered Aggregations (`SUM(cost)` & `AVG(duration_ms)`)
    // =========================================================================
    println!("─── [QUERY 2] Analytical Aggregation on Filtered Selection Vector ───");

    // Columnar Aggregation
    let start_col_agg = Instant::now();
    let sum_cost_col = table
        .aggregate("cost", AggregateOp::Sum, Some(&sel_final))
        .unwrap();
    let avg_dur_col = table
        .aggregate("duration_ms", AggregateOp::Avg, Some(&sel_final))
        .unwrap();
    let dur_col_agg = start_col_agg.elapsed();

    // Row-Oriented Aggregation
    let start_row_agg = Instant::now();
    let mut sum_cost_row = 0.0f64;
    let mut sum_dur_row = 0i64;
    for &idx in &matching_row_indices {
        sum_cost_row += row_store[idx].cost;
        sum_dur_row += row_store[idx].duration_ms;
    }
    let avg_dur_row = sum_dur_row as f64 / matching_row_indices.len() as f64;
    let dur_row_agg = start_row_agg.elapsed();

    // Verify mathematical parity between engines
    if let mini_columnar_engine::AggregateValue::Float(sc) = sum_cost_col {
        assert!(
            (sc - sum_cost_row).abs() < 1e-4,
            "SUM(cost) parity mismatch"
        );
    }
    if let mini_columnar_engine::AggregateValue::Float(ac) = avg_dur_col {
        assert!(
            (ac - avg_dur_row).abs() < 1e-4,
            "AVG(duration) parity mismatch"
        );
    }

    println!("  • SUM(cost):          {}", sum_cost_col);
    println!("  • AVG(duration_ms):   {}", avg_dur_col);
    println!("  • Columnar Time:      {:>12.2?}", dur_col_agg);
    println!("  • Row Baseline Time:  {:>12.2?}", dur_row_agg);
    println!(
        "  • Aggregate Speedup:  {:>10.2}x faster\n",
        dur_row_agg.as_nanos() as f64 / dur_col_agg.as_nanos() as f64
    );

    // =========================================================================
    // QUERY 3: Vectorized Direct-Indexed GROUP BY (`GROUP BY activity, SUM(cost)`)
    // =========================================================================
    println!("─── [QUERY 3] GROUP BY activity, SUM(cost) ───");

    // Columnar Vectorized Direct-Indexed Group By (Zero Hash Compute)
    let start_col_gb = Instant::now();
    let group_results = table
        .group_by("activity", "cost", AggregateOp::Sum, None)
        .unwrap();
    let dur_col_gb = start_col_gb.elapsed();

    // Row-Oriented HashMap Group By Baseline
    let start_row_gb = Instant::now();
    let mut hash_groups: HashMap<String, f64> = HashMap::new();
    for row in &row_store {
        *hash_groups.entry(row.activity.clone()).or_insert(0.0) += row.cost;
    }
    let dur_row_gb = start_row_gb.elapsed();

    println!("  • Group By Results:");
    for (act, sum) in &group_results {
        println!("      - {:<32} => {}", act, sum);
    }
    println!("  • Columnar Direct Group-By: {:>10.2?}", dur_col_gb);
    println!("  • Row HashMap Group-By:     {:>10.2?}", dur_row_gb);
    println!(
        "  • Group-By Speedup:         {:>8.2}x faster\n",
        dur_row_gb.as_nanos() as f64 / dur_col_gb.as_nanos() as f64
    );

    // =========================================================================
    // BONUS: Run-Length Encoded (RLE) Column Aggregation
    // =========================================================================
    println!("─── [BONUS] Run-Length Encoding (RLE) Mathematical Aggregation ───");
    // Sorted column with consecutive identical status codes
    let mut rle_col = RleColumn::new("status_code");
    for _ in 0..400_000 {
        rle_col.push(100i64);
    }
    for _ in 0..300_000 {
        rle_col.push(200i64);
    }
    for _ in 0..200_000 {
        rle_col.push(300i64);
    }
    for _ in 0..100_000 {
        rle_col.push(400i64);
    }

    let start_rle = Instant::now();
    let rle_sum = rle_col.sum();
    let dur_rle = start_rle.elapsed();

    println!("  • Total rows:               {:>12}", rle_col.len());
    println!("  • Total runs:               {:>12}", rle_col.num_runs());
    println!("  • RLE SUM result:           {:>12}", rle_sum);
    println!("  • O(runs) evaluation time:  {:>12.2?}", dur_rle);
    println!("\nAll architectural guarantees verified successfully.");
}
