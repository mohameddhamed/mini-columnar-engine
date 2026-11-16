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
    assert_eq!(sum, Some(140_000));

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
    assert_eq!(sum_i64(&col_i, None), Some(150));
    assert_eq!(avg_i64(&col_i, None), Some(30.0));
    assert_eq!(min_i64(&col_i, None), Some(10));
    assert_eq!(max_i64(&col_i, None), Some(50));

    // With selection vector
    let sel = SelectionVector::from_indices(vec![1, 3]); // values 20 and 40
    assert_eq!(sum_i64(&col_i, Some(&sel)), Some(60));
    assert_eq!(avg_i64(&col_i, Some(&sel)), Some(30.0));
    assert_eq!(min_i64(&col_i, Some(&sel)), Some(20));
    assert_eq!(max_i64(&col_i, Some(&sel)), Some(40));

    // Float64
    let col_f = Float64Column::from_slice("f", &[1.5, 2.5, 3.5]);
    assert!((sum_f64(&col_f, None).unwrap() - 7.5).abs() < 1e-6);
    assert!((avg_f64(&col_f, None).unwrap() - 2.5).abs() < 1e-6);
    assert_eq!(min_f64(&col_f, None), Some(1.5));
    assert_eq!(max_f64(&col_f, None), Some(3.5));

    // Empty col edge case: SQL / PQL standard requires SUM to return None (NULL)
    let empty_i = Int64Column::new("e");
    assert_eq!(sum_i64(&empty_i, None), None);
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

#[test]
fn test_aggregations_with_nulls_and_selection_vector() {
    let col_i = Int64Column::from_optional_iter("v_null", vec![Some(100), None, Some(200)]);
    let sel = SelectionVector::from_indices(vec![0, 1, 2]);

    assert_eq!(sum_i64(&col_i, Some(&sel)), Some(300));
    assert_eq!(avg_i64(&col_i, Some(&sel)), Some(150.0));
    assert_eq!(min_i64(&col_i, Some(&sel)), Some(100));
    assert_eq!(max_i64(&col_i, Some(&sel)), Some(200));

    // Negative numbers with null
    let col_neg = Int64Column::from_optional_iter("v_neg", vec![Some(-100), None, Some(-200)]);
    assert_eq!(max_i64(&col_neg, Some(&sel)), Some(-100));

    // All nulls
    let col_all_null = Int64Column::from_optional_iter("v_all_null", vec![None, None]);
    let sel_all = SelectionVector::from_indices(vec![0, 1]);
    assert_eq!(sum_i64(&col_all_null, Some(&sel_all)), None);
    assert_eq!(min_i64(&col_all_null, Some(&sel_all)), None);
    assert_eq!(max_i64(&col_all_null, Some(&sel_all)), None);
    assert_eq!(avg_i64(&col_all_null, Some(&sel_all)), None);

    // Float64
    let col_f = Float64Column::from_optional_iter("f_null", vec![Some(10.5), None, Some(20.5)]);
    assert_eq!(min_f64(&col_f, Some(&sel)), Some(10.5));
    assert_eq!(max_f64(&col_f, Some(&sel)), Some(20.5));
    assert_eq!(avg_f64(&col_f, Some(&sel)), Some(15.5));
}

#[test]
fn test_group_by_with_null_groups_and_null_values() {
    let mut dict_col = DictionaryColumn::new("status");
    dict_col.push_null(); // Row 0 is null
    dict_col.push("APPROVED"); // Row 1 is APPROVED (key 0)
    dict_col.push("APPROVED"); // Row 2 is APPROVED

    let val_col = Int64Column::from_slice("amt", &[999, 10, 20]);
    let direct_results = group_by_dict_sum_i64(&dict_col, &val_col, None);
    let map: std::collections::HashMap<String, i64> = direct_results.into_iter().collect();

    // APPROVED sum must NOT include row 0 (999) which was NULL!
    assert_eq!(map.get("APPROVED"), Some(&30));

    let count_res = group_by_dict_count(&dict_col, None);
    let count_map: std::collections::HashMap<String, usize> = count_res.into_iter().collect();
    // APPROVED count must be 2, NOT 3!
    assert_eq!(count_map.get("APPROVED"), Some(&2));
}

#[test]
fn test_count_aggregation_with_nulls() {
    let mut table = ColumnTable::new("null_test");
    let col_i = Int64Column::from_optional_iter("nullable_int", vec![Some(1), None, Some(3)]);
    table.add_column(AnyColumn::Int64(col_i)).unwrap();

    let count_val = table
        .aggregate("nullable_int", AggregateOp::Count, None)
        .unwrap();
    // Non-null count must be 2, not 3!
    assert_eq!(count_val, AggregateValue::Count(2));

    let sel = SelectionVector::from_indices(vec![0, 1]); // row 0: valid, row 1: null
    let count_sel = table
        .aggregate("nullable_int", AggregateOp::Count, Some(&sel))
        .unwrap();
    assert_eq!(count_sel, AggregateValue::Count(1));
}

#[test]
fn test_selection_vector_bitmap_conversions_and_not() {
    let sv = SelectionVector::from_indices(vec![1, 3, 5]);
    let bm = sv.to_bitmap(7);

    assert_eq!(bm.len(), 7);
    assert!(!bm.is_valid(0));
    assert!(bm.is_valid(1));
    assert!(!bm.is_valid(2));
    assert!(bm.is_valid(3));
    assert!(!bm.is_valid(4));
    assert!(bm.is_valid(5));
    assert!(!bm.is_valid(6));

    let sv_roundtrip = SelectionVector::from_bitmap(&bm);
    assert_eq!(sv_roundtrip.as_slice(), &[1, 3, 5]);

    let sv_not = sv.not(7);
    assert_eq!(sv_not.as_slice(), &[0, 2, 4, 6]);
}

#[test]
fn test_null_bitmap_bit_not_and_to_indices() {
    let bm = NullBitmap::from_bools(&[true, false, true, false, true]);
    let not_bm = bm.bit_not();

    assert_eq!(
        (0..5).map(|i| not_bm.is_valid(i)).collect::<Vec<_>>(),
        vec![false, true, false, true, false]
    );

    let indices = bm.to_indices();
    assert_eq!(indices, vec![0, 2, 4]);
}

#[test]
fn test_rle_aggregations_and_table_pipeline() {
    let mut rle = RleColumn::new("runs");
    for _ in 0..100 {
        rle.push(10i64);
    }
    for _ in 0..50 {
        rle.push(30i64);
    }
    for _ in 0..50 {
        rle.push(20i64);
    }

    assert_eq!(rle.sum(), Some(3500)); // 100*10 + 50*30 + 50*20 = 1000 + 1500 + 1000 = 3500
    assert_eq!(rle.min(), Some(10));
    assert_eq!(rle.max(), Some(30));
    assert_eq!(rle.avg(), Some(17.5)); // 3500 / 200 = 17.5

    let mut table = ColumnTable::new("rle_table");
    table.add_column(AnyColumn::RleInt64(rle)).unwrap();

    assert_eq!(
        table.aggregate("runs", AggregateOp::Sum, None).unwrap(),
        AggregateValue::Int(3500)
    );
    assert_eq!(
        table.aggregate("runs", AggregateOp::Min, None).unwrap(),
        AggregateValue::Int(10)
    );
    assert_eq!(
        table.aggregate("runs", AggregateOp::Max, None).unwrap(),
        AggregateValue::Int(30)
    );
    assert_eq!(
        table.aggregate("runs", AggregateOp::Avg, None).unwrap(),
        AggregateValue::Float(17.5)
    );
}

#[test]
fn test_group_by_avg_min_max() {
    let mut table = ColumnTable::new("sales");
    let region = DictionaryColumn::from_iter("region", ["EMEA", "APAC", "EMEA", "APAC", "EMEA"]);
    let revenue_i = Int64Column::from_slice("rev_i", &[100, 200, 300, 400, 500]);
    let revenue_f = Float64Column::from_slice("rev_f", &[10.0, 20.0, 30.0, 40.0, 50.0]);

    table.add_column(AnyColumn::Dictionary(region)).unwrap();
    table.add_column(AnyColumn::Int64(revenue_i)).unwrap();
    table.add_column(AnyColumn::Float64(revenue_f)).unwrap();

    // Group by AVG (Int64): EMEA (100+300+500)/3 = 300.0, APAC (200+400)/2 = 300.0
    let gb_avg_i = table
        .group_by("region", "rev_i", AggregateOp::Avg, None)
        .unwrap();
    let map_avg_i: std::collections::HashMap<_, _> = gb_avg_i.into_iter().collect();
    assert_eq!(map_avg_i.get("EMEA"), Some(&AggregateValue::Float(300.0)));
    assert_eq!(map_avg_i.get("APAC"), Some(&AggregateValue::Float(300.0)));

    // Group by MIN (Int64)
    let gb_min_i = table
        .group_by("region", "rev_i", AggregateOp::Min, None)
        .unwrap();
    let map_min_i: std::collections::HashMap<_, _> = gb_min_i.into_iter().collect();
    assert_eq!(map_min_i.get("EMEA"), Some(&AggregateValue::Int(100)));
    assert_eq!(map_min_i.get("APAC"), Some(&AggregateValue::Int(200)));

    // Group by MAX (Int64)
    let gb_max_i = table
        .group_by("region", "rev_i", AggregateOp::Max, None)
        .unwrap();
    let map_max_i: std::collections::HashMap<_, _> = gb_max_i.into_iter().collect();
    assert_eq!(map_max_i.get("EMEA"), Some(&AggregateValue::Int(500)));
    assert_eq!(map_max_i.get("APAC"), Some(&AggregateValue::Int(400)));

    // Group by MIN (Float64)
    let gb_min_f = table
        .group_by("region", "rev_f", AggregateOp::Min, None)
        .unwrap();
    let map_min_f: std::collections::HashMap<_, _> = gb_min_f.into_iter().collect();
    assert_eq!(map_min_f.get("EMEA"), Some(&AggregateValue::Float(10.0)));
    assert_eq!(map_min_f.get("APAC"), Some(&AggregateValue::Float(20.0)));

    // Group by MAX (Float64)
    let gb_max_f = table
        .group_by("region", "rev_f", AggregateOp::Max, None)
        .unwrap();
    let map_max_f: std::collections::HashMap<_, _> = gb_max_f.into_iter().collect();
    assert_eq!(map_max_f.get("EMEA"), Some(&AggregateValue::Float(50.0)));
    assert_eq!(map_max_f.get("APAC"), Some(&AggregateValue::Float(40.0)));
}

#[test]
fn test_compound_filter_f64_and_dict() {
    let mut table = ColumnTable::new("events");
    let act =
        DictionaryColumn::from_iter("act", ["Create", "Approve", "Create", "Approve", "Reject"]);
    let cost = Float64Column::from_slice("cost", &[10.0, 50.0, 15.0, 80.0, 5.0]);

    table.add_column(AnyColumn::Dictionary(act)).unwrap();
    table.add_column(AnyColumn::Float64(cost)).unwrap();

    // 1. Filter act == "Approve" -> indices [1, 3]
    let sel_act = table
        .filter_dict("act", ComparisonOp::Equal, "Approve")
        .unwrap();
    assert_eq!(sel_act.as_slice(), &[1, 3]);

    // 2. Chained filter cost > 60.0 on sel_act -> index [3]
    let sel_cost = table
        .filter_f64_with_sel("cost", ComparisonOp::GreaterThan, 60.0, &sel_act)
        .unwrap();
    assert_eq!(sel_cost.as_slice(), &[3]);

    // 3. Chained filter act != "Reject" on sel_cost -> index [3]
    let sel_dict_compound = table
        .filter_dict_with_sel("act", ComparisonOp::NotEqual, "Reject", &sel_cost)
        .unwrap();
    assert_eq!(sel_dict_compound.as_slice(), &[3]);
}

#[test]
fn test_chunked_vector_processing() {
    let vals: Vec<i64> = (0..2500).map(|i| i as i64).collect();
    let col = Int64Column::from_slice("chunk_col", &vals);

    let chunks: Vec<_> = mini_columnar_engine::vector_chunks(col.len(), 1024).collect();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].num_rows, 1024);
    assert_eq!(chunks[1].num_rows, 1024);
    assert_eq!(chunks[2].num_rows, 452); // 2500 - 2048 = 452

    let mut total_matches = 0;
    for chunk in &chunks {
        let sel = mini_columnar_engine::scan_chunk_filter_i64(
            &col,
            chunk,
            ComparisonOp::GreaterThanOrEqual,
            2000,
        );
        total_matches += sel.len();
    }
    assert_eq!(total_matches, 500); // 2000..2500
}

#[test]
fn test_dictionary_range_filtering() {
    let col = DictionaryColumn::from_iter("code", ["B", "D", "A", "C", "E"]);

    let sel_gt = mini_columnar_engine::scan_filter_dict(&col, ComparisonOp::GreaterThan, "B");
    assert_eq!(sel_gt.as_slice(), &[1, 3, 4]); // D, C, E

    let sel_lte = mini_columnar_engine::scan_filter_dict(&col, ComparisonOp::LessThanOrEqual, "B");
    assert_eq!(sel_lte.as_slice(), &[0, 2]); // B, A
}

#[test]
fn test_group_by_filtered_does_not_emit_phantom_groups() {
    let categories = vec!["A", "B", "C", "A", "B", "C"];
    let amounts = vec![10i64, 20, 30, 40, 50, 60];

    let dict_col = DictionaryColumn::from_iter("cat", &categories);
    let val_col = Int64Column::from_slice("amt", &amounts);

    // Filter to only rows where amount < 25 (rows 0: A=10, 1: B=20). Category C has NO matching rows!
    let sel = SelectionVector::from_indices(vec![0, 1]);

    let results = group_by_dict_sum_i64(&dict_col, &val_col, Some(&sel));
    let map: std::collections::HashMap<String, i64> = results.into_iter().collect();

    assert_eq!(
        map.len(),
        2,
        "Phantom group 'C' must not be emitted when it has 0 rows in selection"
    );
    assert_eq!(map.get("A"), Some(&10));
    assert_eq!(map.get("B"), Some(&20));
    assert_eq!(map.get("C"), None);
}

#[test]
fn test_group_by_count_excludes_null_values() {
    let categories = vec!["A", "A", "A", "B", "B"];
    let amounts = vec![Some(10i64), None, Some(30), None, None]; // A has 2 valid, B has 0 valid

    let mut table = ColumnTable::new("t");
    let dict_col = DictionaryColumn::from_iter("cat", &categories);
    let val_col = Int64Column::from_optional_iter("amt", amounts);

    table.add_column(AnyColumn::Dictionary(dict_col)).unwrap();
    table.add_column(AnyColumn::Int64(val_col)).unwrap();

    // 1. COUNT(amt) - must ignore NULL values in amt
    let count_col_res = table
        .group_by("cat", "amt", AggregateOp::Count, None)
        .unwrap();
    let map_col: std::collections::HashMap<String, usize> = count_col_res
        .into_iter()
        .map(|(k, v)| match v {
            AggregateValue::Count(c) => (k, c),
            _ => unreachable!(),
        })
        .collect();

    assert_eq!(map_col.get("A"), Some(&2));
    assert_eq!(
        map_col.get("B"),
        None,
        "Category B has all nulls for amt so COUNT(amt) is 0 and group is pruned"
    );

    // 2. COUNT(*) - must count all rows for each group regardless of amt nulls
    let count_star_res = table.group_by_count_star("cat", None).unwrap();
    let map_star: std::collections::HashMap<String, usize> = count_star_res.into_iter().collect();

    assert_eq!(map_star.get("A"), Some(&3));
    assert_eq!(map_star.get("B"), Some(&2));
}

#[test]
fn test_table_aggregate_empty_returns_null() {
    let mut table = ColumnTable::new("empty_table");
    let empty_i = Int64Column::new("i");
    let empty_f = Float64Column::new("f");
    table.add_column(AnyColumn::Int64(empty_i)).unwrap();
    table.add_column(AnyColumn::Float64(empty_f)).unwrap();

    assert_eq!(
        table.aggregate("i", AggregateOp::Sum, None).unwrap(),
        AggregateValue::Null
    );
    assert_eq!(
        table.aggregate("f", AggregateOp::Sum, None).unwrap(),
        AggregateValue::Null
    );
    assert_eq!(
        table.aggregate("i", AggregateOp::Avg, None).unwrap(),
        AggregateValue::Null
    );
}

#[test]
fn test_zero_allocation_filter_into_and_selection_vector_pool() {
    let mut pool = mini_columnar_engine::SelectionVectorPool::new();
    let mut sel = pool.acquire(1024);

    let vals: Vec<i64> = (0..10_000).collect();
    let col = Int64Column::from_slice("v", &vals);

    mini_columnar_engine::scan_filter_i64_into(&col, ComparisonOp::LessThan, 50, &mut sel);
    assert_eq!(sel.len(), 50);

    let mut sel2 = pool.acquire(1024);
    mini_columnar_engine::scan_filter_i64_with_selection_into(
        &col,
        &sel,
        ComparisonOp::GreaterThanOrEqual,
        40,
        &mut sel2,
    );
    assert_eq!(sel2.len(), 10);
    assert_eq!(sel2.as_slice(), &[40, 41, 42, 43, 44, 45, 46, 47, 48, 49]);

    pool.release(sel);
    pool.release(sel2);
}

#[test]
fn test_l1_morsel_chunked_pipelined_aggregation() {
    let mut table = ColumnTable::new("chunk_pipeline");
    let filter_vals: Vec<i64> = (0..5000).collect();
    let data_vals: Vec<i64> = (0..5000).map(|i| i * 2).collect();

    table
        .add_column(AnyColumn::Int64(Int64Column::from_slice("f", &filter_vals)))
        .unwrap();
    table
        .add_column(AnyColumn::Int64(Int64Column::from_slice("d", &data_vals)))
        .unwrap();

    // Condition: f >= 2000 AND f < 4000 (2000 rows, d values are 4000..8000 (step 2))
    // Standard filtered execution
    let sel = table
        .filter_i64("f", ComparisonOp::GreaterThanOrEqual, 2000)
        .unwrap();
    let agg_sum_standard = table.aggregate("d", AggregateOp::Sum, Some(&sel)).unwrap();

    // Chunked pipelined execution
    let agg_sum_chunked = table
        .aggregate_filtered_chunked_i64(
            "f",
            ComparisonOp::GreaterThanOrEqual,
            2000,
            "d",
            AggregateOp::Sum,
        )
        .unwrap();

    assert_eq!(agg_sum_standard, agg_sum_chunked);
}

#[test]
fn test_dictionary_bits_per_key_and_bit_packed_bytes() {
    let col_binary = DictionaryColumn::from_iter("flag", ["YES", "NO"]);
    assert_eq!(col_binary.bits_per_key(), 1);
    assert_eq!(col_binary.bit_packed_keys_bytes(), 1); // ceil(2 * 1 / 8) = 1

    let col_hex = DictionaryColumn::from_iter(
        "hex",
        [
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F",
        ],
    );
    assert_eq!(col_hex.bits_per_key(), 4);
    assert_eq!(col_hex.bit_packed_keys_bytes(), 8); // ceil(16 * 4 / 8) = 8
}
