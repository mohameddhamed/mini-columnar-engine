//! # mini-columnar-engine
//!
//! A lean, production-grade vectorized in-memory OLAP micro-database engine in Rust,
//! architected to demonstrate the core database internals powering SAP HANA and Celonis PQL:
//!
//! 1. **Contiguous Columnar Memory Layout:** Cache-line aligned primitive buffers (`Vec<i64>`, `Vec<f64>`)
//!    and bit-packed null masks (`NullBitmap`) enabling hardware prefetching and zero pointer chasing.
//! 2. **Dictionary Compression:** Low-cardinality string pool storage with compact 32-bit surrogate integer keys,
//!    enabling instant predicate evaluation without string decoding.
//! 3. **Run-Length Encoding (RLE):** Compression for consecutive repeated values with O(runs) mathematical aggregations.
//! 4. **Vectorized Filter Pushdown:** Zero-copy `SelectionVector` operators replacing row-by-row Volcano iterators.
//! 5. **Vectorized Analytical Aggregations & Group By:** Direct-indexed accumulator updates over dictionary keys,
//!    bypassing hash tables entirely with 100% L1 cache locality.

pub mod column;
pub mod execution;
pub mod table;

pub use column::bitmap::NullBitmap;
pub use column::dictionary::DictionaryColumn;
pub use column::numeric::{Float64Column, Int64Column};
pub use column::rle::RleColumn;
pub use column::string::StringColumn;
pub use column::{Column, ColumnType};

pub use execution::aggregate::{
    avg_f64, avg_i64, group_by_dict_count, group_by_dict_sum_f64, group_by_dict_sum_i64,
    group_by_raw_string_sum_i64, max_f64, max_i64, min_f64, min_i64, sum_f64, sum_i64, AggregateOp,
    AggregateValue,
};
pub use execution::filter::{
    scan_filter_dict, scan_filter_f64, scan_filter_i64, scan_filter_i64_with_selection,
    ComparisonOp, SelectionVector, VECTOR_CHUNK_SIZE,
};

pub use table::{AnyColumn, ColumnDef, ColumnTable};
