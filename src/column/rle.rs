use super::bitmap::NullBitmap;
use super::{Column, ColumnType};

/// Run-Length Encoded (RLE) column for highly repeated consecutive values.
///
/// In OLAP engines (e.g. SAP HANA, Snowflake, ClickHouse), sorted columns (such as
/// timestamps, status flags, or partition keys) exhibit long contiguous runs of identical values.
/// RLE collapses `N` repeated entries into a single `(value, count)` tuple, dramatically
/// reducing memory bandwidth and enabling mathematical aggregations in O(runs) rather than O(rows).
#[derive(Debug, Clone, PartialEq)]
pub struct RleColumn<T: Clone + PartialEq> {
    name: String,
    values: Vec<T>,
    counts: Vec<u32>,
    cumulative_counts: Vec<usize>,
    total_len: usize,
}

impl<T: Clone + PartialEq> RleColumn<T> {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            values: Vec::new(),
            counts: Vec::new(),
            cumulative_counts: Vec::new(),
            total_len: 0,
        }
    }

    pub fn from_slice(name: impl Into<String>, slice: &[T]) -> Self {
        let mut col = Self::new(name);
        for item in slice {
            col.push(item.clone());
        }
        col
    }

    /// Push a value, coalescing it with the current active run if equal.
    pub fn push(&mut self, val: T) {
        if let Some(last_val) = self.values.last() {
            if *last_val == val {
                let last_idx = self.counts.len() - 1;
                self.counts[last_idx] += 1;
                self.cumulative_counts[last_idx] += 1;
                self.total_len += 1;
                return;
            }
        }

        self.values.push(val);
        self.counts.push(1);
        let cum = self.total_len + 1;
        self.cumulative_counts.push(cum);
        self.total_len += 1;
    }

    /// Number of distinct runs.
    pub fn num_runs(&self) -> usize {
        self.values.len()
    }

    /// Total number of logical rows represented.
    pub fn len(&self) -> usize {
        self.total_len
    }

    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    /// Fetch value at logical row `idx` using binary search over cumulative run boundaries.
    pub fn get(&self, idx: usize) -> Option<&T> {
        if idx >= self.total_len {
            return None;
        }
        let run_idx = match self.cumulative_counts.binary_search(&idx) {
            Ok(exact) => exact + 1,
            Err(insertion) => insertion,
        };
        self.values.get(run_idx)
    }

    /// Access the underlying runs directly.
    pub fn runs(&self) -> (&[T], &[u32]) {
        (&self.values, &self.counts)
    }

    /// Materialize back into a flat `Vec<T>`.
    pub fn decompress(&self) -> Vec<T> {
        let mut flat = Vec::with_capacity(self.total_len);
        for (val, &count) in self.values.iter().zip(self.counts.iter()) {
            for _ in 0..count {
                flat.push(val.clone());
            }
        }
        flat
    }
}

impl RleColumn<i64> {
    /// Vectorized/algebraic SUM on RLE-compressed i64 data.
    /// Runs in O(num_runs) instead of O(total_len) using multiplication!
    pub fn sum(&self) -> i64 {
        let mut total = 0i64;
        for (&val, &cnt) in self.values.iter().zip(self.counts.iter()) {
            total += val * (cnt as i64);
        }
        total
    }

    /// O(runs) MIN over RLE-compressed i64 data.
    pub fn min(&self) -> Option<i64> {
        self.values.iter().copied().min()
    }

    /// O(runs) MAX over RLE-compressed i64 data.
    pub fn max(&self) -> Option<i64> {
        self.values.iter().copied().max()
    }

    /// O(runs) AVG over RLE-compressed i64 data.
    pub fn avg(&self) -> Option<f64> {
        if self.total_len == 0 {
            None
        } else {
            Some(self.sum() as f64 / self.total_len as f64)
        }
    }
}

impl Column for RleColumn<i64> {
    fn name(&self) -> &str {
        &self.name
    }

    fn len(&self) -> usize {
        self.total_len
    }

    fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    fn column_type(&self) -> ColumnType {
        ColumnType::Int64
    }

    fn null_bitmap(&self) -> Option<&NullBitmap> {
        None
    }

    fn is_null(&self, _idx: usize) -> bool {
        false
    }

    fn memory_bytes(&self) -> usize {
        (self.values.capacity() * std::mem::size_of::<i64>())
            + (self.counts.capacity() * std::mem::size_of::<u32>())
            + (self.cumulative_counts.capacity() * std::mem::size_of::<usize>())
            + self.name.capacity()
    }
}
