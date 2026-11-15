use crate::column::dictionary::DictionaryColumn;
use crate::column::numeric::{Float64Column, Int64Column};

/// Standard vector chunk size (1024 elements) matching CPU L1 data cache sizing.
pub const VECTOR_CHUNK_SIZE: usize = 1024;

/// Predicate comparison operations for filter pushdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
}

impl std::fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComparisonOp::Equal => write!(f, "=="),
            ComparisonOp::NotEqual => write!(f, "!="),
            ComparisonOp::GreaterThan => write!(f, ">"),
            ComparisonOp::GreaterThanOrEqual => write!(f, ">="),
            ComparisonOp::LessThan => write!(f, "<"),
            ComparisonOp::LessThanOrEqual => write!(f, "<="),
        }
    }
}

/// Selection Vector: Compact array of matching row indices.
///
/// In modern vectorized query engines (SAP HANA, MonetDB, DuckDB), filter evaluation
/// produces a selection vector instead of materializing full tuple buffers or intermediate
/// column copies. Downstream operators (aggregators, subsequent filters, group-by)
/// consume this selection vector directly, avoiding expensive row allocation and cache copies.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SelectionVector {
    indices: Vec<usize>,
}

impl SelectionVector {
    /// Create an empty selection vector (zero matching rows).
    pub fn empty() -> Self {
        Self {
            indices: Vec::new(),
        }
    }

    /// Create from a pre-computed vector of row indices (must be in ascending order).
    pub fn from_indices(indices: Vec<usize>) -> Self {
        Self { indices }
    }

    /// Create an all-matching selection vector representing `0..total_rows`.
    pub fn all(total_rows: usize) -> Self {
        Self {
            indices: (0..total_rows).collect(),
        }
    }

    /// Number of matching rows in the selection vector.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// True if no rows matched the filter.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Read-only slice of matching row indices.
    #[inline(always)]
    pub fn as_slice(&self) -> &[usize] {
        &self.indices
    }

    /// Calculate filter selectivity percentage (0.0 to 1.0).
    pub fn selectivity(&self, total_rows: usize) -> f64 {
        if total_rows == 0 {
            0.0
        } else {
            self.indices.len() as f64 / total_rows as f64
        }
    }

    /// Fast sorted two-pointer intersection (logical AND between two selection vectors).
    pub fn and(&self, other: &Self) -> Self {
        let mut result = Vec::with_capacity(self.indices.len().min(other.indices.len()));
        let mut i = 0;
        let mut j = 0;

        let a = &self.indices;
        let b = &other.indices;

        while i < a.len() && j < b.len() {
            if a[i] == b[j] {
                result.push(a[i]);
                i += 1;
                j += 1;
            } else if a[i] < b[j] {
                i += 1;
            } else {
                j += 1;
            }
        }

        Self { indices: result }
    }

    /// Fast sorted two-pointer union (logical OR between two selection vectors).
    pub fn or(&self, other: &Self) -> Self {
        let mut result = Vec::with_capacity(self.indices.len() + other.indices.len());
        let mut i = 0;
        let mut j = 0;

        let a = &self.indices;
        let b = &other.indices;

        while i < a.len() && j < b.len() {
            if a[i] == b[j] {
                result.push(a[i]);
                i += 1;
                j += 1;
            } else if a[i] < b[j] {
                result.push(a[i]);
                i += 1;
            } else {
                result.push(b[j]);
                j += 1;
            }
        }

        while i < a.len() {
            result.push(a[i]);
            i += 1;
        }

        while j < b.len() {
            result.push(b[j]);
            j += 1;
        }

        Self { indices: result }
    }
}

/// Vectorized filter pushdown operator for signed 64-bit integer columns (`Int64Column`).
///
/// Evaluates predicates in fixed-size cache-friendly vector chunks with loop unrolling,
/// generating a `SelectionVector` without materializing intermediate data.
pub fn scan_filter_i64(col: &Int64Column, op: ComparisonOp, target: i64) -> SelectionVector {
    let data = col.as_slice();
    let total_rows = data.len();
    let mut matching_indices = Vec::with_capacity(total_rows / 4);

    if let Some(bm) = col.null_bitmap() {
        // Scans with null bitmap check
        for (i, &val) in data.iter().enumerate() {
            if bm.is_valid(i) && evaluate_i64(val, op, target) {
                matching_indices.push(i);
            }
        }
    } else {
        // Vectorized tight loop over contiguous primitive slice (L1 cache prefetching & SIMD)
        let chunks = total_rows / 8;
        let remainder_start = chunks * 8;

        for chunk_idx in 0..chunks {
            let base = chunk_idx * 8;
            // 8-way unrolled evaluation to maximize CPU instruction-level parallelism
            if evaluate_i64(data[base], op, target) {
                matching_indices.push(base);
            }
            if evaluate_i64(data[base + 1], op, target) {
                matching_indices.push(base + 1);
            }
            if evaluate_i64(data[base + 2], op, target) {
                matching_indices.push(base + 2);
            }
            if evaluate_i64(data[base + 3], op, target) {
                matching_indices.push(base + 3);
            }
            if evaluate_i64(data[base + 4], op, target) {
                matching_indices.push(base + 4);
            }
            if evaluate_i64(data[base + 5], op, target) {
                matching_indices.push(base + 5);
            }
            if evaluate_i64(data[base + 6], op, target) {
                matching_indices.push(base + 6);
            }
            if evaluate_i64(data[base + 7], op, target) {
                matching_indices.push(base + 7);
            }
        }

        for (i, &val) in data.iter().enumerate().skip(remainder_start) {
            if evaluate_i64(val, op, target) {
                matching_indices.push(i);
            }
        }
    }

    SelectionVector::from_indices(matching_indices)
}

/// Filter pushdown applied on an existing `SelectionVector` (compound predicate pushdown).
pub fn scan_filter_i64_with_selection(
    col: &Int64Column,
    sel: &SelectionVector,
    op: ComparisonOp,
    target: i64,
) -> SelectionVector {
    let data = col.as_slice();
    let mut filtered = Vec::with_capacity(sel.len());

    if let Some(bm) = col.null_bitmap() {
        for &idx in sel.as_slice() {
            if bm.is_valid(idx) && evaluate_i64(data[idx], op, target) {
                filtered.push(idx);
            }
        }
    } else {
        for &idx in sel.as_slice() {
            if evaluate_i64(data[idx], op, target) {
                filtered.push(idx);
            }
        }
    }

    SelectionVector::from_indices(filtered)
}

/// Vectorized filter pushdown operator for 64-bit float columns (`Float64Column`).
pub fn scan_filter_f64(col: &Float64Column, op: ComparisonOp, target: f64) -> SelectionVector {
    let data = col.as_slice();
    let total_rows = data.len();
    let mut matching_indices = Vec::with_capacity(total_rows / 4);

    if let Some(bm) = col.null_bitmap() {
        for (i, &val) in data.iter().enumerate() {
            if bm.is_valid(i) && evaluate_f64(val, op, target) {
                matching_indices.push(i);
            }
        }
    } else {
        let chunks = total_rows / 8;
        let remainder_start = chunks * 8;

        for chunk_idx in 0..chunks {
            let base = chunk_idx * 8;
            if evaluate_f64(data[base], op, target) {
                matching_indices.push(base);
            }
            if evaluate_f64(data[base + 1], op, target) {
                matching_indices.push(base + 1);
            }
            if evaluate_f64(data[base + 2], op, target) {
                matching_indices.push(base + 2);
            }
            if evaluate_f64(data[base + 3], op, target) {
                matching_indices.push(base + 3);
            }
            if evaluate_f64(data[base + 4], op, target) {
                matching_indices.push(base + 4);
            }
            if evaluate_f64(data[base + 5], op, target) {
                matching_indices.push(base + 5);
            }
            if evaluate_f64(data[base + 6], op, target) {
                matching_indices.push(base + 6);
            }
            if evaluate_f64(data[base + 7], op, target) {
                matching_indices.push(base + 7);
            }
        }

        for (i, &val) in data.iter().enumerate().skip(remainder_start) {
            if evaluate_f64(val, op, target) {
                matching_indices.push(i);
            }
        }
    }

    SelectionVector::from_indices(matching_indices)
}

/// Vectorized filter pushdown on `DictionaryColumn`.
///
/// Translates string filtering (`activity == "Approve PO"`) into a single O(1) dictionary
/// lookup to acquire `target_id: u32`, followed by an ultra-fast contiguous 32-bit integer scan.
pub fn scan_filter_dict(col: &DictionaryColumn, op: ComparisonOp, target: &str) -> SelectionVector {
    let target_id_opt = col.lookup_id(target);

    match op {
        ComparisonOp::Equal => {
            let target_id = match target_id_opt {
                Some(id) => id,
                None => return SelectionVector::empty(), // Instant pruning: string doesn't exist
            };

            let keys = col.keys();
            let total_rows = keys.len();
            let mut matching_indices = Vec::with_capacity(total_rows / 4);

            if let Some(bm) = col.null_bitmap() {
                for (i, &key) in keys.iter().enumerate() {
                    if bm.is_valid(i) && key == target_id {
                        matching_indices.push(i);
                    }
                }
            } else {
                let chunks = total_rows / 8;
                let remainder_start = chunks * 8;

                for chunk_idx in 0..chunks {
                    let base = chunk_idx * 8;
                    if keys[base] == target_id {
                        matching_indices.push(base);
                    }
                    if keys[base + 1] == target_id {
                        matching_indices.push(base + 1);
                    }
                    if keys[base + 2] == target_id {
                        matching_indices.push(base + 2);
                    }
                    if keys[base + 3] == target_id {
                        matching_indices.push(base + 3);
                    }
                    if keys[base + 4] == target_id {
                        matching_indices.push(base + 4);
                    }
                    if keys[base + 5] == target_id {
                        matching_indices.push(base + 5);
                    }
                    if keys[base + 6] == target_id {
                        matching_indices.push(base + 6);
                    }
                    if keys[base + 7] == target_id {
                        matching_indices.push(base + 7);
                    }
                }

                for (i, &k) in keys.iter().enumerate().skip(remainder_start) {
                    if k == target_id {
                        matching_indices.push(i);
                    }
                }
            }

            SelectionVector::from_indices(matching_indices)
        }
        ComparisonOp::NotEqual => {
            let keys = col.keys();
            let total_rows = keys.len();
            let mut matching_indices = Vec::with_capacity(total_rows);

            match target_id_opt {
                None => {
                    // All non-null rows match
                    if let Some(bm) = col.null_bitmap() {
                        for i in 0..total_rows {
                            if bm.is_valid(i) {
                                matching_indices.push(i);
                            }
                        }
                    } else {
                        return SelectionVector::all(total_rows);
                    }
                }
                Some(target_id) => {
                    if let Some(bm) = col.null_bitmap() {
                        for (i, &key) in keys.iter().enumerate() {
                            if bm.is_valid(i) && key != target_id {
                                matching_indices.push(i);
                            }
                        }
                    } else {
                        for (i, &key) in keys.iter().enumerate() {
                            if key != target_id {
                                matching_indices.push(i);
                            }
                        }
                    }
                }
            }

            SelectionVector::from_indices(matching_indices)
        }
        _ => {
            // Lexicographical comparisons against string target
            let dict = col.dictionary();
            let mut matching_dict_ids = std::collections::HashSet::new();

            for (id, val) in dict.iter().enumerate() {
                let matches = match op {
                    ComparisonOp::GreaterThan => val.as_str() > target,
                    ComparisonOp::GreaterThanOrEqual => val.as_str() >= target,
                    ComparisonOp::LessThan => val.as_str() < target,
                    ComparisonOp::LessThanOrEqual => val.as_str() <= target,
                    _ => unreachable!(),
                };
                if matches {
                    matching_dict_ids.insert(id as u32);
                }
            }

            let keys = col.keys();
            let mut matching_indices = Vec::new();
            if let Some(bm) = col.null_bitmap() {
                for (i, &key) in keys.iter().enumerate() {
                    if bm.is_valid(i) && matching_dict_ids.contains(&key) {
                        matching_indices.push(i);
                    }
                }
            } else {
                for (i, &key) in keys.iter().enumerate() {
                    if matching_dict_ids.contains(&key) {
                        matching_indices.push(i);
                    }
                }
            }

            SelectionVector::from_indices(matching_indices)
        }
    }
}

#[inline(always)]
fn evaluate_i64(val: i64, op: ComparisonOp, target: i64) -> bool {
    match op {
        ComparisonOp::Equal => val == target,
        ComparisonOp::NotEqual => val != target,
        ComparisonOp::GreaterThan => val > target,
        ComparisonOp::GreaterThanOrEqual => val >= target,
        ComparisonOp::LessThan => val < target,
        ComparisonOp::LessThanOrEqual => val <= target,
    }
}

#[inline(always)]
fn evaluate_f64(val: f64, op: ComparisonOp, target: f64) -> bool {
    match op {
        ComparisonOp::Equal => val == target,
        ComparisonOp::NotEqual => val != target,
        ComparisonOp::GreaterThan => val > target,
        ComparisonOp::GreaterThanOrEqual => val >= target,
        ComparisonOp::LessThan => val < target,
        ComparisonOp::LessThanOrEqual => val <= target,
    }
}
