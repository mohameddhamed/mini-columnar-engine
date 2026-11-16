use crate::column::dictionary::DictionaryColumn;
use crate::column::numeric::{Float64Column, Int64Column};
use crate::column::Column;

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

    /// Pre-allocate buffer capacity to avoid dynamic heap reallocations.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            indices: Vec::with_capacity(capacity),
        }
    }

    /// Clear all matching indices, retaining backing memory buffer for reuse.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.indices.clear();
    }

    /// Push a matching row index into the vector.
    #[inline(always)]
    pub fn push(&mut self, idx: usize) {
        self.indices.push(idx);
    }

    /// In-place predicate filtering without allocating intermediate memory.
    /// Crucial for zero-allocation chained filter pushdowns.
    #[inline(always)]
    pub fn filter_in_place<F: Fn(usize) -> bool>(&mut self, predicate: F) {
        self.indices.retain(|&idx| predicate(idx));
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

    /// Fast sorted two-pointer intersection writing to pre-allocated `out`.
    pub fn and_into(&self, other: &Self, out: &mut Self) {
        out.clear();
        let a = &self.indices;
        let b = &other.indices;
        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            if a[i] == b[j] {
                out.push(a[i]);
                i += 1;
                j += 1;
            } else if a[i] < b[j] {
                i += 1;
            } else {
                j += 1;
            }
        }
    }

    /// Fast sorted two-pointer intersection (logical AND between two selection vectors).
    pub fn and(&self, other: &Self) -> Self {
        let mut out = Self::with_capacity(self.indices.len().min(other.indices.len()));
        self.and_into(other, &mut out);
        out
    }

    /// Fast sorted two-pointer union writing to pre-allocated `out`.
    pub fn or_into(&self, other: &Self, out: &mut Self) {
        out.clear();
        let a = &self.indices;
        let b = &other.indices;
        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            if a[i] == b[j] {
                out.push(a[i]);
                i += 1;
                j += 1;
            } else if a[i] < b[j] {
                out.push(a[i]);
                i += 1;
            } else {
                out.push(b[j]);
                j += 1;
            }
        }

        while i < a.len() {
            out.push(a[i]);
            i += 1;
        }

        while j < b.len() {
            out.push(b[j]);
            j += 1;
        }
    }

    /// Fast sorted two-pointer union (logical OR between two selection vectors).
    pub fn or(&self, other: &Self) -> Self {
        let mut out = Self::with_capacity(self.indices.len() + other.indices.len());
        self.or_into(other, &mut out);
        out
    }

    /// Convert selection vector into a bit-packed `NullBitmap`.
    pub fn to_bitmap(&self, total_rows: usize) -> crate::column::bitmap::NullBitmap {
        let mut bm = crate::column::bitmap::NullBitmap::new(total_rows, false);
        for &idx in &self.indices {
            if idx < total_rows {
                bm.set_valid(idx);
            }
        }
        bm
    }

    /// Construct a selection vector from a `NullBitmap` (1-bits become indices).
    pub fn from_bitmap(bm: &crate::column::bitmap::NullBitmap) -> Self {
        Self {
            indices: bm.to_indices(),
        }
    }

    /// Invert selection vector (NOT logic) relative to `total_rows` into pre-allocated `out`.
    pub fn not_into(&self, total_rows: usize, out: &mut Self) {
        out.clear();
        let mut cur = 0;
        let mut sel_idx = 0;
        let sel = &self.indices;

        while cur < total_rows {
            if sel_idx < sel.len() && sel[sel_idx] == cur {
                sel_idx += 1;
            } else {
                out.push(cur);
            }
            cur += 1;
        }
    }

    /// Invert selection vector (NOT logic) relative to `total_rows`.
    pub fn not(&self, total_rows: usize) -> Self {
        let mut out = Self::with_capacity(total_rows.saturating_sub(self.indices.len()));
        self.not_into(total_rows, &mut out);
        out
    }
}

/// Scratch buffer pool for zero-allocation SelectionVector reuse across query operators.
#[derive(Debug, Default)]
pub struct SelectionVectorPool {
    pool: Vec<SelectionVector>,
}

impl SelectionVectorPool {
    pub fn new() -> Self {
        Self { pool: Vec::new() }
    }

    pub fn acquire(&mut self, capacity: usize) -> SelectionVector {
        match self.pool.pop() {
            Some(mut sv) => {
                sv.clear();
                if sv.indices.capacity() < capacity {
                    sv.indices.reserve(capacity - sv.indices.capacity());
                }
                sv
            }
            None => SelectionVector::with_capacity(capacity),
        }
    }

    pub fn release(&mut self, mut sv: SelectionVector) {
        sv.clear();
        self.pool.push(sv);
    }
}

/// Vector Batch: fixed-size chunk of rows (default `VECTOR_CHUNK_SIZE = 1024`) processed together
/// to ensure operands remain 100% resident in CPU L1 data cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorBatch {
    pub start_row: usize,
    pub num_rows: usize,
}

impl VectorBatch {
    pub fn new(start_row: usize, num_rows: usize) -> Self {
        Self {
            start_row,
            num_rows,
        }
    }

    /// End row index (exclusive).
    #[inline(always)]
    pub fn end_row(&self) -> usize {
        self.start_row + self.num_rows
    }
}

/// Create an iterator over vector chunks (each of size `chunk_size`, e.g. `VECTOR_CHUNK_SIZE`).
pub fn vector_chunks(total_rows: usize, chunk_size: usize) -> impl Iterator<Item = VectorBatch> {
    let num_chunks = total_rows.div_ceil(chunk_size);
    (0..num_chunks).map(move |i| {
        let start = i * chunk_size;
        let count = (total_rows - start).min(chunk_size);
        VectorBatch::new(start, count)
    })
}

/// Vectorized chunk-level filter evaluation on `Int64Column` writing into pre-allocated `out`.
pub fn scan_chunk_filter_i64_into(
    col: &Int64Column,
    batch: &VectorBatch,
    op: ComparisonOp,
    target: i64,
    out: &mut SelectionVector,
) {
    let data = col.as_slice();
    let start = batch.start_row;
    let end = (start + batch.num_rows).min(data.len());
    out.clear();
    out.indices
        .reserve((batch.num_rows / 4).saturating_sub(out.indices.capacity()));

    if let Some(bm) = col.null_bitmap() {
        for (offset, &val) in data[start..end].iter().enumerate() {
            let i = start + offset;
            if bm.is_valid(i) && evaluate_i64(val, op, target) {
                out.push(i);
            }
        }
    } else {
        for (offset, &val) in data[start..end].iter().enumerate() {
            let i = start + offset;
            if evaluate_i64(val, op, target) {
                out.push(i);
            }
        }
    }
}

/// Vectorized chunk-level filter evaluation on `Int64Column`.
pub fn scan_chunk_filter_i64(
    col: &Int64Column,
    batch: &VectorBatch,
    op: ComparisonOp,
    target: i64,
) -> SelectionVector {
    let mut out = SelectionVector::with_capacity(batch.num_rows / 4);
    scan_chunk_filter_i64_into(col, batch, op, target, &mut out);
    out
}

/// Vectorized filter pushdown operator for signed 64-bit integer columns (`Int64Column`)
/// writing matching row offsets into pre-allocated `out`.
pub fn scan_filter_i64_into(
    col: &Int64Column,
    op: ComparisonOp,
    target: i64,
    out: &mut SelectionVector,
) {
    let data = col.as_slice();
    let total_rows = data.len();
    out.clear();
    out.indices
        .reserve((total_rows / 4).saturating_sub(out.indices.capacity()));

    if let Some(bm) = col.null_bitmap() {
        // Bit-parallel null skipping: skip 64-row null words in a single clock cycle
        let words = bm.words();
        for (w_idx, &word) in words.iter().enumerate() {
            if word == 0 {
                continue; // 64 consecutive NULL rows skipped instantaneously
            }
            let base = w_idx * 64;
            let end = (base + 64).min(total_rows);
            if word == u64::MAX && end == base + 64 {
                // All 64 rows valid: 8-way unrolled scan without bitmap checks
                for chunk_idx in 0..8 {
                    let b = base + chunk_idx * 8;
                    let slice: &[i64; 8] = match data[b..b + 8].try_into() {
                        Ok(s) => s,
                        Err(_) => unreachable!(),
                    };
                    for (lane, &val) in slice.iter().enumerate() {
                        if evaluate_i64(val, op, target) {
                            out.push(b + lane);
                        }
                    }
                }
            } else {
                for (bit_offset, &val) in data[base..end].iter().enumerate() {
                    if (word & (1u64 << bit_offset)) != 0 && evaluate_i64(val, op, target) {
                        out.push(base + bit_offset);
                    }
                }
            }
        }
    } else {
        let chunks = total_rows / 8;
        let remainder_start = chunks * 8;

        for chunk_idx in 0..chunks {
            let base = chunk_idx * 8;
            let slice: &[i64; 8] = match data[base..base + 8].try_into() {
                Ok(s) => s,
                Err(_) => unreachable!(),
            };
            for (lane, &val) in slice.iter().enumerate() {
                if evaluate_i64(val, op, target) {
                    out.push(base + lane);
                }
            }
        }

        for (i, &val) in data.iter().enumerate().skip(remainder_start) {
            if evaluate_i64(val, op, target) {
                out.push(i);
            }
        }
    }
}

/// Vectorized filter pushdown operator for signed 64-bit integer columns (`Int64Column`).
pub fn scan_filter_i64(col: &Int64Column, op: ComparisonOp, target: i64) -> SelectionVector {
    let mut out = SelectionVector::with_capacity(col.len() / 4);
    scan_filter_i64_into(col, op, target, &mut out);
    out
}

/// Filter pushdown applied on an existing `SelectionVector` writing to `out`.
pub fn scan_filter_i64_with_selection_into(
    col: &Int64Column,
    sel: &SelectionVector,
    op: ComparisonOp,
    target: i64,
    out: &mut SelectionVector,
) {
    let data = col.as_slice();
    out.clear();
    out.indices
        .reserve(sel.len().saturating_sub(out.indices.capacity()));

    if let Some(bm) = col.null_bitmap() {
        for &idx in sel.as_slice() {
            if bm.is_valid(idx) && evaluate_i64(data[idx], op, target) {
                out.push(idx);
            }
        }
    } else {
        for &idx in sel.as_slice() {
            if evaluate_i64(data[idx], op, target) {
                out.push(idx);
            }
        }
    }
}

/// Filter pushdown applied on an existing `SelectionVector` (compound predicate pushdown).
pub fn scan_filter_i64_with_selection(
    col: &Int64Column,
    sel: &SelectionVector,
    op: ComparisonOp,
    target: i64,
) -> SelectionVector {
    let mut out = SelectionVector::with_capacity(sel.len());
    scan_filter_i64_with_selection_into(col, sel, op, target, &mut out);
    out
}

/// Vectorized filter pushdown operator for 64-bit float columns (`Float64Column`) writing to `out`.
pub fn scan_filter_f64_into(
    col: &Float64Column,
    op: ComparisonOp,
    target: f64,
    out: &mut SelectionVector,
) {
    let data = col.as_slice();
    let total_rows = data.len();
    out.clear();
    out.indices
        .reserve((total_rows / 4).saturating_sub(out.indices.capacity()));

    if let Some(bm) = col.null_bitmap() {
        let words = bm.words();
        for (w_idx, &word) in words.iter().enumerate() {
            if word == 0 {
                continue;
            }
            let base = w_idx * 64;
            let end = (base + 64).min(total_rows);
            if word == u64::MAX && end == base + 64 {
                for chunk_idx in 0..8 {
                    let b = base + chunk_idx * 8;
                    let slice: &[f64; 8] = match data[b..b + 8].try_into() {
                        Ok(s) => s,
                        Err(_) => unreachable!(),
                    };
                    for (lane, &val) in slice.iter().enumerate() {
                        if evaluate_f64(val, op, target) {
                            out.push(b + lane);
                        }
                    }
                }
            } else {
                for (bit_offset, &val) in data[base..end].iter().enumerate() {
                    if (word & (1u64 << bit_offset)) != 0 && evaluate_f64(val, op, target) {
                        out.push(base + bit_offset);
                    }
                }
            }
        }
    } else {
        let chunks = total_rows / 8;
        let remainder_start = chunks * 8;

        for chunk_idx in 0..chunks {
            let base = chunk_idx * 8;
            let slice: &[f64; 8] = match data[base..base + 8].try_into() {
                Ok(s) => s,
                Err(_) => unreachable!(),
            };
            for (lane, &val) in slice.iter().enumerate() {
                if evaluate_f64(val, op, target) {
                    out.push(base + lane);
                }
            }
        }

        for (i, &val) in data.iter().enumerate().skip(remainder_start) {
            if evaluate_f64(val, op, target) {
                out.push(i);
            }
        }
    }
}

/// Vectorized filter pushdown operator for 64-bit float columns (`Float64Column`).
pub fn scan_filter_f64(col: &Float64Column, op: ComparisonOp, target: f64) -> SelectionVector {
    let mut out = SelectionVector::with_capacity(col.len() / 4);
    scan_filter_f64_into(col, op, target, &mut out);
    out
}

/// Filter pushdown applied on an existing `SelectionVector` for `Float64Column` writing to `out`.
pub fn scan_filter_f64_with_selection_into(
    col: &Float64Column,
    sel: &SelectionVector,
    op: ComparisonOp,
    target: f64,
    out: &mut SelectionVector,
) {
    let data = col.as_slice();
    out.clear();
    out.indices
        .reserve(sel.len().saturating_sub(out.indices.capacity()));

    if let Some(bm) = col.null_bitmap() {
        for &idx in sel.as_slice() {
            if bm.is_valid(idx) && evaluate_f64(data[idx], op, target) {
                out.push(idx);
            }
        }
    } else {
        for &idx in sel.as_slice() {
            if evaluate_f64(data[idx], op, target) {
                out.push(idx);
            }
        }
    }
}

/// Filter pushdown applied on an existing `SelectionVector` for `Float64Column`.
pub fn scan_filter_f64_with_selection(
    col: &Float64Column,
    sel: &SelectionVector,
    op: ComparisonOp,
    target: f64,
) -> SelectionVector {
    let mut out = SelectionVector::with_capacity(sel.len());
    scan_filter_f64_with_selection_into(col, sel, op, target, &mut out);
    out
}

/// Vectorized filter pushdown on `DictionaryColumn` writing matching rows to `out`.
pub fn scan_filter_dict_into(
    col: &DictionaryColumn,
    op: ComparisonOp,
    target: &str,
    out: &mut SelectionVector,
) {
    let target_id_opt = col.lookup_id(target);
    out.clear();

    match op {
        ComparisonOp::Equal => {
            let target_id = match target_id_opt {
                Some(id) => id,
                None => return, // Instant pruning: string doesn't exist
            };

            let keys = col.keys();
            let total_rows = keys.len();
            out.indices
                .reserve((total_rows / 4).saturating_sub(out.indices.capacity()));

            if let Some(bm) = col.null_bitmap() {
                let words = bm.words();
                for (w_idx, &word) in words.iter().enumerate() {
                    if word == 0 {
                        continue;
                    }
                    let base = w_idx * 64;
                    let end = (base + 64).min(total_rows);
                    if word == u64::MAX && end == base + 64 {
                        for chunk_idx in 0..8 {
                            let b = base + chunk_idx * 8;
                            let slice: &[u32; 8] = match keys[b..b + 8].try_into() {
                                Ok(s) => s,
                                Err(_) => unreachable!(),
                            };
                            for (lane, &k) in slice.iter().enumerate() {
                                if k == target_id {
                                    out.push(b + lane);
                                }
                            }
                        }
                    } else {
                        for (bit_offset, &key) in keys[base..end].iter().enumerate() {
                            if (word & (1u64 << bit_offset)) != 0 && key == target_id {
                                out.push(base + bit_offset);
                            }
                        }
                    }
                }
            } else {
                let chunks = total_rows / 8;
                let remainder_start = chunks * 8;

                for chunk_idx in 0..chunks {
                    let base = chunk_idx * 8;
                    let slice: &[u32; 8] = match keys[base..base + 8].try_into() {
                        Ok(s) => s,
                        Err(_) => unreachable!(),
                    };
                    for (lane, &k) in slice.iter().enumerate() {
                        if k == target_id {
                            out.push(base + lane);
                        }
                    }
                }

                for (i, &k) in keys.iter().enumerate().skip(remainder_start) {
                    if k == target_id {
                        out.push(i);
                    }
                }
            }
        }
        ComparisonOp::NotEqual => {
            let keys = col.keys();
            let total_rows = keys.len();
            out.indices
                .reserve(total_rows.saturating_sub(out.indices.capacity()));

            match target_id_opt {
                None => {
                    // All non-null rows match
                    if let Some(bm) = col.null_bitmap() {
                        for i in 0..total_rows {
                            if bm.is_valid(i) {
                                out.push(i);
                            }
                        }
                    } else {
                        for i in 0..total_rows {
                            out.push(i);
                        }
                    }
                }
                Some(target_id) => {
                    if let Some(bm) = col.null_bitmap() {
                        for (i, &key) in keys.iter().enumerate() {
                            if bm.is_valid(i) && key != target_id {
                                out.push(i);
                            }
                        }
                    } else {
                        for (i, &key) in keys.iter().enumerate() {
                            if key != target_id {
                                out.push(i);
                            }
                        }
                    }
                }
            }
        }
        _ => {
            // Lexicographical comparisons against string target
            let dict = col.dictionary();
            let mut matches = vec![false; dict.len()];

            for (id, val) in dict.iter().enumerate() {
                matches[id] = match op {
                    ComparisonOp::GreaterThan => val.as_str() > target,
                    ComparisonOp::GreaterThanOrEqual => val.as_str() >= target,
                    ComparisonOp::LessThan => val.as_str() < target,
                    ComparisonOp::LessThanOrEqual => val.as_str() <= target,
                    _ => unreachable!(),
                };
            }

            let keys = col.keys();
            if let Some(bm) = col.null_bitmap() {
                for (i, &key) in keys.iter().enumerate() {
                    let k = key as usize;
                    if bm.is_valid(i) && k < matches.len() && matches[k] {
                        out.push(i);
                    }
                }
            } else {
                for (i, &key) in keys.iter().enumerate() {
                    let k = key as usize;
                    if k < matches.len() && matches[k] {
                        out.push(i);
                    }
                }
            }
        }
    }
}

/// Vectorized filter pushdown on `DictionaryColumn`.
pub fn scan_filter_dict(col: &DictionaryColumn, op: ComparisonOp, target: &str) -> SelectionVector {
    let mut out = SelectionVector::with_capacity(col.len() / 4);
    scan_filter_dict_into(col, op, target, &mut out);
    out
}

/// Filter pushdown applied on an existing `SelectionVector` for `DictionaryColumn` writing to `out`.
pub fn scan_filter_dict_with_selection_into(
    col: &DictionaryColumn,
    sel: &SelectionVector,
    op: ComparisonOp,
    target: &str,
    out: &mut SelectionVector,
) {
    let target_id_opt = col.lookup_id(target);
    let keys = col.keys();
    let bm_opt = col.null_bitmap();
    out.clear();
    out.indices
        .reserve(sel.len().saturating_sub(out.indices.capacity()));

    match op {
        ComparisonOp::Equal => {
            let target_id = match target_id_opt {
                Some(id) => id,
                None => return,
            };
            if let Some(bm) = bm_opt {
                for &idx in sel.as_slice() {
                    if bm.is_valid(idx) && keys[idx] == target_id {
                        out.push(idx);
                    }
                }
            } else {
                for &idx in sel.as_slice() {
                    if keys[idx] == target_id {
                        out.push(idx);
                    }
                }
            }
        }
        ComparisonOp::NotEqual => match target_id_opt {
            None => {
                if let Some(bm) = bm_opt {
                    for &idx in sel.as_slice() {
                        if bm.is_valid(idx) {
                            out.push(idx);
                        }
                    }
                } else {
                    for &idx in sel.as_slice() {
                        out.push(idx);
                    }
                }
            }
            Some(target_id) => {
                if let Some(bm) = bm_opt {
                    for &idx in sel.as_slice() {
                        if bm.is_valid(idx) && keys[idx] != target_id {
                            out.push(idx);
                        }
                    }
                } else {
                    for &idx in sel.as_slice() {
                        if keys[idx] != target_id {
                            out.push(idx);
                        }
                    }
                }
            }
        },
        _ => {
            let dict = col.dictionary();
            let mut matches = vec![false; dict.len()];
            for (id, val) in dict.iter().enumerate() {
                matches[id] = match op {
                    ComparisonOp::GreaterThan => val.as_str() > target,
                    ComparisonOp::GreaterThanOrEqual => val.as_str() >= target,
                    ComparisonOp::LessThan => val.as_str() < target,
                    ComparisonOp::LessThanOrEqual => val.as_str() <= target,
                    _ => unreachable!(),
                };
            }
            if let Some(bm) = bm_opt {
                for &idx in sel.as_slice() {
                    let k = keys[idx] as usize;
                    if bm.is_valid(idx) && k < matches.len() && matches[k] {
                        out.push(idx);
                    }
                }
            } else {
                for &idx in sel.as_slice() {
                    let k = keys[idx] as usize;
                    if k < matches.len() && matches[k] {
                        out.push(idx);
                    }
                }
            }
        }
    }
}

/// Filter pushdown applied on an existing `SelectionVector` for `DictionaryColumn`.
pub fn scan_filter_dict_with_selection(
    col: &DictionaryColumn,
    sel: &SelectionVector,
    op: ComparisonOp,
    target: &str,
) -> SelectionVector {
    let mut out = SelectionVector::with_capacity(sel.len());
    scan_filter_dict_with_selection_into(col, sel, op, target, &mut out);
    out
}

#[inline(always)]
pub fn evaluate_i64(val: i64, op: ComparisonOp, target: i64) -> bool {
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
pub fn evaluate_f64(val: f64, op: ComparisonOp, target: f64) -> bool {
    match op {
        ComparisonOp::Equal => val == target,
        ComparisonOp::NotEqual => val != target,
        ComparisonOp::GreaterThan => val > target,
        ComparisonOp::GreaterThanOrEqual => val >= target,
        ComparisonOp::LessThan => val < target,
        ComparisonOp::LessThanOrEqual => val <= target,
    }
}
