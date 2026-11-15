use super::filter::SelectionVector;
use crate::column::dictionary::DictionaryColumn;
use crate::column::numeric::{Float64Column, Int64Column};
use crate::column::Column;
use std::collections::HashMap;

/// Aggregation operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateOp {
    Sum,
    Avg,
    Min,
    Max,
    Count,
}

impl std::fmt::Display for AggregateOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregateOp::Sum => write!(f, "SUM"),
            AggregateOp::Avg => write!(f, "AVG"),
            AggregateOp::Min => write!(f, "MIN"),
            AggregateOp::Max => write!(f, "MAX"),
            AggregateOp::Count => write!(f, "COUNT"),
        }
    }
}

/// Result of an analytical aggregation.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregateValue {
    Int(i64),
    Float(f64),
    Count(usize),
    Null,
}

impl std::fmt::Display for AggregateValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregateValue::Int(v) => write!(f, "{}", v),
            AggregateValue::Float(v) => write!(f, "{:.4}", v),
            AggregateValue::Count(v) => write!(f, "{}", v),
            AggregateValue::Null => write!(f, "NULL"),
        }
    }
}

// -----------------------------------------------------------------------------
// Vectorized Numeric Aggregations (Int64)
// -----------------------------------------------------------------------------

/// Vectorized SUM over `Int64Column`.
///
/// Uses 4-way independent accumulator unrolling when scanning full contiguous slices,
/// breaking CPU instruction dependency chains and maximizing superscalar pipeline throughput.
pub fn sum_i64(col: &Int64Column, sel: Option<&SelectionVector>) -> i64 {
    let data = col.as_slice();

    match sel {
        Some(sv) => {
            let indices = sv.as_slice();
            let mut sum0 = 0i64;
            let mut sum1 = 0i64;
            let mut sum2 = 0i64;
            let mut sum3 = 0i64;

            let chunks = indices.len() / 4;
            let rem = chunks * 4;

            for i in 0..chunks {
                let base = i * 4;
                sum0 += data[indices[base]];
                sum1 += data[indices[base + 1]];
                sum2 += data[indices[base + 2]];
                sum3 += data[indices[base + 3]];
            }

            for &idx in &indices[rem..] {
                sum0 += data[idx];
            }

            sum0 + sum1 + sum2 + sum3
        }
        None => {
            if let Some(bm) = col.null_bitmap() {
                let mut sum = 0i64;
                for (i, &v) in data.iter().enumerate() {
                    if bm.is_valid(i) {
                        sum += v;
                    }
                }
                sum
            } else {
                let mut sum0 = 0i64;
                let mut sum1 = 0i64;
                let mut sum2 = 0i64;
                let mut sum3 = 0i64;

                let chunks = data.len() / 4;
                let rem = chunks * 4;

                for i in 0..chunks {
                    let base = i * 4;
                    sum0 += data[base];
                    sum1 += data[base + 1];
                    sum2 += data[base + 2];
                    sum3 += data[base + 3];
                }

                for &v in &data[rem..] {
                    sum0 += v;
                }

                sum0 + sum1 + sum2 + sum3
            }
        }
    }
}

/// Vectorized AVG over `Int64Column`.
pub fn avg_i64(col: &Int64Column, sel: Option<&SelectionVector>) -> Option<f64> {
    let count = match sel {
        Some(sv) => sv.len(),
        None => {
            if let Some(bm) = col.null_bitmap() {
                col.len() - bm.null_count()
            } else {
                col.len()
            }
        }
    };

    if count == 0 {
        return None;
    }

    let sum = sum_i64(col, sel);
    Some(sum as f64 / count as f64)
}

/// Vectorized MIN over `Int64Column`.
pub fn min_i64(col: &Int64Column, sel: Option<&SelectionVector>) -> Option<i64> {
    let data = col.as_slice();

    match sel {
        Some(sv) => {
            let indices = sv.as_slice();
            if indices.is_empty() {
                return None;
            }
            let mut min_val = i64::MAX;
            for &idx in indices {
                let val = data[idx];
                if val < min_val {
                    min_val = val;
                }
            }
            Some(min_val)
        }
        None => {
            if data.is_empty() {
                return None;
            }
            if let Some(bm) = col.null_bitmap() {
                let mut min_val = i64::MAX;
                let mut found = false;
                for (i, &v) in data.iter().enumerate() {
                    if bm.is_valid(i) {
                        found = true;
                        if v < min_val {
                            min_val = v;
                        }
                    }
                }
                if found {
                    Some(min_val)
                } else {
                    None
                }
            } else {
                data.iter().copied().min()
            }
        }
    }
}

/// Vectorized MAX over `Int64Column`.
pub fn max_i64(col: &Int64Column, sel: Option<&SelectionVector>) -> Option<i64> {
    let data = col.as_slice();

    match sel {
        Some(sv) => {
            let indices = sv.as_slice();
            if indices.is_empty() {
                return None;
            }
            let mut max_val = i64::MIN;
            for &idx in indices {
                let val = data[idx];
                if val > max_val {
                    max_val = val;
                }
            }
            Some(max_val)
        }
        None => {
            if data.is_empty() {
                return None;
            }
            if let Some(bm) = col.null_bitmap() {
                let mut max_val = i64::MIN;
                let mut found = false;
                for (i, &v) in data.iter().enumerate() {
                    if bm.is_valid(i) {
                        found = true;
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
                if found {
                    Some(max_val)
                } else {
                    None
                }
            } else {
                data.iter().copied().max()
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Vectorized Numeric Aggregations (Float64)
// -----------------------------------------------------------------------------

pub fn sum_f64(col: &Float64Column, sel: Option<&SelectionVector>) -> f64 {
    let data = col.as_slice();

    match sel {
        Some(sv) => {
            let indices = sv.as_slice();
            let mut sum0 = 0.0f64;
            let mut sum1 = 0.0f64;
            let mut sum2 = 0.0f64;
            let mut sum3 = 0.0f64;

            let chunks = indices.len() / 4;
            let rem = chunks * 4;

            for i in 0..chunks {
                let base = i * 4;
                sum0 += data[indices[base]];
                sum1 += data[indices[base + 1]];
                sum2 += data[indices[base + 2]];
                sum3 += data[indices[base + 3]];
            }

            for &idx in &indices[rem..] {
                sum0 += data[idx];
            }

            sum0 + sum1 + sum2 + sum3
        }
        None => {
            if let Some(bm) = col.null_bitmap() {
                let mut sum = 0.0f64;
                for (i, &v) in data.iter().enumerate() {
                    if bm.is_valid(i) {
                        sum += v;
                    }
                }
                sum
            } else {
                let mut sum0 = 0.0f64;
                let mut sum1 = 0.0f64;
                let mut sum2 = 0.0f64;
                let mut sum3 = 0.0f64;

                let chunks = data.len() / 4;
                let rem = chunks * 4;

                for i in 0..chunks {
                    let base = i * 4;
                    sum0 += data[base];
                    sum1 += data[base + 1];
                    sum2 += data[base + 2];
                    sum3 += data[base + 3];
                }

                for &v in &data[rem..] {
                    sum0 += v;
                }

                sum0 + sum1 + sum2 + sum3
            }
        }
    }
}

pub fn avg_f64(col: &Float64Column, sel: Option<&SelectionVector>) -> Option<f64> {
    let count = match sel {
        Some(sv) => sv.len(),
        None => {
            if let Some(bm) = col.null_bitmap() {
                col.len() - bm.null_count()
            } else {
                col.len()
            }
        }
    };

    if count == 0 {
        return None;
    }

    let sum = sum_f64(col, sel);
    Some(sum / count as f64)
}

pub fn min_f64(col: &Float64Column, sel: Option<&SelectionVector>) -> Option<f64> {
    let data = col.as_slice();

    match sel {
        Some(sv) => {
            let indices = sv.as_slice();
            if indices.is_empty() {
                return None;
            }
            let mut min_val = f64::INFINITY;
            for &idx in indices {
                let val = data[idx];
                if val < min_val {
                    min_val = val;
                }
            }
            Some(min_val)
        }
        None => {
            if data.is_empty() {
                return None;
            }
            if let Some(bm) = col.null_bitmap() {
                let mut min_val = f64::INFINITY;
                let mut found = false;
                for (i, &v) in data.iter().enumerate() {
                    if bm.is_valid(i) {
                        found = true;
                        if v < min_val {
                            min_val = v;
                        }
                    }
                }
                if found {
                    Some(min_val)
                } else {
                    None
                }
            } else {
                data.iter().copied().reduce(f64::min)
            }
        }
    }
}

pub fn max_f64(col: &Float64Column, sel: Option<&SelectionVector>) -> Option<f64> {
    let data = col.as_slice();

    match sel {
        Some(sv) => {
            let indices = sv.as_slice();
            if indices.is_empty() {
                return None;
            }
            let mut max_val = f64::NEG_INFINITY;
            for &idx in indices {
                let val = data[idx];
                if val > max_val {
                    max_val = val;
                }
            }
            Some(max_val)
        }
        None => {
            if data.is_empty() {
                return None;
            }
            if let Some(bm) = col.null_bitmap() {
                let mut max_val = f64::NEG_INFINITY;
                let mut found = false;
                for (i, &v) in data.iter().enumerate() {
                    if bm.is_valid(i) {
                        found = true;
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
                if found {
                    Some(max_val)
                } else {
                    None
                }
            } else {
                data.iter().copied().reduce(f64::max)
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Vectorized Group-By over Dictionary-Encoded Keys
// -----------------------------------------------------------------------------

/// Single-pass vectorized GROUP BY over `DictionaryColumn` with `SUM(i64)`.
///
/// **The SAP HANA / Celonis PQL Advantage:**
/// Because the grouping column is dictionary-encoded, its distinct values are mapped
/// to compact integers `0..cardinality-1`. Instead of inserting into an expensive
/// `HashMap` (which causes cache misses, hash computations, and collisions),
/// this algorithm allocates a flat contiguous accumulator vector `Vec<i64>` of length
/// `cardinality`.
///
/// Group aggregation updates are direct array accesses: `accumulators[key as usize] += val`.
/// Zero hash lookups, zero allocations during scan, 100% L1 cache resident.
pub fn group_by_dict_sum_i64(
    group_col: &DictionaryColumn,
    val_col: &Int64Column,
    sel: Option<&SelectionVector>,
) -> Vec<(String, i64)> {
    let card = group_col.cardinality();
    let mut sums = vec![0i64; card];
    let keys = group_col.keys();
    let vals = val_col.as_slice();

    match sel {
        Some(sv) => {
            for &idx in sv.as_slice() {
                let key = keys[idx] as usize;
                if key < card {
                    sums[key] += vals[idx];
                }
            }
        }
        None => {
            let total = keys.len().min(vals.len());
            for i in 0..total {
                let key = keys[i] as usize;
                if key < card {
                    sums[key] += vals[i];
                }
            }
        }
    }

    let dict = group_col.dictionary();
    dict.iter()
        .enumerate()
        .map(|(id, token)| (token.clone(), sums[id]))
        .collect()
}

/// Single-pass vectorized GROUP BY over `DictionaryColumn` with `SUM(f64)`.
pub fn group_by_dict_sum_f64(
    group_col: &DictionaryColumn,
    val_col: &Float64Column,
    sel: Option<&SelectionVector>,
) -> Vec<(String, f64)> {
    let card = group_col.cardinality();
    let mut sums = vec![0.0f64; card];
    let keys = group_col.keys();
    let vals = val_col.as_slice();

    match sel {
        Some(sv) => {
            for &idx in sv.as_slice() {
                let key = keys[idx] as usize;
                if key < card {
                    sums[key] += vals[idx];
                }
            }
        }
        None => {
            let total = keys.len().min(vals.len());
            for i in 0..total {
                let key = keys[i] as usize;
                if key < card {
                    sums[key] += vals[i];
                }
            }
        }
    }

    let dict = group_col.dictionary();
    dict.iter()
        .enumerate()
        .map(|(id, token)| (token.clone(), sums[id]))
        .collect()
}

/// Single-pass vectorized GROUP BY over `DictionaryColumn` with `COUNT(*)`.
pub fn group_by_dict_count(
    group_col: &DictionaryColumn,
    sel: Option<&SelectionVector>,
) -> Vec<(String, usize)> {
    let card = group_col.cardinality();
    let mut counts = vec![0usize; card];
    let keys = group_col.keys();

    match sel {
        Some(sv) => {
            for &idx in sv.as_slice() {
                let key = keys[idx] as usize;
                if key < card {
                    counts[key] += 1;
                }
            }
        }
        None => {
            for &k in keys {
                let key = k as usize;
                if key < card {
                    counts[key] += 1;
                }
            }
        }
    }

    let dict = group_col.dictionary();
    dict.iter()
        .enumerate()
        .map(|(id, token)| (token.clone(), counts[id]))
        .collect()
}

/// Traditional HashMap-based group-by over raw strings (used for benchmark comparison).
pub fn group_by_raw_string_sum_i64(
    groups: &[String],
    vals: &[i64],
    sel: Option<&SelectionVector>,
) -> Vec<(String, i64)> {
    let mut map: HashMap<String, i64> = HashMap::new();

    match sel {
        Some(sv) => {
            for &idx in sv.as_slice() {
                *map.entry(groups[idx].clone()).or_insert(0) += vals[idx];
            }
        }
        None => {
            let total = groups.len().min(vals.len());
            for i in 0..total {
                *map.entry(groups[i].clone()).or_insert(0) += vals[i];
            }
        }
    }

    map.into_iter().collect()
}
