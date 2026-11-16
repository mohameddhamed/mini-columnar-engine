use std::collections::HashMap;

use crate::column::bitmap::NullBitmap;
use crate::column::dictionary::DictionaryColumn;
use crate::column::numeric::{Float64Column, Int64Column};
use crate::column::rle::RleColumn;
use crate::column::string::StringColumn;
use crate::column::{Column, ColumnType};
use crate::execution::aggregate::{
    avg_f64, avg_i64, count_dict, count_f64, count_i64, count_string, group_by_dict_avg_f64,
    group_by_dict_avg_i64, group_by_dict_count, group_by_dict_count_col, group_by_dict_max_f64,
    group_by_dict_max_i64, group_by_dict_min_f64, group_by_dict_min_i64, group_by_dict_sum_f64,
    group_by_dict_sum_i64, max_f64, max_i64, min_f64, min_i64, sum_f64, sum_i64, AggregateOp,
    AggregateValue,
};
use crate::execution::filter::{
    scan_chunk_filter_i64_into, scan_filter_dict, scan_filter_dict_into,
    scan_filter_dict_with_selection, scan_filter_dict_with_selection_into, scan_filter_f64,
    scan_filter_f64_into, scan_filter_f64_with_selection, scan_filter_f64_with_selection_into,
    scan_filter_i64, scan_filter_i64_into, scan_filter_i64_with_selection,
    scan_filter_i64_with_selection_into, ComparisonOp, SelectionVector, VectorBatch,
    VECTOR_CHUNK_SIZE,
};

/// Unified enum container representing any supported column storage type.
#[derive(Debug, Clone, PartialEq)]
pub enum AnyColumn {
    Int64(Int64Column),
    Float64(Float64Column),
    String(StringColumn),
    Dictionary(DictionaryColumn),
    RleInt64(RleColumn<i64>),
}

impl AnyColumn {
    pub fn name(&self) -> &str {
        match self {
            AnyColumn::Int64(c) => c.name(),
            AnyColumn::Float64(c) => c.name(),
            AnyColumn::String(c) => c.name(),
            AnyColumn::Dictionary(c) => c.name(),
            AnyColumn::RleInt64(c) => c.name(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            AnyColumn::Int64(c) => c.len(),
            AnyColumn::Float64(c) => c.len(),
            AnyColumn::String(c) => c.len(),
            AnyColumn::Dictionary(c) => c.len(),
            AnyColumn::RleInt64(c) => c.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn column_type(&self) -> ColumnType {
        match self {
            AnyColumn::Int64(c) => c.column_type(),
            AnyColumn::Float64(c) => c.column_type(),
            AnyColumn::String(c) => c.column_type(),
            AnyColumn::Dictionary(c) => c.column_type(),
            AnyColumn::RleInt64(c) => c.column_type(),
        }
    }

    pub fn null_bitmap(&self) -> Option<&NullBitmap> {
        match self {
            AnyColumn::Int64(c) => c.null_bitmap(),
            AnyColumn::Float64(c) => c.null_bitmap(),
            AnyColumn::String(c) => c.null_bitmap(),
            AnyColumn::Dictionary(c) => c.null_bitmap(),
            AnyColumn::RleInt64(c) => c.null_bitmap(),
        }
    }

    pub fn memory_bytes(&self) -> usize {
        match self {
            AnyColumn::Int64(c) => c.memory_bytes(),
            AnyColumn::Float64(c) => c.memory_bytes(),
            AnyColumn::String(c) => c.memory_bytes(),
            AnyColumn::Dictionary(c) => c.memory_bytes(),
            AnyColumn::RleInt64(c) => c.memory_bytes(),
        }
    }
}

/// Metadata definition for a single column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: ColumnType,
}

/// In-memory OLAP Table storing columns in contiguous columnar buffers.
///
/// Designed to emulate SAP HANA's TableStore and Celonis's PQL internal representation:
/// schema validation, zero-copy vector references, selection-vector predicate pushdown,
/// and direct-indexed aggregations.
#[derive(Debug, Clone, Default)]
pub struct ColumnTable {
    name: String,
    columns: HashMap<String, AnyColumn>,
    column_order: Vec<String>,
    row_count: usize,
}

impl ColumnTable {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            columns: HashMap::new(),
            column_order: Vec::new(),
            row_count: 0,
        }
    }

    /// Table name identifier.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Add a column to the table. Enforces row count consistency across columns.
    pub fn add_column(&mut self, col: AnyColumn) -> Result<(), String> {
        let name = col.name().to_string();
        let col_len = col.len();

        if self.columns.is_empty() {
            self.row_count = col_len;
        } else if self.row_count != col_len {
            return Err(format!(
                "Column '{}' length ({}) does not match table row count ({})",
                name, col_len, self.row_count
            ));
        }

        self.columns.insert(name.clone(), col);
        if !self.column_order.contains(&name) {
            self.column_order.push(name);
        }
        Ok(())
    }

    /// Number of rows across all columns.
    #[inline(always)]
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Number of columns in the table.
    #[inline(always)]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Retrieve column by name.
    pub fn get_column(&self, name: &str) -> Option<&AnyColumn> {
        self.columns.get(name)
    }

    /// Helper to get a reference to an `Int64Column`.
    pub fn get_int64_column(&self, name: &str) -> Option<&Int64Column> {
        match self.columns.get(name) {
            Some(AnyColumn::Int64(c)) => Some(c),
            _ => None,
        }
    }

    /// Helper to get a reference to a `Float64Column`.
    pub fn get_float64_column(&self, name: &str) -> Option<&Float64Column> {
        match self.columns.get(name) {
            Some(AnyColumn::Float64(c)) => Some(c),
            _ => None,
        }
    }

    /// Helper to get a reference to a `DictionaryColumn`.
    pub fn get_dict_column(&self, name: &str) -> Option<&DictionaryColumn> {
        match self.columns.get(name) {
            Some(AnyColumn::Dictionary(c)) => Some(c),
            _ => None,
        }
    }

    /// Total memory footprint of all columns in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.columns.values().map(|c| c.memory_bytes()).sum()
    }

    /// Execute a vectorized filter scan over an integer column.
    pub fn filter_i64(
        &self,
        col_name: &str,
        op: ComparisonOp,
        target: i64,
    ) -> Result<SelectionVector, String> {
        let col = self
            .get_int64_column(col_name)
            .ok_or_else(|| format!("Int64 column '{}' not found", col_name))?;
        Ok(scan_filter_i64(col, op, target))
    }

    /// In-place filter over an Int64 column writing matching row indices into `out`.
    pub fn filter_i64_into(
        &self,
        col_name: &str,
        op: ComparisonOp,
        target: i64,
        out: &mut SelectionVector,
    ) -> Result<(), String> {
        let col = self
            .get_int64_column(col_name)
            .ok_or_else(|| format!("Int64 column '{}' not found", col_name))?;
        scan_filter_i64_into(col, op, target, out);
        Ok(())
    }

    /// Execute a vectorized filter scan over an integer column with an existing selection vector.
    pub fn filter_i64_with_sel(
        &self,
        col_name: &str,
        op: ComparisonOp,
        target: i64,
        sel: &SelectionVector,
    ) -> Result<SelectionVector, String> {
        let col = self
            .get_int64_column(col_name)
            .ok_or_else(|| format!("Int64 column '{}' not found", col_name))?;
        Ok(scan_filter_i64_with_selection(col, sel, op, target))
    }

    /// In-place chained filter over an Int64 column refining an existing `sel` into `out`.
    pub fn filter_i64_with_sel_into(
        &self,
        col_name: &str,
        op: ComparisonOp,
        target: i64,
        sel: &SelectionVector,
        out: &mut SelectionVector,
    ) -> Result<(), String> {
        let col = self
            .get_int64_column(col_name)
            .ok_or_else(|| format!("Int64 column '{}' not found", col_name))?;
        scan_filter_i64_with_selection_into(col, sel, op, target, out);
        Ok(())
    }

    /// Execute a vectorized filter scan over a float column.
    pub fn filter_f64(
        &self,
        col_name: &str,
        op: ComparisonOp,
        target: f64,
    ) -> Result<SelectionVector, String> {
        let col = self
            .get_float64_column(col_name)
            .ok_or_else(|| format!("Float64 column '{}' not found", col_name))?;
        Ok(scan_filter_f64(col, op, target))
    }

    /// In-place filter over a Float64 column writing matching row indices into `out`.
    pub fn filter_f64_into(
        &self,
        col_name: &str,
        op: ComparisonOp,
        target: f64,
        out: &mut SelectionVector,
    ) -> Result<(), String> {
        let col = self
            .get_float64_column(col_name)
            .ok_or_else(|| format!("Float64 column '{}' not found", col_name))?;
        scan_filter_f64_into(col, op, target, out);
        Ok(())
    }

    /// Execute a vectorized filter scan over a float column with an existing selection vector.
    pub fn filter_f64_with_sel(
        &self,
        col_name: &str,
        op: ComparisonOp,
        target: f64,
        sel: &SelectionVector,
    ) -> Result<SelectionVector, String> {
        let col = self
            .get_float64_column(col_name)
            .ok_or_else(|| format!("Float64 column '{}' not found", col_name))?;
        Ok(scan_filter_f64_with_selection(col, sel, op, target))
    }

    /// In-place chained filter over a Float64 column refining an existing `sel` into `out`.
    pub fn filter_f64_with_sel_into(
        &self,
        col_name: &str,
        op: ComparisonOp,
        target: f64,
        sel: &SelectionVector,
        out: &mut SelectionVector,
    ) -> Result<(), String> {
        let col = self
            .get_float64_column(col_name)
            .ok_or_else(|| format!("Float64 column '{}' not found", col_name))?;
        scan_filter_f64_with_selection_into(col, sel, op, target, out);
        Ok(())
    }

    /// Execute a vectorized filter scan over a dictionary column.
    pub fn filter_dict(
        &self,
        col_name: &str,
        op: ComparisonOp,
        target: &str,
    ) -> Result<SelectionVector, String> {
        let col = self
            .get_dict_column(col_name)
            .ok_or_else(|| format!("Dictionary column '{}' not found", col_name))?;
        Ok(scan_filter_dict(col, op, target))
    }

    /// In-place filter over a Dictionary column writing matching row indices into `out`.
    pub fn filter_dict_into(
        &self,
        col_name: &str,
        op: ComparisonOp,
        target: &str,
        out: &mut SelectionVector,
    ) -> Result<(), String> {
        let col = self
            .get_dict_column(col_name)
            .ok_or_else(|| format!("Dictionary column '{}' not found", col_name))?;
        scan_filter_dict_into(col, op, target, out);
        Ok(())
    }

    /// Execute a vectorized filter scan over a dictionary column with an existing selection vector.
    pub fn filter_dict_with_sel(
        &self,
        col_name: &str,
        op: ComparisonOp,
        target: &str,
        sel: &SelectionVector,
    ) -> Result<SelectionVector, String> {
        let col = self
            .get_dict_column(col_name)
            .ok_or_else(|| format!("Dictionary column '{}' not found", col_name))?;
        Ok(scan_filter_dict_with_selection(col, sel, op, target))
    }

    /// In-place chained filter over a Dictionary column refining an existing `sel` into `out`.
    pub fn filter_dict_with_sel_into(
        &self,
        col_name: &str,
        op: ComparisonOp,
        target: &str,
        sel: &SelectionVector,
        out: &mut SelectionVector,
    ) -> Result<(), String> {
        let col = self
            .get_dict_column(col_name)
            .ok_or_else(|| format!("Dictionary column '{}' not found", col_name))?;
        scan_filter_dict_with_selection_into(col, sel, op, target, out);
        Ok(())
    }

    /// COUNT(*) across the table, respecting any active selection vector.
    pub fn count_star(&self, sel: Option<&SelectionVector>) -> usize {
        match sel {
            Some(s) => s.len(),
            None => self.row_count,
        }
    }

    /// Single-pass vectorized GROUP BY COUNT(*) over a dictionary grouping column.
    pub fn group_by_count_star(
        &self,
        group_col_name: &str,
        sel: Option<&SelectionVector>,
    ) -> Result<Vec<(String, usize)>, String> {
        let group_col = self
            .get_dict_column(group_col_name)
            .ok_or_else(|| format!("Dictionary group column '{}' not found", group_col_name))?;
        Ok(group_by_dict_count(group_col, sel))
    }

    /// Pipelined chunked execution: filters `filter_col` and aggregates `val_col`
    /// chunk-by-chunk (morsels of 1024 elements).
    ///
    /// The intermediate selection vector has at most 1024 entries (`4 KB`),
    /// keeping intermediate indices strictly resident in L1 D-Cache without ever
    /// allocating a global table-wide selection vector buffer.
    pub fn aggregate_filtered_chunked_i64(
        &self,
        filter_col_name: &str,
        filter_op: ComparisonOp,
        filter_target: i64,
        val_col_name: &str,
        agg_op: AggregateOp,
    ) -> Result<AggregateValue, String> {
        let f_col = self
            .get_int64_column(filter_col_name)
            .ok_or_else(|| format!("Int64 filter column '{}' not found", filter_col_name))?;
        let v_col = self
            .get_int64_column(val_col_name)
            .ok_or_else(|| format!("Int64 val column '{}' not found", val_col_name))?;

        let mut sel_chunk = SelectionVector::with_capacity(VECTOR_CHUNK_SIZE);
        let val_slice = v_col.as_slice();
        let val_bm = v_col.null_bitmap();

        let mut sum: i64 = 0;
        let mut count: usize = 0;
        let mut min_val: i64 = i64::MAX;
        let mut max_val: i64 = i64::MIN;
        let mut found = false;

        let total_rows = self.row_count;
        let mut chunk_start = 0;

        while chunk_start < total_rows {
            let chunk_end = (chunk_start + VECTOR_CHUNK_SIZE).min(total_rows);
            let batch = VectorBatch::new(chunk_start, chunk_end - chunk_start);
            scan_chunk_filter_i64_into(f_col, &batch, filter_op, filter_target, &mut sel_chunk);

            for &row_idx in sel_chunk.as_slice() {
                if let Some(bm) = val_bm {
                    if bm.is_null(row_idx) {
                        continue;
                    }
                }
                let val = val_slice[row_idx];
                found = true;
                count += 1;
                match agg_op {
                    AggregateOp::Sum => sum += val,
                    AggregateOp::Avg => sum += val,
                    AggregateOp::Min => {
                        if val < min_val {
                            min_val = val;
                        }
                    }
                    AggregateOp::Max => {
                        if val > max_val {
                            max_val = val;
                        }
                    }
                    AggregateOp::Count => {}
                }
            }

            chunk_start = chunk_end;
        }

        match agg_op {
            AggregateOp::Sum => {
                if found {
                    Ok(AggregateValue::Int(sum))
                } else {
                    Ok(AggregateValue::Null)
                }
            }
            AggregateOp::Avg => {
                if count > 0 {
                    Ok(AggregateValue::Float(sum as f64 / count as f64))
                } else {
                    Ok(AggregateValue::Null)
                }
            }
            AggregateOp::Min => {
                if found {
                    Ok(AggregateValue::Int(min_val))
                } else {
                    Ok(AggregateValue::Null)
                }
            }
            AggregateOp::Max => {
                if found {
                    Ok(AggregateValue::Int(max_val))
                } else {
                    Ok(AggregateValue::Null)
                }
            }
            AggregateOp::Count => Ok(AggregateValue::Count(count)),
        }
    }

    /// Execute an analytical aggregation over a numeric column, optionally filtered by `sel`.
    pub fn aggregate(
        &self,
        col_name: &str,
        op: AggregateOp,
        sel: Option<&SelectionVector>,
    ) -> Result<AggregateValue, String> {
        let col = self
            .get_column(col_name)
            .ok_or_else(|| format!("Column '{}' not found", col_name))?;

        match col {
            AnyColumn::Int64(c) => match op {
                AggregateOp::Sum => match sum_i64(c, sel) {
                    Some(s) => Ok(AggregateValue::Int(s)),
                    None => Ok(AggregateValue::Null),
                },
                AggregateOp::Avg => match avg_i64(c, sel) {
                    Some(a) => Ok(AggregateValue::Float(a)),
                    None => Ok(AggregateValue::Null),
                },
                AggregateOp::Min => match min_i64(c, sel) {
                    Some(m) => Ok(AggregateValue::Int(m)),
                    None => Ok(AggregateValue::Null),
                },
                AggregateOp::Max => match max_i64(c, sel) {
                    Some(m) => Ok(AggregateValue::Int(m)),
                    None => Ok(AggregateValue::Null),
                },
                AggregateOp::Count => Ok(AggregateValue::Count(count_i64(c, sel))),
            },
            AnyColumn::Float64(c) => match op {
                AggregateOp::Sum => match sum_f64(c, sel) {
                    Some(s) => Ok(AggregateValue::Float(s)),
                    None => Ok(AggregateValue::Null),
                },
                AggregateOp::Avg => match avg_f64(c, sel) {
                    Some(a) => Ok(AggregateValue::Float(a)),
                    None => Ok(AggregateValue::Null),
                },
                AggregateOp::Min => match min_f64(c, sel) {
                    Some(m) => Ok(AggregateValue::Float(m)),
                    None => Ok(AggregateValue::Null),
                },
                AggregateOp::Max => match max_f64(c, sel) {
                    Some(m) => Ok(AggregateValue::Float(m)),
                    None => Ok(AggregateValue::Null),
                },
                AggregateOp::Count => Ok(AggregateValue::Count(count_f64(c, sel))),
            },
            AnyColumn::RleInt64(c) => match op {
                AggregateOp::Sum => match c.sum() {
                    Some(s) => Ok(AggregateValue::Int(s)),
                    None => Ok(AggregateValue::Null),
                },
                AggregateOp::Count => Ok(AggregateValue::Count(c.len())),
                AggregateOp::Min => match c.min() {
                    Some(m) => Ok(AggregateValue::Int(m)),
                    None => Ok(AggregateValue::Null),
                },
                AggregateOp::Max => match c.max() {
                    Some(m) => Ok(AggregateValue::Int(m)),
                    None => Ok(AggregateValue::Null),
                },
                AggregateOp::Avg => match c.avg() {
                    Some(a) => Ok(AggregateValue::Float(a)),
                    None => Ok(AggregateValue::Null),
                },
            },
            AnyColumn::Dictionary(c) => match op {
                AggregateOp::Count => Ok(AggregateValue::Count(count_dict(c, sel))),
                _ => Err(format!(
                    "Cannot compute numeric aggregate {:?} on Dictionary column",
                    op
                )),
            },
            AnyColumn::String(c) => match op {
                AggregateOp::Count => Ok(AggregateValue::Count(count_string(c, sel))),
                _ => Err(format!(
                    "Cannot compute numeric aggregate {:?} on String column",
                    op
                )),
            },
        }
    }

    /// Single-pass vectorized GROUP BY over a dictionary grouping column.
    pub fn group_by(
        &self,
        group_col_name: &str,
        val_col_name: &str,
        op: AggregateOp,
        sel: Option<&SelectionVector>,
    ) -> Result<Vec<(String, AggregateValue)>, String> {
        let group_col = self
            .get_dict_column(group_col_name)
            .ok_or_else(|| format!("Dictionary group column '{}' not found", group_col_name))?;

        let val_col = self
            .get_column(val_col_name)
            .ok_or_else(|| format!("Value column '{}' not found", val_col_name))?;

        match (val_col, op) {
            (AnyColumn::Int64(ic), AggregateOp::Sum) => {
                let results = group_by_dict_sum_i64(group_col, ic, sel);
                Ok(results
                    .into_iter()
                    .map(|(k, v)| (k, AggregateValue::Int(v)))
                    .collect())
            }
            (AnyColumn::Int64(ic), AggregateOp::Avg) => {
                let results = group_by_dict_avg_i64(group_col, ic, sel);
                Ok(results
                    .into_iter()
                    .map(|(k, v)| {
                        (
                            k,
                            match v {
                                Some(a) => AggregateValue::Float(a),
                                None => AggregateValue::Null,
                            },
                        )
                    })
                    .collect())
            }
            (AnyColumn::Int64(ic), AggregateOp::Min) => {
                let results = group_by_dict_min_i64(group_col, ic, sel);
                Ok(results
                    .into_iter()
                    .map(|(k, v)| {
                        (
                            k,
                            match v {
                                Some(m) => AggregateValue::Int(m),
                                None => AggregateValue::Null,
                            },
                        )
                    })
                    .collect())
            }
            (AnyColumn::Int64(ic), AggregateOp::Max) => {
                let results = group_by_dict_max_i64(group_col, ic, sel);
                Ok(results
                    .into_iter()
                    .map(|(k, v)| {
                        (
                            k,
                            match v {
                                Some(m) => AggregateValue::Int(m),
                                None => AggregateValue::Null,
                            },
                        )
                    })
                    .collect())
            }
            (AnyColumn::Float64(fc), AggregateOp::Sum) => {
                let results = group_by_dict_sum_f64(group_col, fc, sel);
                Ok(results
                    .into_iter()
                    .map(|(k, v)| (k, AggregateValue::Float(v)))
                    .collect())
            }
            (AnyColumn::Float64(fc), AggregateOp::Avg) => {
                let results = group_by_dict_avg_f64(group_col, fc, sel);
                Ok(results
                    .into_iter()
                    .map(|(k, v)| {
                        (
                            k,
                            match v {
                                Some(a) => AggregateValue::Float(a),
                                None => AggregateValue::Null,
                            },
                        )
                    })
                    .collect())
            }
            (AnyColumn::Float64(fc), AggregateOp::Min) => {
                let results = group_by_dict_min_f64(group_col, fc, sel);
                Ok(results
                    .into_iter()
                    .map(|(k, v)| {
                        (
                            k,
                            match v {
                                Some(m) => AggregateValue::Float(m),
                                None => AggregateValue::Null,
                            },
                        )
                    })
                    .collect())
            }
            (AnyColumn::Float64(fc), AggregateOp::Max) => {
                let results = group_by_dict_max_f64(group_col, fc, sel);
                Ok(results
                    .into_iter()
                    .map(|(k, v)| {
                        (
                            k,
                            match v {
                                Some(m) => AggregateValue::Float(m),
                                None => AggregateValue::Null,
                            },
                        )
                    })
                    .collect())
            }
            (_, AggregateOp::Count) => {
                let results = group_by_dict_count_col(group_col, val_col.null_bitmap(), sel);
                Ok(results
                    .into_iter()
                    .map(|(k, v)| (k, AggregateValue::Count(v)))
                    .collect())
            }
            _ => Err(format!(
                "Unsupported group-by combination: column '{}' with op {:?}",
                val_col_name, op
            )),
        }
    }
}
