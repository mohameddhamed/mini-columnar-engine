use super::bitmap::NullBitmap;
use super::{Column, ColumnType};

/// Contiguous in-memory column storing 64-bit signed integers (`i64`).
///
/// Under the hood, values reside in a flat `Vec<i64>`, ensuring strict cache-line
/// alignment, sequential prefetching, and zero pointer chasing during analytical scans.
/// Nullability is handled via an optional compact `NullBitmap`.
#[derive(Debug, Clone, PartialEq)]
pub struct Int64Column {
    name: String,
    data: Vec<i64>,
    null_bitmap: Option<NullBitmap>,
}

impl Int64Column {
    /// Create an empty `Int64Column` with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data: Vec::new(),
            null_bitmap: None,
        }
    }

    /// Pre-allocate capacity for `capacity` rows.
    pub fn with_capacity(name: impl Into<String>, capacity: usize) -> Self {
        Self {
            name: name.into(),
            data: Vec::with_capacity(capacity),
            null_bitmap: None,
        }
    }

    /// Access null bitmap directly.
    #[inline(always)]
    pub fn null_bitmap(&self) -> Option<&NullBitmap> {
        self.null_bitmap.as_ref()
    }

    /// Construct a non-null `Int64Column` directly from a slice.
    pub fn from_slice(name: impl Into<String>, slice: &[i64]) -> Self {
        Self {
            name: name.into(),
            data: slice.to_vec(),
            null_bitmap: None,
        }
    }

    /// Construct a column from an iterator of primitive `i64` values (all valid).
    pub fn from_iter<I: IntoIterator<Item = i64>>(name: impl Into<String>, iter: I) -> Self {
        let data: Vec<i64> = iter.into_iter().collect();
        Self {
            name: name.into(),
            data,
            null_bitmap: None,
        }
    }

    /// Construct a column from an iterator of optional values.
    pub fn from_optional_iter<I: IntoIterator<Item = Option<i64>>>(
        name: impl Into<String>,
        iter: I,
    ) -> Self {
        let name = name.into();
        let items: Vec<Option<i64>> = iter.into_iter().collect();
        let mut data = Vec::with_capacity(items.len());
        let mut bm = NullBitmap::new(items.len(), true);
        let mut has_null = false;

        for (i, opt) in items.into_iter().enumerate() {
            match opt {
                Some(val) => data.push(val),
                None => {
                    data.push(0);
                    bm.set_null(i);
                    has_null = true;
                }
            }
        }

        Self {
            name,
            data,
            null_bitmap: if has_null { Some(bm) } else { None },
        }
    }

    /// Push a non-null value.
    pub fn push(&mut self, val: i64) {
        if let Some(ref mut bm) = self.null_bitmap {
            bm.append_valid();
        }
        self.data.push(val);
    }

    /// Push a null value.
    pub fn push_null(&mut self) {
        if self.null_bitmap.is_none() {
            let mut bm = NullBitmap::new(self.data.len(), true);
            bm.append_null();
            self.null_bitmap = Some(bm);
        } else if let Some(ref mut bm) = self.null_bitmap {
            bm.append_null();
        }
        self.data.push(0);
    }

    /// Contiguous read-only view of the data payload.
    #[inline(always)]
    pub fn as_slice(&self) -> &[i64] {
        &self.data
    }

    /// Get value at row `idx`, returning `None` if the value is null or out of bounds.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> Option<i64> {
        if idx >= self.data.len() {
            return None;
        }
        if let Some(ref bm) = self.null_bitmap {
            if bm.is_null(idx) {
                return None;
            }
        }
        Some(self.data[idx])
    }

    /// Directly access value assuming row is valid and in bounds.
    #[inline(always)]
    pub fn value_unchecked(&self, idx: usize) -> i64 {
        self.data[idx]
    }

    /// Returns true if this column contains any null values.
    #[inline(always)]
    pub fn has_nulls(&self) -> bool {
        self.null_bitmap
            .as_ref()
            .map(|bm| bm.null_count() > 0)
            .unwrap_or(false)
    }

    /// Yield contiguous chunks of `chunk_size` elements for vectorized iteration.
    #[inline(always)]
    pub fn chunks(&self, chunk_size: usize) -> std::slice::Chunks<'_, i64> {
        self.data.chunks(chunk_size)
    }
}

impl Column for Int64Column {
    fn name(&self) -> &str {
        &self.name
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn column_type(&self) -> ColumnType {
        ColumnType::Int64
    }

    fn null_bitmap(&self) -> Option<&NullBitmap> {
        self.null_bitmap.as_ref()
    }

    fn is_null(&self, idx: usize) -> bool {
        self.null_bitmap
            .as_ref()
            .map(|bm| bm.is_null(idx))
            .unwrap_or(false)
    }

    fn memory_bytes(&self) -> usize {
        let data_bytes = self.data.capacity() * std::mem::size_of::<i64>();
        let bm_bytes = self
            .null_bitmap
            .as_ref()
            .map(|bm| bm.words().len() * 8)
            .unwrap_or(0);
        data_bytes + bm_bytes + self.name.capacity()
    }
}

/// Contiguous in-memory column storing 64-bit IEEE-754 floating point numbers (`f64`).
#[derive(Debug, Clone, PartialEq)]
pub struct Float64Column {
    name: String,
    data: Vec<f64>,
    null_bitmap: Option<NullBitmap>,
}

impl Float64Column {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data: Vec::new(),
            null_bitmap: None,
        }
    }

    pub fn with_capacity(name: impl Into<String>, capacity: usize) -> Self {
        Self {
            name: name.into(),
            data: Vec::with_capacity(capacity),
            null_bitmap: None,
        }
    }

    #[inline(always)]
    pub fn null_bitmap(&self) -> Option<&NullBitmap> {
        self.null_bitmap.as_ref()
    }

    pub fn from_slice(name: impl Into<String>, slice: &[f64]) -> Self {
        Self {
            name: name.into(),
            data: slice.to_vec(),
            null_bitmap: None,
        }
    }

    pub fn from_iter<I: IntoIterator<Item = f64>>(name: impl Into<String>, iter: I) -> Self {
        let data: Vec<f64> = iter.into_iter().collect();
        Self {
            name: name.into(),
            data,
            null_bitmap: None,
        }
    }

    pub fn from_optional_iter<I: IntoIterator<Item = Option<f64>>>(
        name: impl Into<String>,
        iter: I,
    ) -> Self {
        let name = name.into();
        let items: Vec<Option<f64>> = iter.into_iter().collect();
        let mut data = Vec::with_capacity(items.len());
        let mut bm = NullBitmap::new(items.len(), true);
        let mut has_null = false;

        for (i, opt) in items.into_iter().enumerate() {
            match opt {
                Some(val) => data.push(val),
                None => {
                    data.push(0.0);
                    bm.set_null(i);
                    has_null = true;
                }
            }
        }

        Self {
            name,
            data,
            null_bitmap: if has_null { Some(bm) } else { None },
        }
    }

    pub fn push(&mut self, val: f64) {
        if let Some(ref mut bm) = self.null_bitmap {
            bm.append_valid();
        }
        self.data.push(val);
    }

    pub fn push_null(&mut self) {
        if self.null_bitmap.is_none() {
            let mut bm = NullBitmap::new(self.data.len(), true);
            bm.append_null();
            self.null_bitmap = Some(bm);
        } else if let Some(ref mut bm) = self.null_bitmap {
            bm.append_null();
        }
        self.data.push(0.0);
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    #[inline(always)]
    pub fn get(&self, idx: usize) -> Option<f64> {
        if idx >= self.data.len() {
            return None;
        }
        if let Some(ref bm) = self.null_bitmap {
            if bm.is_null(idx) {
                return None;
            }
        }
        Some(self.data[idx])
    }

    #[inline(always)]
    pub fn value_unchecked(&self, idx: usize) -> f64 {
        self.data[idx]
    }

    #[inline(always)]
    pub fn has_nulls(&self) -> bool {
        self.null_bitmap
            .as_ref()
            .map(|bm| bm.null_count() > 0)
            .unwrap_or(false)
    }

    #[inline(always)]
    pub fn chunks(&self, chunk_size: usize) -> std::slice::Chunks<'_, f64> {
        self.data.chunks(chunk_size)
    }
}

impl Column for Float64Column {
    fn name(&self) -> &str {
        &self.name
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn column_type(&self) -> ColumnType {
        ColumnType::Float64
    }

    fn null_bitmap(&self) -> Option<&NullBitmap> {
        self.null_bitmap.as_ref()
    }

    fn is_null(&self, idx: usize) -> bool {
        self.null_bitmap
            .as_ref()
            .map(|bm| bm.is_null(idx))
            .unwrap_or(false)
    }

    fn memory_bytes(&self) -> usize {
        let data_bytes = self.data.capacity() * std::mem::size_of::<f64>();
        let bm_bytes = self
            .null_bitmap
            .as_ref()
            .map(|bm| bm.words().len() * 8)
            .unwrap_or(0);
        data_bytes + bm_bytes + self.name.capacity()
    }
}
