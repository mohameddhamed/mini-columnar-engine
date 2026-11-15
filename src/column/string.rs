use super::bitmap::NullBitmap;
use super::{Column, ColumnType};

/// Contiguous Raw String Column storing variable-length string data in Arrow format:
/// an offset array (`Vec<usize>`) pointing into a flat byte buffer (`Vec<u8>`).
///
/// Used for direct analytical benchmarks against `DictionaryColumn` to demonstrate
/// the cache thrashing and string comparison overhead of uncompressed raw strings.
#[derive(Debug, Clone, PartialEq)]
pub struct StringColumn {
    name: String,
    offsets: Vec<usize>,
    data: Vec<u8>,
    null_bitmap: Option<NullBitmap>,
}

impl StringColumn {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            offsets: vec![0],
            data: Vec::new(),
            null_bitmap: None,
        }
    }

    pub fn with_capacity(
        name: impl Into<String>,
        row_capacity: usize,
        byte_capacity: usize,
    ) -> Self {
        let mut offsets = Vec::with_capacity(row_capacity + 1);
        offsets.push(0);
        Self {
            name: name.into(),
            offsets,
            data: Vec::with_capacity(byte_capacity),
            null_bitmap: None,
        }
    }

    pub fn from_iter<S: AsRef<str>, I: IntoIterator<Item = S>>(
        name: impl Into<String>,
        iter: I,
    ) -> Self {
        let mut col = Self::new(name);
        for s in iter {
            col.push(s.as_ref());
        }
        col
    }

    pub fn push(&mut self, val: &str) {
        if let Some(ref mut bm) = self.null_bitmap {
            bm.append_valid();
        }
        self.data.extend_from_slice(val.as_bytes());
        self.offsets.push(self.data.len());
    }

    pub fn push_null(&mut self) {
        if self.null_bitmap.is_none() {
            let row_count = self.offsets.len() - 1;
            let mut bm = NullBitmap::new(row_count, true);
            bm.append_null();
            self.null_bitmap = Some(bm);
        } else if let Some(ref mut bm) = self.null_bitmap {
            bm.append_null();
        }
        self.offsets.push(self.data.len());
    }

    #[inline(always)]
    pub fn get(&self, idx: usize) -> Option<&str> {
        if idx + 1 >= self.offsets.len() {
            return None;
        }
        if let Some(ref bm) = self.null_bitmap {
            if bm.is_null(idx) {
                return None;
            }
        }
        let start = self.offsets[idx];
        let end = self.offsets[idx + 1];
        let bytes = &self.data[start..end];
        // Safety: all inserted data came from valid Rust &str
        Some(unsafe { std::str::from_utf8_unchecked(bytes) })
    }

    /// Linear scan comparing raw string slices one by one (baseline benchmark).
    pub fn scan_equals_raw(&self, target: &str) -> Vec<usize> {
        let target_bytes = target.as_bytes();
        let num_rows = self.len();
        let mut matches = Vec::new();

        for i in 0..num_rows {
            if let Some(ref bm) = self.null_bitmap {
                if bm.is_null(i) {
                    continue;
                }
            }
            let start = self.offsets[i];
            let end = self.offsets[i + 1];
            if &self.data[start..end] == target_bytes {
                matches.push(i);
            }
        }

        matches
    }
}

impl Column for StringColumn {
    fn name(&self) -> &str {
        &self.name
    }

    fn len(&self) -> usize {
        if self.offsets.is_empty() {
            0
        } else {
            self.offsets.len() - 1
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn column_type(&self) -> ColumnType {
        ColumnType::String
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
        let offsets_bytes = self.offsets.capacity() * std::mem::size_of::<usize>();
        let data_bytes = self.data.capacity();
        let bm_bytes = self
            .null_bitmap
            .as_ref()
            .map(|bm| bm.words().len() * 8)
            .unwrap_or(0);
        offsets_bytes + data_bytes + bm_bytes + self.name.capacity()
    }
}
