use super::bitmap::NullBitmap;
use super::{Column, ColumnType};
use std::collections::HashMap;

/// Dictionary-encoded column for low-to-medium cardinality string data.
///
/// This implements the signature compression and query acceleration architecture of
/// SAP HANA and Celonis PQL:
/// 1. Contiguous String Dictionary: Unique strings are stored once in `Vec<String>`.
/// 2. Compact Surrogate Keys: Column cells store 32-bit surrogate integer IDs in `Vec<u32>`.
/// 3. Instant Predicate Evaluation: String filters (e.g. `activity == "Approve PO"`)
///    first resolve the string to its `u32` surrogate ID via dictionary lookup (O(1)),
///    then execute the scan directly over contiguous integer slices without touching
///    string bytes or triggering heap pointer indirection.
#[derive(Debug, Clone, PartialEq)]
pub struct DictionaryColumn {
    name: String,
    /// Unique string tokens pool: token at index `i` has surrogate ID `i`.
    dictionary: Vec<String>,
    /// Reverse lookup map for O(1) surrogate ID encoding during ingestion.
    reverse_map: HashMap<String, u32>,
    /// Contiguous array of surrogate integer IDs for each row.
    keys: Vec<u32>,
    /// Bit-packed null mask.
    null_bitmap: Option<NullBitmap>,
}

impl DictionaryColumn {
    /// Create a new empty `DictionaryColumn`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dictionary: Vec::new(),
            reverse_map: HashMap::new(),
            keys: Vec::new(),
            null_bitmap: None,
        }
    }

    /// Pre-allocate capacity for rows and anticipated unique dictionary values.
    pub fn with_capacity(
        name: impl Into<String>,
        row_capacity: usize,
        dict_capacity: usize,
    ) -> Self {
        Self {
            name: name.into(),
            dictionary: Vec::with_capacity(dict_capacity),
            reverse_map: HashMap::with_capacity(dict_capacity),
            keys: Vec::with_capacity(row_capacity),
            null_bitmap: None,
        }
    }

    /// Construct from an iterator of string slices (all non-null).
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

    /// Construct from an iterator of optional string slices.
    pub fn from_optional_iter<S: AsRef<str>, I: IntoIterator<Item = Option<S>>>(
        name: impl Into<String>,
        iter: I,
    ) -> Self {
        let mut col = Self::new(name);
        for opt in iter {
            match opt {
                Some(s) => {
                    col.push(s.as_ref());
                }
                None => {
                    col.push_null();
                }
            }
        }
        col
    }

    /// Access null bitmap directly.
    #[inline(always)]
    pub fn null_bitmap(&self) -> Option<&NullBitmap> {
        self.null_bitmap.as_ref()
    }

    /// Push a string value into the column, encoding it with a surrogate ID.
    pub fn push(&mut self, val: &str) -> u32 {
        if let Some(ref mut bm) = self.null_bitmap {
            bm.append_valid();
        }

        let surrogate_id = if let Some(&id) = self.reverse_map.get(val) {
            id
        } else {
            let id = self.dictionary.len() as u32;
            let val_string = val.to_string();
            self.reverse_map.insert(val_string.clone(), id);
            self.dictionary.push(val_string);
            id
        };

        self.keys.push(surrogate_id);
        surrogate_id
    }

    /// Push a null value into the column.
    pub fn push_null(&mut self) {
        if self.null_bitmap.is_none() {
            let mut bm = NullBitmap::new(self.keys.len(), true);
            bm.append_null();
            self.null_bitmap = Some(bm);
        } else if let Some(ref mut bm) = self.null_bitmap {
            bm.append_null();
        }
        // Use 0 as dummy surrogate ID for null rows
        self.keys.push(0);
    }

    /// Look up the surrogate ID for a given string token in O(1) time.
    #[inline(always)]
    pub fn lookup_id(&self, val: &str) -> Option<u32> {
        self.reverse_map.get(val).copied()
    }

    /// Resolve a surrogate ID back to its string representation.
    #[inline(always)]
    pub fn resolve(&self, id: u32) -> Option<&str> {
        self.dictionary.get(id as usize).map(|s| s.as_str())
    }

    /// Get the string value at row `idx`, returning `None` if null or out of bounds.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> Option<&str> {
        if idx >= self.keys.len() {
            return None;
        }
        if let Some(ref bm) = self.null_bitmap {
            if bm.is_null(idx) {
                return None;
            }
        }
        let id = self.keys[idx];
        self.resolve(id)
    }

    /// Return raw surrogate key at row `idx` without null or bounds checking.
    #[inline(always)]
    pub fn key_unchecked(&self, idx: usize) -> u32 {
        self.keys[idx]
    }

    /// Read-only slice view of the contiguous surrogate integer keys buffer.
    #[inline(always)]
    pub fn keys(&self) -> &[u32] {
        &self.keys
    }

    /// Read-only slice view of the unique dictionary tokens.
    #[inline(always)]
    pub fn dictionary(&self) -> &[String] {
        &self.dictionary
    }

    /// Number of unique values (cardinality) stored in the dictionary.
    #[inline(always)]
    pub fn cardinality(&self) -> usize {
        self.dictionary.len()
    }

    /// Calculate memory footprint comparison vs uncompressed raw strings.
    pub fn raw_string_bytes_estimate(&self) -> usize {
        self.keys
            .iter()
            .enumerate()
            .map(|(i, &k)| {
                if self.is_null(i) {
                    0
                } else {
                    self.dictionary
                        .get(k as usize)
                        .map(|s| s.len())
                        .unwrap_or(0)
                }
            })
            .sum()
    }
}

impl Column for DictionaryColumn {
    fn name(&self) -> &str {
        &self.name
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    fn column_type(&self) -> ColumnType {
        ColumnType::Dictionary
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
        let keys_bytes = self.keys.capacity() * std::mem::size_of::<u32>();
        let dict_bytes: usize = self
            .dictionary
            .iter()
            .map(|s| std::mem::size_of::<String>() + s.capacity())
            .sum();
        let map_bytes = self.reverse_map.capacity()
            * (std::mem::size_of::<String>() + std::mem::size_of::<u32>() + 16);
        let bm_bytes = self
            .null_bitmap
            .as_ref()
            .map(|bm| bm.words().len() * 8)
            .unwrap_or(0);

        keys_bytes + dict_bytes + map_bytes + bm_bytes + self.name.capacity()
    }
}
