/// Compact, bit-packed null mask.
///
/// In modern columnar stores (such as SAP HANA and Apache Arrow), nulls are decoupled
/// from the contiguous payload buffer using a 1-bit-per-row bitmap rather than wrapping
/// each value in a tagged union (e.g. `Option<T>`).
///
/// Bit convention:
/// - `1` = Valid (Not Null)
/// - `0` = Null
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NullBitmap {
    words: Vec<u64>,
    len: usize,
}

impl NullBitmap {
    /// Create a new bitmap with `len` rows, all set to either valid (true) or null (false).
    pub fn new(len: usize, default_valid: bool) -> Self {
        let num_words = len.div_ceil(64);
        let word_val = if default_valid { u64::MAX } else { 0 };
        let mut words = vec![word_val; num_words];

        // Mask out excess bits in the last word if valid
        if default_valid && !len.is_multiple_of(64) && num_words > 0 {
            let remainder = len % 64;
            words[num_words - 1] = (1u64 << remainder) - 1;
        }

        Self { words, len }
    }

    /// Construct from a slice of booleans where `true` indicates valid.
    pub fn from_bools(bools: &[bool]) -> Self {
        let mut bm = Self::new(bools.len(), false);
        for (i, &valid) in bools.iter().enumerate() {
            if valid {
                bm.set_valid(i);
            }
        }
        bm
    }

    /// Number of logical rows tracked by this bitmap.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if there are zero rows.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns true if the value at row `idx` is valid (not null).
    #[inline(always)]
    pub fn is_valid(&self, idx: usize) -> bool {
        debug_assert!(idx < self.len, "Index {} out of bounds {}", idx, self.len);
        let word_idx = idx / 64;
        let bit_offset = idx % 64;
        (self.words[word_idx] & (1u64 << bit_offset)) != 0
    }

    /// Returns true if the value at row `idx` is null.
    #[inline(always)]
    pub fn is_null(&self, idx: usize) -> bool {
        !self.is_valid(idx)
    }

    /// Mark row `idx` as valid (not null).
    #[inline(always)]
    pub fn set_valid(&mut self, idx: usize) {
        debug_assert!(idx < self.len);
        let word_idx = idx / 64;
        let bit_offset = idx % 64;
        self.words[word_idx] |= 1u64 << bit_offset;
    }

    /// Mark row `idx` as null.
    #[inline(always)]
    pub fn set_null(&mut self, idx: usize) {
        debug_assert!(idx < self.len);
        let word_idx = idx / 64;
        let bit_offset = idx % 64;
        self.words[word_idx] &= !(1u64 << bit_offset);
    }

    /// Append a new row bit to the end of the bitmap.
    pub fn append(&mut self, valid: bool) {
        let idx = self.len;
        let word_idx = idx / 64;
        let bit_offset = idx % 64;

        if word_idx >= self.words.len() {
            self.words.push(0);
        }

        if valid {
            self.words[word_idx] |= 1u64 << bit_offset;
        } else {
            self.words[word_idx] &= !(1u64 << bit_offset);
        }
        self.len += 1;
    }

    /// Append a valid entry.
    #[inline(always)]
    pub fn append_valid(&mut self) {
        self.append(true);
    }

    /// Append a null entry.
    #[inline(always)]
    pub fn append_null(&mut self) {
        self.append(false);
    }

    /// Count total null entries across the entire column.
    pub fn null_count(&self) -> usize {
        let mut valid_count = 0usize;
        let full_words = self.len / 64;

        for &w in &self.words[..full_words] {
            valid_count += w.count_ones() as usize;
        }

        let remainder = self.len % 64;
        if remainder > 0 && self.words.len() > full_words {
            let last_word = self.words[full_words];
            let mask = (1u64 << remainder) - 1;
            valid_count += (last_word & mask).count_ones() as usize;
        }

        self.len - valid_count
    }

    /// Fast bitwise AND with another bitmap (e.g. for multi-column predicate null propagation).
    pub fn bit_and(&self, other: &Self) -> Self {
        assert_eq!(self.len, other.len, "Bitmap length mismatch in bit_and");
        let words = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(&w1, &w2)| w1 & w2)
            .collect();
        Self {
            words,
            len: self.len,
        }
    }

    /// Fast bitwise OR with another bitmap.
    pub fn bit_or(&self, other: &Self) -> Self {
        assert_eq!(self.len, other.len, "Bitmap length mismatch in bit_or");
        let words = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(&w1, &w2)| w1 | w2)
            .collect();
        Self {
            words,
            len: self.len,
        }
    }

    /// Access raw 64-bit words for SIMD/bit-parallel scan processing.
    #[inline(always)]
    pub fn words(&self) -> &[u64] {
        &self.words
    }
}
