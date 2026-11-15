pub mod bitmap;
pub mod dictionary;
pub mod numeric;
pub mod rle;
pub mod string;

pub use bitmap::NullBitmap;
pub use dictionary::DictionaryColumn;
pub use numeric::{Float64Column, Int64Column};
pub use rle::RleColumn;
pub use string::StringColumn;

/// Data type classification for columnar engine columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColumnType {
    Int64,
    Float64,
    String,
    Dictionary,
}

impl std::fmt::Display for ColumnType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColumnType::Int64 => write!(f, "INT64"),
            ColumnType::Float64 => write!(f, "FLOAT64"),
            ColumnType::String => write!(f, "STRING"),
            ColumnType::Dictionary => write!(f, "DICT_STRING"),
        }
    }
}

/// Dynamic trait defining common capabilities for any physical column storage.
pub trait Column: std::fmt::Debug {
    /// Logical column name.
    fn name(&self) -> &str;

    /// Number of logical rows.
    fn len(&self) -> usize;

    /// True if the column contains zero rows.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Column primitive or compressed data type.
    fn column_type(&self) -> ColumnType;

    /// Null mask bitmap if the column contains missing values.
    fn null_bitmap(&self) -> Option<&NullBitmap>;

    /// Check if row `idx` is null.
    fn is_null(&self, idx: usize) -> bool;

    /// Estimated memory footprint in bytes.
    fn memory_bytes(&self) -> usize;
}
