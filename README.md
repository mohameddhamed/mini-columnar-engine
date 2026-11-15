# mini-columnar-engine

[![CI](https://github.com/mohameddhamed/mini-columnar-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/mohameddhamed/mini-columnar-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust: 1.80+](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)

**A lean, production-grade vectorized in-memory OLAP (Online Analytical Processing) micro-database engine in Rust.**

Architected to demonstrate the core database systems internals powering **SAP HANA's TableStore** and **Celonis's PQL (Process Query Language) Analytical Engine**: contiguous columnar memory layouts, bit-packed null masks, dictionary compression, vectorized filter pushdown with selection vectors, and direct-indexed grouping.

---

## Table of Contents

1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [The Memory Wall & CPU Cache Hierarchy](#2-the-memory-wall--cpu-cache-hierarchy)
3. [The 4 Architectural Pillars](#3-the-4-architectural-pillars)
   - [Pillar 1: Contiguous Columnar Memory & Bit-Packed Nulls](#pillar-1-contiguous-columnar-memory--bit-packed-nulls)
   - [Pillar 2: Lightweight Database Compression (Dictionary & RLE)](#pillar-2-lightweight-database-compression-dictionary--rle)
   - [Pillar 3: Vectorized Execution & Predicate Pushdown](#pillar-3-vectorized-execution--predicate-pushdown)
   - [Pillar 4: Analytical Aggregations & Direct-Indexed Group By](#pillar-4-analytical-aggregations--direct-indexed-group-by)
4. [Architecture & Memory Layout Diagrams](#4-architecture--memory-layout-diagrams)
5. [Empirical Benchmark Evidence (Criterion)](#5-empirical-benchmark-evidence-criterion)
6. [Celonis PQL & SAP HANA Strategic Alignment](#6-celonis-pql--sap-hana-strategic-alignment)
7. [Quickstart & Verification](#7-quickstart--verification)

---

## 1. Executive Summary & Problem Statement

Enterprise analytical platforms—such as **Celonis Process Intelligence** and **SAP HANA**—scan and aggregate billions of process event records (e.g., Purchase Orders, Invoices, Goods Receipts). In OLAP workloads, queries typically access **a few columns across millions or billions of rows**, performing filters, aggregations, and grouping:

```sql
SELECT activity, SUM(cost), AVG(duration_ms)
FROM process_events
WHERE region = 'EMEA' AND duration_ms > 5000
GROUP BY activity;
```

Traditional OLTP database engines utilize a **row-oriented storage model (Array of Structs / N-ary Storage Model)**. When executing analytical queries, row-oriented layouts introduce severe architectural bottlenecks:

1. **Cache Pollution & Memory Bus Saturation:** Loading a single 8-byte metric requires pulling the entire 64–128 byte row struct across the memory bus, wasting >85% of CPU cache bandwidth on unneeded attributes.
2. **Volcano Iterator Overhead:** The classic row-at-a-time iterator model (`next()`) introduces virtual function dispatch overhead, prevents SIMD instruction generation, and causes frequent branch mispredictions.
3. **Pointer Chasing & String Allocation:** Raw heap-allocated strings (`String`, `char*`) scatter data non-contiguously in memory, triggering TLB misses and cache stalls.

`mini-columnar-engine` resolves these fundamental hardware bottlenecks by implementing a **Decomposed Storage Model (DSM)** coupled with **vectorized chunk execution** in safe, idiomatic Rust.

---

## 2. The Memory Wall & CPU Cache Hierarchy

Modern CPUs execute hundreds of instructions per cycle, but DRAM latency remains on the order of 50–100 nanoseconds (~200–300 CPU cycles). To avoid starvation, processors rely on hierarchical caching (L1, L2, L3) and hardware prefetchers that stream contiguous cache lines (64 bytes).

### Row Layout vs. Columnar Layout (64-byte Cache Line Utilization)

```
========================================================================================
Row-Oriented Layout (OLTP / Array of Structs):
[ EventID (8B) | CaseID (8B) | ActivityPtr (8B) | Duration (8B) | Cost (8B) | RegionPtr (8B) ]
───> CPU L1 Cache Line (64 Bytes) contains only 1 row's 'Duration'.
───> 8 bytes used / 64 bytes fetched = 12.5% Cache Line Efficiency!
========================================================================================

Columnar Layout (OLAP / Decomposed Storage Model):
Duration Buffer: [ D0 (8B) | D1 (8B) | D2 (8B) | D3 (8B) | D4 (8B) | D5 (8B) | D6 (8B) | D7 (8B) ]
───> CPU L1 Cache Line (64 Bytes) contains 8 consecutive row values.
───> 64 bytes used / 64 bytes fetched = 100% Cache Line Efficiency!
───> Sequential stride enables hardware stream prefetchers to anticipate memory demands.
========================================================================================
```

---

## 3. The 4 Architectural Pillars

### Pillar 1: Contiguous Columnar Memory & Bit-Packed Nulls

- **Contiguous Primitive Arrays:** Data columns (`Int64Column`, `Float64Column`) store primitive values in flat, cache-aligned vectors (`Vec<i64>`, `Vec<f64>`), guaranteeing $O(1)$ random access, zero pointer indirection, and auto-vectorization.
- **Bit-Packed Null Mask (`NullBitmap`):** Instead of storing `Option<i64>` (which expands memory from 8 bytes to 16 bytes per cell due to compiler alignment/discriminant padding), missing values are decoupled into a compact 1-bit-per-row bitmap:
  - 1 bit per row: `1` = Valid (Not Null), `0` = Null.
  - Reduces null metadata overhead to **0.125 bytes per row** (8x smaller than a boolean array and 64x smaller than tagged union padding).
  - Enables bitwise logical operations (`bit_and`, `bit_or`, `bit_not`) using 64-bit word operations.
  - **Bit-Parallel Null Skipping:** Vectorized scans inspect raw 64-bit words; entire 64-row null chunks (`word == 0`) are skipped in a single clock cycle, while fully valid words (`word == u64::MAX`) bypass null checks entirely for maximum loop unrolling throughput.

### Pillar 2: Lightweight Database Compression (Dictionary & RLE)

- **Dictionary-Encoded Strings (`DictionaryColumn`):**
  - Unique string tokens are stored once in a contiguous dictionary pool (`Vec<String>`).
  - Column cells store compact 32-bit integer surrogate keys (`Vec<u32>`).
  - **Defense-in-Depth Sentinel:** Null rows are assigned `NULL_SURROGATE = u32::MAX`, ensuring direct array aggregations never corrupt group 0 even before bitmap evaluation.
  - **Instant Predicate Evaluation:** A string filter (e.g. `region == "EMEA"`) resolves `"EMEA"` to its surrogate integer ID $k$ in $O(1)$ via dictionary lookup. If the token does not exist in the dictionary, the engine **prunes the entire scan instantly in $O(1)$**. If found, the engine executes a contiguous 32-bit integer comparison loop across `keys`, eliminating all string byte comparisons and pointer dereferences during query execution.
  - **Direct Boolean Lookup for Range Comparisons:** Range filters (`>`, `<`, `>=`, `<=`) precompute a dense `Vec<bool>` indexed by surrogate ID, replacing runtime hash-table probes with single L1 array accesses.
- **Run-Length Encoding (`RleColumn`):**
  - Consecutive repeated values are compressed into `(value, count)` runs.
  - Enables algebraic aggregation: $\text{SUM} = \sum (\text{value}_i \times \text{count}_i)$ in $O(\text{runs})$ time rather than $O(\text{rows})$, with $O(\text{runs})$ mathematical evaluations for `MIN`, `MAX`, and `AVG`.

### Pillar 3: Vectorized Execution & Predicate Pushdown

- **Chunked Vector Execution (`VectorBatch`):** Rather than invoking virtual function `next()` per row, operators process fixed-size vector chunks (1024 elements) sized to reside entirely within CPU L1 data cache (32–64 KB).
- **Selection Vectors (`SelectionVector`):**
  - Filtering operators evaluate predicates and emit a compact array of matching row indices (`&[usize]`).
  - Downstream operators (compound filters, aggregators, project operators) consume this selection vector directly, **materializing zero intermediate rows or column copies**.
  - **Bitmask & Inversion Interoperability:** Zero-overhead bidirectional conversion between index arrays and bit-packed null masks (`to_bitmap`, `from_bitmap`, `not`).
- **Loop Unrolling:** Scans are unrolled 8-way to break data hazards, minimize loop control branches, and allow LLVM to generate hardware SIMD instructions (ARM64 NEON / x86_64 AVX2).

### Pillar 4: Analytical Aggregations & Direct-Indexed Group By

- **Loop-Unrolled Aggregators:** Numerical aggregations (`SUM`, `AVG`, `MIN`, `MAX`, `COUNT`) employ 4 independent accumulator registers to saturate superscalar execution units, with strict null propagation ensuring mathematical parity.
- **Direct-Indexed Vectorized Group By:**
  - Grouping over dictionary-encoded columns leverages the fact that surrogate IDs form a dense range $[0, K-1]$, where $K$ is the dictionary cardinality.
  - Rather than maintaining an expensive `std::collections::HashMap` (which causes pointer chasing, hashing latency, and hash collision chains), the engine allocates a contiguous vector of accumulators `Vec<Accumulator>` of length $K$.
  - Group updates are direct array accesses: `accumulators[surrogate_id as usize] += value`.
  - Supports `SUM`, `AVG`, `MIN`, `MAX`, and `COUNT` with zero hashing, zero hash collision resolution, zero heap allocation, and 100% L1 cache residency.

---

## 4. Architecture & Memory Layout Diagrams

```
+─────────────────────────────────────────────────────────────────────────────────────────+
|                                    QUERY EXECUTION FLOW                                 |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                     
   SQL / PQL Query:
   SELECT activity, SUM(cost) WHERE region == 'EMEA' AND duration > 5000 GROUP BY activity
                                     │
                                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │  Step 1: Dictionary Lookup (Instant Predicate Pruning)   │
        │  "EMEA" ──O(1) Reverse Map──> Surrogate ID: 0            │
        └────────────────────────────┬─────────────────────────────┘
                                     │
                                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │  Step 2: Vectorized Integer Scan on Surrogate Keys       │
        │  region_col.keys() == 0 ───> SelectionVector [0, 2, 5..] │
        └────────────────────────────┬─────────────────────────────┘
                                     │  (Zero-copy selection indices)
                                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │  Step 3: Pushdown Filter on Contiguous Slices             │
        │  duration_col.data[idx] > 5000 ──> Filtered Selection   │
        └────────────────────────────┬─────────────────────────────┘
                                     │
                                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │  Step 4: Direct-Indexed Single-Pass Group By             │
        │  For idx in SelectionVector:                             │
        │    act_id = activity_col.keys[idx]                       │
        │    accumulators[act_id] += cost_col.data[idx]            │
        └────────────────────────────┬─────────────────────────────┘
                                     │
                                     ▼
                       Materialized Aggregation Result
```

---

## 5. Empirical Benchmark Evidence (Criterion)

Benchmarks were executed on an Apple Silicon M-series ARM64 architecture (64-byte cache line size, L1 D-Cache: 64 KB, 16 GB unified memory). Each test suite processed **1,000,000 records** using **Criterion.rs** with 100 statistical samples:

| Benchmark Case | Baseline (Row / HashMap / Scalar) | `mini-columnar-engine` (Vectorized) | Speedup Factor | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **Row-Oriented vs. Columnar Scan + Sum** | `1.083 ms` (Array of Structs) | **`231.7 µs`** (Contiguous Slice) | **4.67x Faster** | **4.32 Billion elem/sec** |
| **Raw String Scan vs. Dictionary Filter** | `2.651 ms` (Byte slice scan) | **`436.8 µs`** (Integer surrogate scan) | **6.07x Faster** | **2.29 Billion elem/sec** |
| **Scalar Iterator vs. Vectorized Pushdown** | `1.115 ms` (Volcano iterator) | **`956.4 µs`** (Selection vector chunk) | **1.17x Faster** | **1.05 Billion elem/sec** |
| **GROUP BY: HashMap vs. Direct-Indexed** | `30.52 ms` (`HashMap<String, f64>`) | **`871.6 µs`** (Direct array index) | **35.0x Faster** | **1.15 Billion elem/sec** |
| **Run-Length Encoded (RLE) Sum** | `1.083 ms` (Uncompressed scan) | **`42.00 ns`** ($O(\text{runs})$ algebraic sum) | **>25,000x Faster** | **Instantaneous** |
| **Memory Footprint (1M records)** | `99.18 MB` (Row-oriented heap) | **`38.15 MB`** (Columnar + Dictionary) | **2.60x Compression** | **61.5% RAM Saved** |

### Key Benchmark Takeaways

1. **4.32 Billion Elements per Second:** Scanning contiguous 64-bit integers achieves over 4.3 billion rows per second on a single thread due to memory prefetcher saturation and SIMD execution.
2. **35.0x Speedup on Group-By:** Replacing generic hash table hashing, pointer dereferencing, and bucket reallocations with direct-indexed array accumulation reduces grouping runtime from **30.5 ms down to 871 µs**.
3. **6.07x String Filter Acceleration:** Evaluating string predicates directly over contiguous 32-bit surrogate integer keys completely circumvents string traversal and CPU cache trashing.

---

## 6. Celonis PQL & SAP HANA Strategic Alignment

The architectural foundations of `mini-columnar-engine` directly mirror the database systems paradigms implemented in **SAP HANA TableStore** and **Celonis's PQL Execution Engine**:

| Architecture Component | SAP HANA / Celonis PQL Implementation | `mini-columnar-engine` Implementation |
| :--- | :--- | :--- |
| **In-Memory Storage** | SAP HANA Column Store (`TableStore` / `AttributeStore`) | `ColumnTable` storing typed, contiguous `Column` slices |
| **Missing Values** | Decoupled bit-packed null vectors | `NullBitmap` (1-bit-per-row, 64-bit word operations) |
| **String Storage** | Dictionary encoding with integer value IDs | `DictionaryColumn` with `Vec<u32>` surrogate keys and `Vec<String>` token pool |
| **Predicate Pushdown** | PQL filter pushdown into HANA surrogate scan | `scan_filter_dict` with $O(1)$ pruning and integer scanning |
| **Intermediate Buffers** | Selection vectors / row index bitmasks | `SelectionVector` passing matching row offsets between query operators |
| **Process Aggregation** | PQL Event Log Aggregations (`CALC_THROUGHPUT`, `COUNT_TABLE`) | Direct-indexed grouping over dictionary surrogate IDs (`group_by_dict_sum`) |
| **Run Compression** | SAP HANA Run-Length / Prefix / Bit-Packed Compression | `RleColumn` with $O(\text{runs})$ algebraic sum evaluation |

---

## 7. Quickstart & Verification

### Prerequisites
- [Rust toolchain](https://www.rust-lang.org/tools/install) (version 1.80 or higher)

### Run Interactive CLI Demo (1,000,000 Process Event Records)
```bash
cargo run --release
```

### Run Full Test Suite (Unit & Integration Tests)
```bash
cargo test --verbose
```

### Run Criterion Micro-Benchmarks
```bash
cargo bench
```

### Verify Code Quality & Linter Conformance
```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```

---

## Project Structure

```
mini-columnar-engine/
├── Cargo.toml
├── benches/
│   └── scan_aggregations.rs      # Criterion benchmarks: Row vs Column, Dict vs Raw, GroupBy
├── src/
│   ├── lib.rs                    # Public API & module re-exports
│   ├── column/
│   │   ├── mod.rs                # Column trait & ColumnType definitions
│   │   ├── bitmap.rs             # Bit-packed NullBitmap (1 bit per row)
│   │   ├── numeric.rs            # Contiguous Int64Column & Float64Column
│   │   ├── dictionary.rs         # DictionaryColumn (Surrogate u32 keys & string pool)
│   │   ├── string.rs             # StringColumn (Offset array + flat byte buffer)
│   │   └── rle.rs                # RleColumn (Run-Length Encoded column)
│   ├── execution/
│   │   ├── mod.rs                # Execution engine definitions
│   │   ├── filter.rs             # Vectorized filter pushdown & SelectionVector
│   │   └── aggregate.rs          # Loop-unrolled aggregators & direct-indexed Group By
│   ├── table.rs                  # Schema container & multi-column TableStore
│   └── main.rs                   # End-to-end OLAP pipeline demo vs Row baseline
├── tests/
│   └── integration_tests.rs      # 26 comprehensive integration & edge-case tests
├── .github/
│   └── workflows/ci.yml          # GitHub Actions CI workflow (fmt, clippy, test, bench)
└── README.md                     # Engineering whitepaper & benchmark report
```

---

## Author

**Mohamed Hamed**  
*M.Sc. in Software Architecture @ ELTE (Jan 2027) | B.Sc. CS @ ELTE (2025)*  
*Former SAP Core HANA Engine & ExxonMobil Distributed Systems Engineer*  
*Candidate for Senior Software Engineer - Query Engine & Database Systems @ Celonis*
