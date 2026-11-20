# mini-columnar-engine

[![CI](https://github.com/mohameddhamed/mini-columnar-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/mohameddhamed/mini-columnar-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust: 1.80+](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)

**An in-memory prototype of typed columnar analytical operators in Rust.**

The operators use contiguous columns, bit-packed null masks, dictionary encoding, selection vectors and direct-indexed grouping. Grouped results can omit null keys and groups whose values are all null.

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
  - **Bit-Parallel Null Skipping:** Vectorized scans inspect raw 64-bit words; entire 64-row null chunks (`word == 0`) are skipped as a unit, while fully valid words (`word == u64::MAX`) bypass null checks entirely for maximum loop unrolling throughput.
  - **Scratch Buffer Arena (`SelectionVectorPool`):** In-place filtering (`scan_filter_*_into`) allows reuse of pre-allocated selection vectors across query stages, achieving zero heap allocations on the hot path.

### Pillar 2: Lightweight Database Compression (Dictionary & RLE)

- **Dictionary-Encoded Strings (`DictionaryColumn`):**
  - Unique string tokens are stored once in a contiguous dictionary pool (`Vec<String>`).
  - Column cells store compact 32-bit integer surrogate keys (`Vec<u32>`).
  - **Bit-Packed Surrogates:** Computes optimal bit-width $\lceil\log_2(\text{cardinality})\rceil$ via `bits_per_key()` to support bit-packing dense categorical dimensions.
  - **Defense-in-Depth Sentinel:** Null rows are assigned `NULL_SURROGATE = u32::MAX`, ensuring direct array aggregations never corrupt group 0 even before bitmap evaluation.
  - **Instant Predicate Evaluation:** A string filter (e.g. `region == "EMEA"`) resolves `"EMEA"` to its surrogate integer ID $k$ in $O(1)$ via dictionary lookup. If the token does not exist in the dictionary, the engine **prunes the entire scan instantly in $O(1)$**. If found, the engine executes a contiguous 32-bit integer comparison loop across `keys`, eliminating all string byte comparisons and pointer dereferences during query execution.
  - **Direct Boolean Lookup for Range Comparisons:** Range filters (`>`, `<`, `>=`, `<=`) precompute a dense `Vec<bool>` indexed by surrogate ID, replacing runtime hash-table probes with single L1 array accesses.
- **Run-Length Encoding (`RleColumn`):**
  - Consecutive repeated values are compressed into `(value, count)` runs.
  - Enables algebraic aggregation: $\text{SUM} = \sum (\text{value}_i \times \text{count}_i)$ in $O(\text{runs})$ time rather than $O(\text{rows})$, with $O(\text{runs})$ mathematical evaluations for `MIN`, `MAX`, and `AVG`.
  - **Empty reductions:** Where applicable, scalar reductions return `None` for empty inputs.

### Pillar 3: Vectorized Execution & Predicate Pushdown

- **Chunked Vector Execution (`VectorBatch`):** Rather than invoking virtual function `next()` per row, operators process fixed-size vector chunks (1024 elements) sized to reside entirely within CPU L1 data cache (32–64 KB).
- **Morsel Chunk Pipelining (`aggregate_filtered_chunked_i64`):** Executes filter pushdown and aggregation in a single cache-conscious pass over 1024-element morsels without ever allocating a table-wide intermediate selection vector buffer.
- **Selection Vectors (`SelectionVector`):**
  - Filtering operators evaluate predicates and emit a compact array of matching row indices (`&[usize]`).
  - Downstream operators (compound filters, aggregators, project operators) consume this selection vector directly, **materializing zero intermediate rows or column copies**.
  - **Zero-Allocation Set Algebra:** High-throughput sorted two-pointer intersections and unions (`and_into`, `or_into`, `not_into`, `filter_in_place`).
  - **Bitmask & Inversion Interoperability:** Zero-overhead bidirectional conversion between index arrays and bit-packed null masks (`to_bitmap`, `from_bitmap`, `not`).
- **Loop Unrolling & SIMD Bounds Elimination:** Scans are unrolled 8-way with fixed-size slice matches (`&[i64; 8]`), completely eliminating compiler bounds check branches and allowing LLVM to generate hardware SIMD instructions (ARM64 NEON / x86_64 AVX2).

### Pillar 4: Analytical Aggregations & Direct-Indexed Group By

- **Loop-Unrolled Aggregators:** Numerical operations use independent accumulators on some scan paths and validity checks for nullable input.
- **Scalar aggregate behavior:**
  - `SUM` over all-null or empty selection yields `NULL` (`None`).
  - `AVG` strictly divides by `valid_count`, never total selection length.
  - `COUNT(col)` respects null masks, while `COUNT(*)` counts all physical rows.
- **Direct-Indexed Vectorized Group By:**
  - Grouping over dictionary-encoded columns leverages the fact that surrogate IDs form a dense range $[0, K-1]$, where $K$ is the dictionary cardinality.
  - Rather than maintaining an expensive `std::collections::HashMap` (which causes pointer chasing, hashing latency, and hash collision chains), the engine allocates a contiguous vector of accumulators `Vec<Accumulator>` of length $K$.
  - Group updates are direct array accesses: `accumulators[surrogate_id as usize] += value`.
  - Supports `SUM`, `AVG`, `MIN`, `MAX`, and `COUNT` with zero hashing, zero hash collision resolution, zero heap allocation, and 100% L1 cache residency.
  - **Group materialization:** Output can omit null grouping keys and groups whose selected values are all null.

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

Archived Apple Silicon Criterion estimates for one million integer values give 0.2525 ms for contiguous scan plus sum (about 3.96 billion elements/s). For one million rows and eight repeating keys, direct indexing took 1.025 ms, compared with 16.806 ms using owned string keys and 30.223 ms using borrowed string keys in the two hash map baselines. Dictionary construction was outside the timed comparison.

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
│   └── integration_tests.rs      # 32 comprehensive integration & edge-case tests
├── .github/
│   └── workflows/ci.yml          # GitHub Actions CI workflow (fmt, clippy, test, bench)
└── README.md                     # Engineering whitepaper & benchmark report
```

---

## Author

**Mohamed Hamed**  
*M.Sc. in Software Architecture @ ELTE*  
*SAP & Ex-ExxonMobil Systems Engineer*
