# Spark Optimization Report

## Workload
The same analytics workload was executed twice: it loads and cleans the retail
transactions, then runs several aggregations that **reuse** the working
DataFrames (revenue by country, top products, per-customer aggregates, and a
daily-sales table used for three further aggregations).

## Optimizations applied

### 1. Partitioning
The working set is repartitioned by **`Country`** into
**8 partitions** (`repartition(8, "Country")`).
8 partitions matches the available local cores, avoiding both the overhead of
hundreds of tiny tasks and the under-parallelism of too few.

### 2. Caching
Two DataFrames that are reused by multiple actions are cached with `.cache()`:
- the **cleaned base table** (reused by 3 aggregations + the daily rollup), and
- the **daily-sales table** (reused by 3 further aggregations).

Without caching, each action recomputes the whole read → filter → derive chain.
Cache usage is visible in the Spark UI **Storage** tab and as skipped stages in
the **Stages** tab.

### 3. Configuration tuning
| Parameter | Baseline | Optimized |
|-----------|----------|-----------|
| `spark.sql.shuffle.partitions` | 200 | 8 |
| `spark.sql.adaptive.enabled` | false | true |
| `spark.sql.adaptive.coalescePartitions.enabled` | false | true |

(On a local single-machine setup, driver memory acts as executor memory; the
shuffle-partition count and Adaptive Query Execution are the parameters with the
most impact here.)

## Before / after execution time
| Run | Configuration | Execution time |
|-----|---------------|----------------|
| **Before** | baseline (200 shuffle partitions, no cache, AQE off) | **45.37 s** |
| **After**  | optimized (8 partitions, cached, AQE on) | **10.79 s** |

**Improvement: 76.2% faster** (45.37s → 10.79s).

Spark UI screenshots (Stages tab):
- `screenshots/spark_ui_before.png`
- `screenshots/spark_ui_after.png`

## Summary (which optimization mattered most, and why)
**Caching had by far the biggest impact.** The workload reuses the same cleaned
base table and daily-sales table across six separate actions. In the baseline,
Spark is lazy and stateless, so every action re-executes the full read → filter
→ derive-column chain from the CSV on disk — the expensive work is repeated six
times. Calling `.cache()` (and materialising it once) keeps the computed rows in
memory, so the five later actions skip that recomputation entirely, which the
Spark UI shows as skipped stages. Cutting `spark.sql.shuffle.partitions` from 200
to 8 was the second biggest win: with only a few cores, 200 shuffle partitions
create hundreds of tiny tasks whose scheduling overhead dwarfs the actual work,
whereas 8 partitions keep every core busy without that overhead. Adaptive Query
Execution and key-based repartitioning added smaller, complementary gains.
