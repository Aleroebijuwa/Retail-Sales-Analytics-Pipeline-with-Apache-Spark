"""
Data Cleaning - Retail Sales Analytics Pipeline (Step 2a)
=========================================================

Transforms the raw UCI Online Retail transactions into a clean, analysis-ready
Spark DataFrame by:

  1. Handling missing values with a documented, per-column strategy.
  2. Removing invalid business records (cancellations / non-positive values).
  3. Detecting and handling outliers with the IQR method (winsorization).

The cleaned data is returned as a Spark DataFrame (via `clean_data`) and can be
persisted to disk (via `main`). `feature_engineering.py` consumes the output.

Run standalone:
    python data_cleaning.py
"""

import os

from pyspark.sql import functions as F

# Reuse the dataset download/loader from Step 1 so we have a single source of
# truth for where the data lives and what its schema is.
from data_ingestion import ensure_dataset, build_spark, load_dataframe, DATA_DIR

# Where the cleaned dataset is written (git-ignored; regenerated on demand).
CLEANED_CSV = os.path.join(DATA_DIR, "cleaned_retail.csv")


# --------------------------------------------------------------------------- #
# Step 1: Handle missing values
# --------------------------------------------------------------------------- #
def handle_missing_values(df):
    """Handle nulls with a per-column strategy. Justification inline.

    Null profile of the raw Online Retail data:
      * CustomerID  ~ 135,000 nulls (~25%)  -> guest / unregistered orders
      * Description ~ 1,450 nulls           -> corrupt / adjustment rows
      * All other columns are effectively complete.
    """
    n_before = df.count()

    # --- CustomerID -------------------------------------------------------- #
    # DROP rows with a null CustomerID. Rationale: CustomerID is an *identity*
    # key, not a measurable quantity, so mean/median/mode imputation is
    # meaningless (you cannot invent a real customer). These rows also cannot
    # feed the customer-level behaviour aggregations built in Step 2b, so we
    # remove them rather than fabricate identities.
    df = df.filter(F.col("CustomerID").isNotNull())

    # --- Description ------------------------------------------------------- #
    # DROP rows with a null/blank Description. Rationale: these are typically
    # non-product adjustment lines (postage, bad debt, manual corrections);
    # they are few (<0.5%) and carry no product meaning, so imputing a "mode"
    # description would create fake product records.
    df = df.filter(
        F.col("Description").isNotNull() & (F.trim(F.col("Description")) != "")
    )

    # --- UnitPrice --------------------------------------------------------- #
    # IMPUTE any remaining null UnitPrice with the column MEDIAN. Rationale:
    # price is a continuous numeric value where the median is robust to the
    # heavy right-skew of retail prices (far better than the mean, which the
    # few very expensive items would inflate).
    if df.filter(F.col("UnitPrice").isNull()).count() > 0:
        median_price = df.approxQuantile("UnitPrice", [0.5], 0.01)[0]
        df = df.fillna({"UnitPrice": median_price})

    n_after = df.count()
    print(f"[missing] rows: {n_before:,} -> {n_after:,} "
          f"({n_before - n_after:,} removed)")
    return df


# --------------------------------------------------------------------------- #
# Step 2: Remove invalid business records
# --------------------------------------------------------------------------- #
def remove_invalid_records(df):
    """Drop rows that are not valid completed sales.

    * InvoiceNo starting with 'C' = a cancellation/return (not a sale).
    * Quantity <= 0 or UnitPrice <= 0 = returns, freebies or data errors.
    These are business-rule invalids, distinct from statistical outliers.
    """
    n_before = df.count()
    df = df.filter(
        (~F.col("InvoiceNo").startswith("C"))
        & (F.col("Quantity") > 0)
        & (F.col("UnitPrice") > 0)
    )
    n_after = df.count()
    print(f"[invalid] rows: {n_before:,} -> {n_after:,} "
          f"({n_before - n_after:,} removed)")
    return df


# --------------------------------------------------------------------------- #
# Step 3: Detect and handle outliers (IQR method)
# --------------------------------------------------------------------------- #
def handle_outliers_iqr(df, columns, k=1.5):
    """Cap (winsorize) outliers using the Inter-Quartile Range rule.

    For each column: bounds = [Q1 - k*IQR, Q3 + k*IQR], IQR = Q3 - Q1.

    We CAP values to the bounds rather than DELETE the rows. Rationale: a very
    large but genuine wholesale order is still a real sale; deleting it would
    throw away information and bias revenue downward. Capping keeps every
    transaction while preventing a handful of extreme values from dominating
    later scaling/aggregation.
    """
    for col in columns:
        q1, q3 = df.approxQuantile(col, [0.25, 0.75], 0.01)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        df = df.withColumn(
            col,
            F.when(F.col(col) < lower, F.lit(lower))
            .when(F.col(col) > upper, F.lit(upper))
            .otherwise(F.col(col)),
        )
        print(f"[outliers] {col}: capped to [{lower:.2f}, {upper:.2f}] "
              f"(Q1={q1:.2f}, Q3={q3:.2f})")
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def clean_data(spark):
    """Full cleaning pipeline -> returns a cleaned Spark DataFrame."""
    ensure_dataset()
    df = load_dataframe(spark)

    df = handle_missing_values(df)
    df = remove_invalid_records(df)
    df = handle_outliers_iqr(df, ["Quantity", "UnitPrice"])

    # Derived monetary column used downstream.
    df = df.withColumn("Revenue", F.round(F.col("Quantity") * F.col("UnitPrice"), 2))
    return df


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    cleaned = clean_data(spark)

    print("\n=== Cleaned data preview ===")
    cleaned.show(10, truncate=False)
    cleaned.printSchema()
    print(f"Final cleaned row count: {cleaned.count():,}")

    # Persist cleaned data for reuse (written via pandas to stay portable on
    # Windows, where Spark's native writer needs winutils/HADOOP_HOME).
    os.makedirs(DATA_DIR, exist_ok=True)
    cleaned.toPandas().to_csv(CLEANED_CSV, index=False)
    print(f"[output] Wrote cleaned dataset -> {CLEANED_CSV}")

    spark.stop()


if __name__ == "__main__":
    main()
