"""
Feature Engineering - Retail Sales Analytics Pipeline (Step 2b)
==============================================================

Turns cleaned retail transactions into an analysis-ready feature set:

  1. Temporal features   (day_of_week, month, quarter, is_weekend, is_holiday, ...)
  2. Customer behaviour  (total_purchases, average_transaction_value,
                          days_since_last_purchase, and RFM-style aggregates)
  3. Feature scaling     (StandardScaler in a reusable scikit-learn Pipeline,
                          persisted to models/ with joblib)

Temporal + behaviour engineering runs in Spark (scales to the full dataset and
demonstrates a Spark UDF). The final scaling step operates on the small
customer-level table (~4.3k rows), so it uses a scikit-learn Pipeline that is
saved to disk for reuse in later modelling steps.

Run standalone:
    python feature_engineering.py
"""

import os

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType

from data_ingestion import build_spark, OUTPUT_DIR
from data_cleaning import clean_data

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# UK public / bank holidays covering the dataset window (Dec 2010 - Dec 2011).
# Used by the is_holiday feature. Kept as an explicit set to avoid an extra
# dependency and to make the definition auditable.
UK_HOLIDAYS = {
    "2010-12-25", "2010-12-26", "2010-12-27", "2010-12-28",  # Christmas period
    "2011-01-01", "2011-01-03",                              # New Year
    "2011-04-22", "2011-04-25",                              # Good Friday / Easter Mon
    "2011-04-29",                                            # Royal Wedding (bank hol)
    "2011-05-02", "2011-05-30",                              # Early May / Spring
    "2011-08-29",                                            # Summer bank holiday
    "2011-12-25", "2011-12-26", "2011-12-27",                # Christmas
}


# --------------------------------------------------------------------------- #
# Temporal features
# --------------------------------------------------------------------------- #
def create_temporal_features(df):
    """Add >= 5 temporal features derived from InvoiceDate.

    Spark's dayofweek(): 1 = Sunday ... 7 = Saturday.
    """
    # A Spark UDF that flags holidays from the explicit UK_HOLIDAYS set.
    # The UDF closes over UK_HOLIDAYS (a small set), which Spark serializes to
    # the workers automatically.
    @F.udf(returnType=BooleanType())
    def is_holiday_udf(date_str):
        return date_str in UK_HOLIDAYS

    df = (
        df.withColumn("day_of_week", F.dayofweek("InvoiceDate"))            # 1..7
        .withColumn("day_name", F.date_format("InvoiceDate", "EEEE"))       # Monday..
        .withColumn("month", F.month("InvoiceDate"))                        # 1..12
        .withColumn("quarter", F.quarter("InvoiceDate"))                    # 1..4
        .withColumn("year", F.year("InvoiceDate"))
        .withColumn("hour", F.hour("InvoiceDate"))
        .withColumn(
            "is_weekend",
            F.when(F.dayofweek("InvoiceDate").isin(1, 7), 1).otherwise(0),
        )
        .withColumn(
            "is_holiday",
            is_holiday_udf(F.date_format("InvoiceDate", "yyyy-MM-dd")).cast("int"),
        )
    )
    return df


# --------------------------------------------------------------------------- #
# Customer behaviour aggregations
# --------------------------------------------------------------------------- #
def create_customer_features(df):
    """Aggregate per-customer behaviour features (RFM + requested metrics)."""
    # Snapshot date = day after the last transaction, used for recency.
    max_date = df.agg(F.max("InvoiceDate")).collect()[0][0]

    # Revenue per invoice first, so average_transaction_value = avg per order.
    invoice_totals = df.groupBy("CustomerID", "InvoiceNo").agg(
        F.sum("Revenue").alias("invoice_revenue")
    )
    avg_txn = invoice_totals.groupBy("CustomerID").agg(
        F.round(F.avg("invoice_revenue"), 2).alias("average_transaction_value")
    )

    customer = (
        df.groupBy("CustomerID")
        .agg(
            F.countDistinct("InvoiceNo").alias("total_purchases"),        # frequency
            F.sum("Quantity").alias("total_quantity"),
            F.round(F.sum("Revenue"), 2).alias("total_revenue"),         # monetary
            F.countDistinct("StockCode").alias("unique_products"),
            F.max("InvoiceDate").alias("last_purchase_date"),
        )
        # days_since_last_purchase (recency), relative to the snapshot date.
        .withColumn(
            "days_since_last_purchase",
            F.datediff(F.lit(max_date), F.col("last_purchase_date")),
        )
        .join(avg_txn, on="CustomerID", how="left")
        .drop("last_purchase_date")
    )
    return customer


# --------------------------------------------------------------------------- #
# Scaling (reusable scikit-learn pipeline, persisted for reuse)
# --------------------------------------------------------------------------- #
def scale_and_save(customer_pdf, feature_columns):
    """Fit a StandardScaler pipeline on the numeric features and save it.

    Returns the scaled feature DataFrame. The fitted pipeline is written to
    models/ so later steps can transform new customers identically.
    """
    pipeline = Pipeline([("scaler", StandardScaler())])
    scaled_values = pipeline.fit_transform(customer_pdf[feature_columns])

    scaled_pdf = customer_pdf.copy()
    for i, col in enumerate(feature_columns):
        scaled_pdf[col] = scaled_values[:, i]

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "customer_feature_scaler.joblib")
    joblib.dump({"pipeline": pipeline, "feature_columns": feature_columns}, model_path)
    print(f"[model] Saved scaling pipeline -> {model_path}")
    return scaled_pdf


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # 1) Clean (Step 2a) -> 2) temporal features
    df = clean_data(spark)
    df = create_temporal_features(df)

    print("\n=== Transactions with temporal features (first 10) ===")
    df.select(
        "InvoiceDate", "day_of_week", "day_name", "month", "quarter",
        "is_weekend", "is_holiday",
    ).show(10, truncate=False)

    # 3) Customer behaviour aggregations
    customer = create_customer_features(df)
    print("\n=== Customer behaviour features (first 10) ===")
    customer.show(10, truncate=False)

    # Bring the small customer table to pandas for sklearn scaling.
    customer_pdf = customer.toPandas()

    numeric_features = [
        "total_purchases",
        "total_quantity",
        "total_revenue",
        "unique_products",
        "days_since_last_purchase",
        "average_transaction_value",
    ]

    # 4) Scale numeric features with a saved StandardScaler pipeline.
    scaled_pdf = scale_and_save(customer_pdf, numeric_features)

    # Persist both raw and scaled feature tables.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUTPUT_DIR, "customer_features.csv")
    scaled_path = os.path.join(OUTPUT_DIR, "customer_features_scaled.csv")
    customer_pdf.to_csv(raw_path, index=False)
    scaled_pdf.to_csv(scaled_path, index=False)
    print(f"[output] Wrote {raw_path}")
    print(f"[output] Wrote {scaled_path}")

    print("\nFeature engineering complete.")
    spark.stop()


if __name__ == "__main__":
    main()
