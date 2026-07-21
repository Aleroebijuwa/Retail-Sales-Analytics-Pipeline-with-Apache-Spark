"""
Data Ingestion and Exploration - Retail Sales Analytics Pipeline
================================================================

Loads the UCI Online Retail transaction dataset into Spark, explores its
structure, and calculates basic business metrics using both the DataFrame API
and Spark SQL.

Definition of done covered by this script:
  1. Load a retail dataset into a Spark DataFrame with an explicit schema.
  2. Display the first 10 rows and print the schema.
  3. Calculate >= 3 business metrics: total sales, average transaction value,
     and top 5 products by quantity sold (plus daily sales & country breakdown).
  4. Save the calculated metrics to CSV files in the output/ directory.

Dataset source (public):
  UCI Machine Learning Repository - "Online Retail" Data Set
  https://archive.ics.uci.edu/dataset/352/online+retail

Run:
  python data_ingestion.py

The script downloads the dataset automatically on first run (into data/) and
converts the Excel workbook to CSV so Spark can read it natively.
"""

import os
import urllib.request

import pandas as pd

import pyspark

# --------------------------------------------------------------------------- #
# Environment fix-up: make Spark start reliably regardless of any stale
# SPARK_HOME set on the machine. We point SPARK_HOME at the pip-installed
# PySpark package so its launcher scripts are always found.
# --------------------------------------------------------------------------- #
_PYSPARK_HOME = os.path.dirname(pyspark.__file__)
if os.environ.get("SPARK_HOME") != _PYSPARK_HOME:
    os.environ["SPARK_HOME"] = _PYSPARK_HOME

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    TimestampType,
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
XLSX_PATH = os.path.join(DATA_DIR, "online_retail.xlsx")
CSV_PATH = os.path.join(DATA_DIR, "online_retail.csv")

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00352/Online%20Retail.xlsx"
)


# --------------------------------------------------------------------------- #
# Step 0: Ensure the dataset is available locally (download + convert to CSV)
# --------------------------------------------------------------------------- #
def ensure_dataset():
    """Download the UCI Online Retail workbook and convert it to CSV if needed."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH):
        print(f"[data] Using cached CSV: {CSV_PATH}")
        return

    if not os.path.exists(XLSX_PATH):
        print(f"[data] Downloading dataset from {DATASET_URL} ...")
        req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(XLSX_PATH, "wb") as out:
            out.write(resp.read())
        print(f"[data] Saved workbook to {XLSX_PATH}")

    print("[data] Converting Excel workbook to CSV (one-time) ...")
    pdf = pd.read_excel(XLSX_PATH, engine="openpyxl")
    # Normalise the InvoiceDate to a string Spark can parse deterministically.
    pdf["InvoiceDate"] = pd.to_datetime(pdf["InvoiceDate"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    pdf.to_csv(CSV_PATH, index=False)
    print(f"[data] Wrote CSV: {CSV_PATH} ({len(pdf):,} rows)")


# --------------------------------------------------------------------------- #
# Step 1: Spark session + explicit schema
# --------------------------------------------------------------------------- #
def build_spark():
    return (
        SparkSession.builder.appName("RetailSalesAnalytics")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


# Explicit schema (preferred over inferSchema for correctness & speed).
RETAIL_SCHEMA = StructType(
    [
        StructField("InvoiceNo", StringType(), True),
        StructField("StockCode", StringType(), True),
        StructField("Description", StringType(), True),
        StructField("Quantity", IntegerType(), True),
        StructField("InvoiceDate", TimestampType(), True),
        StructField("UnitPrice", DoubleType(), True),
        StructField("CustomerID", DoubleType(), True),
        StructField("Country", StringType(), True),
    ]
)


def load_dataframe(spark):
    df = (
        spark.read.option("header", True)
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .schema(RETAIL_SCHEMA)
        .csv(CSV_PATH)
    )
    return df


# --------------------------------------------------------------------------- #
# Helper: write a small Spark DataFrame to a single clean CSV file
# --------------------------------------------------------------------------- #
def write_single_csv(sdf, filename):
    """Collect a small result to the driver and write one tidy CSV file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    sdf.toPandas().to_csv(path, index=False)
    print(f"[output] Wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ensure_dataset()

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    df = load_dataframe(spark)

    # ----- Step 2: Explore structure ------------------------------------- #
    print("\n=== Dataset shape ===")
    total_rows = df.count()
    print(f"Rows: {total_rows:,}, Columns: {len(df.columns)}")

    print("\n=== First 10 rows ===")
    df.show(10, truncate=False)

    print("\n=== Schema ===")
    df.printSchema()

    # Add a line-level revenue column: Quantity * UnitPrice
    df = df.withColumn("Revenue", F.col("Quantity") * F.col("UnitPrice"))

    # "Valid sales" = completed sales only: exclude cancellations (InvoiceNo
    # starting with 'C'), non-positive quantities, and non-positive prices.
    sales = df.filter(
        (~F.col("InvoiceNo").startswith("C"))
        & (F.col("Quantity") > 0)
        & (F.col("UnitPrice") > 0)
    )

    # Register a temp view so we can also demonstrate Spark SQL.
    sales.createOrReplaceTempView("sales")

    # ----- Step 3: Business metrics -------------------------------------- #

    # Metric 1: Total sales (revenue) -- DataFrame API
    total_sales = sales.agg(F.round(F.sum("Revenue"), 2).alias("total_sales")).collect()[0][
        "total_sales"
    ]

    # Metric 2: Average transaction value -- Spark SQL.
    # A "transaction" = one invoice; value = total revenue on that invoice.
    avg_txn_value = spark.sql(
        """
        SELECT ROUND(AVG(invoice_total), 2) AS avg_transaction_value
        FROM (
            SELECT InvoiceNo, SUM(Revenue) AS invoice_total
            FROM sales
            GROUP BY InvoiceNo
        )
        """
    ).collect()[0]["avg_transaction_value"]

    num_transactions = sales.select("InvoiceNo").distinct().count()
    total_quantity = sales.agg(F.sum("Quantity")).collect()[0][0]
    unique_products = sales.select("StockCode").distinct().count()
    unique_customers = sales.filter(F.col("CustomerID").isNotNull()).select(
        "CustomerID"
    ).distinct().count()

    print("\n=== Key business metrics ===")
    print(f"Total sales (revenue):        {total_sales:,.2f}")
    print(f"Average transaction value:    {avg_txn_value:,.2f}")
    print(f"Number of transactions:       {num_transactions:,}")
    print(f"Total quantity sold:          {int(total_quantity):,}")
    print(f"Unique products:              {unique_products:,}")
    print(f"Unique customers:             {unique_customers:,}")

    # Metric 3: Top 5 products by quantity sold -- DataFrame API
    top_products = (
        sales.groupBy("StockCode", "Description")
        .agg(
            F.sum("Quantity").alias("total_quantity"),
            F.round(F.sum("Revenue"), 2).alias("total_revenue"),
        )
        .orderBy(F.desc("total_quantity"))
        .limit(5)
    )
    print("\n=== Top 5 products by quantity sold ===")
    top_products.show(truncate=False)

    # Bonus: daily sales (product/business performance over time) -- Spark SQL
    daily_sales = spark.sql(
        """
        SELECT
            CAST(InvoiceDate AS DATE)          AS sale_date,
            ROUND(SUM(Revenue), 2)             AS daily_revenue,
            COUNT(DISTINCT InvoiceNo)          AS transactions,
            SUM(Quantity)                      AS units_sold
        FROM sales
        GROUP BY CAST(InvoiceDate AS DATE)
        ORDER BY sale_date
        """
    )
    print("\n=== Daily sales (first 10 days) ===")
    daily_sales.show(10, truncate=False)

    # Bonus: sales by country
    country_sales = (
        sales.groupBy("Country")
        .agg(F.round(F.sum("Revenue"), 2).alias("total_revenue"))
        .orderBy(F.desc("total_revenue"))
    )

    # ----- Step 4: Save metrics to CSV ----------------------------------- #
    # Primary deliverable: a single tidy metrics.csv (section, name, value)
    # that contains all three required metrics.
    summary_rows = [
        ("summary", "total_sales", float(total_sales)),
        ("summary", "average_transaction_value", float(avg_txn_value)),
        ("summary", "number_of_transactions", float(num_transactions)),
        ("summary", "total_quantity_sold", float(total_quantity)),
        ("summary", "unique_products", float(unique_products)),
        ("summary", "unique_customers", float(unique_customers)),
        ("summary", "total_line_items", float(total_rows)),
    ]
    for row in top_products.collect():
        label = f"{row['Description']} ({row['StockCode']})"
        summary_rows.append(("top_product_by_quantity", label, float(row["total_quantity"])))

    metrics_pdf = pd.DataFrame(summary_rows, columns=["section", "name", "value"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    metrics_pdf.to_csv(metrics_path, index=False)
    print(f"\n[output] Wrote {metrics_path}")

    # Supporting detail files (nice for the notebook / portfolio).
    write_single_csv(top_products, "top_products.csv")
    write_single_csv(daily_sales, "daily_sales.csv")
    write_single_csv(country_sales, "sales_by_country.csv")

    print("\nDone. All metrics written to the output/ directory.")
    spark.stop()


if __name__ == "__main__":
    main()
