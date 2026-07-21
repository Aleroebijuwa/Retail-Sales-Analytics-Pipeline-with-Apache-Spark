"""Data processing: Spark session creation and loading retail sales data."""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    TimestampType,
)

from .utils import configure_spark_home

#: Explicit schema for the UCI Online Retail dataset.
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


def get_spark_session(app_name="RetailPipeline", master="local[*]"):
    """Create (or reuse) a configured :class:`~pyspark.sql.SparkSession`.

    Parameters
    ----------
    app_name : str, optional
        Name shown in the Spark UI. Defaults to ``"RetailPipeline"``.
    master : str, optional
        Spark master URL. Defaults to ``"local[*]"`` (all local cores).

    Returns
    -------
    pyspark.sql.SparkSession
        A ready-to-use Spark session.
    """
    configure_spark_home()
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def load_sales_data(spark, path, infer_schema=False):
    """Load a retail sales CSV file into a Spark DataFrame.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        The active Spark session.
    path : str
        Path to the CSV file (header row expected).
    infer_schema : bool, optional
        If ``True``, let Spark infer column types; otherwise apply the explicit
        :data:`RETAIL_SCHEMA` (recommended). Defaults to ``False``.

    Returns
    -------
    pyspark.sql.DataFrame
        The loaded transactions, one row per invoice line.
    """
    reader = (
        spark.read.option("header", True)
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
    )
    if infer_schema:
        reader = reader.option("inferSchema", True)
    else:
        reader = reader.schema(RETAIL_SCHEMA)
    return reader.csv(path)
