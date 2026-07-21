"""Analysis: business-metric aggregations over cleaned retail transactions.

All functions expect a DataFrame that already contains a ``Revenue`` column
(see :func:`retail_pipeline.transformations.add_revenue`).
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def aggregate_sales(df):
    """Compute headline sales metrics as a single-row DataFrame.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Cleaned transactions with a ``Revenue`` column.

    Returns
    -------
    pyspark.sql.DataFrame
        One row with ``total_sales``, ``total_quantity`` and
        ``num_transactions`` (distinct invoices).
    """
    return df.agg(
        F.round(F.sum("Revenue"), 2).alias("total_sales"),
        F.sum("Quantity").alias("total_quantity"),
        F.countDistinct("InvoiceNo").alias("num_transactions"),
    )


def top_products(df, n=5):
    """Return the top ``n`` products by total quantity sold.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Cleaned transactions with a ``Revenue`` column.
    n : int, optional
        Number of products to return. Defaults to ``5``.

    Returns
    -------
    pyspark.sql.DataFrame
        Columns ``StockCode``, ``Description``, ``total_quantity``,
        ``total_revenue`` ordered by quantity descending.
    """
    return (
        df.groupBy("StockCode", "Description")
        .agg(
            F.sum("Quantity").alias("total_quantity"),
            F.round(F.sum("Revenue"), 2).alias("total_revenue"),
        )
        .orderBy(F.desc("total_quantity"))
        .limit(n)
    )


def sales_by_country(df):
    """Aggregate total revenue per country.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Cleaned transactions with a ``Revenue`` column.

    Returns
    -------
    pyspark.sql.DataFrame
        Columns ``Country`` and ``total_revenue`` ordered descending.
    """
    return (
        df.groupBy("Country")
        .agg(F.round(F.sum("Revenue"), 2).alias("total_revenue"))
        .orderBy(F.desc("total_revenue"))
    )


def daily_sales(df):
    """Aggregate revenue, transactions and units sold per calendar day.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Cleaned transactions with a ``Revenue`` column and ``InvoiceDate``.

    Returns
    -------
    pyspark.sql.DataFrame
        Columns ``sale_date``, ``daily_revenue``, ``transactions``,
        ``units_sold`` ordered by date.
    """
    return (
        df.groupBy(F.to_date("InvoiceDate").alias("sale_date"))
        .agg(
            F.round(F.sum("Revenue"), 2).alias("daily_revenue"),
            F.countDistinct("InvoiceNo").alias("transactions"),
            F.sum("Quantity").alias("units_sold"),
        )
        .orderBy("sale_date")
    )
