"""Data processing: cleaning and transformation logic for retail transactions."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def add_revenue(df):
    """Add a ``Revenue`` column equal to ``Quantity * UnitPrice``.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Transactions containing ``Quantity`` and ``UnitPrice`` columns.

    Returns
    -------
    pyspark.sql.DataFrame
        The input with an additional ``Revenue`` column (rounded to 2 dp).
    """
    return df.withColumn(
        "Revenue", F.round(F.col("Quantity") * F.col("UnitPrice"), 2)
    )


def remove_invalid_sales(df):
    """Drop rows that are not valid completed sales.

    Removes cancellations (``InvoiceNo`` starting with ``"C"``) and rows with a
    non-positive ``Quantity`` or ``UnitPrice``.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Raw transactions.

    Returns
    -------
    pyspark.sql.DataFrame
        Only valid sales rows.
    """
    return df.filter(
        (~F.col("InvoiceNo").startswith("C"))
        & (F.col("Quantity") > 0)
        & (F.col("UnitPrice") > 0)
    )


def clean_data(df, drop_missing_customer=True):
    """Clean raw retail transactions.

    Applies, in order: drop rows with a null/blank ``Description``; optionally
    drop rows with a null ``CustomerID``; remove invalid sales
    (see :func:`remove_invalid_sales`).

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Raw transactions.
    drop_missing_customer : bool, optional
        If ``True`` (default), drop rows without a ``CustomerID``.

    Returns
    -------
    pyspark.sql.DataFrame
        The cleaned transactions.
    """
    df = df.filter(
        F.col("Description").isNotNull() & (F.trim(F.col("Description")) != "")
    )
    if drop_missing_customer:
        df = df.filter(F.col("CustomerID").isNotNull())
    return remove_invalid_sales(df)


def cap_outliers_iqr(df, columns, k=1.5):
    """Cap (winsorize) outliers in numeric columns using the IQR rule.

    For each column, values outside ``[Q1 - k*IQR, Q3 + k*IQR]`` are clamped to
    the nearest bound instead of being dropped.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input data.
    columns : list of str
        Numeric column names to cap.
    k : float, optional
        IQR multiplier defining the whisker length. Defaults to ``1.5``.

    Returns
    -------
    pyspark.sql.DataFrame
        Data with the given columns winsorized.
    """
    for col in columns:
        q1, q3 = df.approxQuantile(col, [0.25, 0.75], 0.01)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        df = df.withColumn(
            col,
            F.when(F.col(col) < lower, F.lit(lower))
            .when(F.col(col) > upper, F.lit(upper))
            .otherwise(F.col(col)),
        )
    return df
