"""retail_pipeline - a reusable Spark analytics library for retail sales data.

Public API
----------
Data processing:
    * :func:`get_spark_session`, :func:`load_sales_data`, :data:`RETAIL_SCHEMA`
    * :func:`add_revenue`, :func:`remove_invalid_sales`, :func:`clean_data`,
      :func:`cap_outliers_iqr`
Analysis:
    * :func:`aggregate_sales`, :func:`top_products`, :func:`sales_by_country`,
      :func:`daily_sales`
"""

from .data_loader import get_spark_session, load_sales_data, RETAIL_SCHEMA
from .transformations import (
    add_revenue,
    remove_invalid_sales,
    clean_data,
    cap_outliers_iqr,
)
from .analysis import (
    aggregate_sales,
    top_products,
    sales_by_country,
    daily_sales,
)

__version__ = "0.1.0"

__all__ = [
    "get_spark_session",
    "load_sales_data",
    "RETAIL_SCHEMA",
    "add_revenue",
    "remove_invalid_sales",
    "clean_data",
    "cap_outliers_iqr",
    "aggregate_sales",
    "top_products",
    "sales_by_country",
    "daily_sales",
    "__version__",
]
