"""Shared pytest fixtures for the retail_pipeline test suite."""

import pytest

from retail_pipeline import get_spark_session


@pytest.fixture(scope="session")
def spark():
    """A single local Spark session shared across the test session."""
    session = get_spark_session(app_name="retail_pipeline_tests", master="local[1]")
    yield session
    session.stop()


@pytest.fixture
def sample_transactions(spark):
    """A tiny, hand-built transactions DataFrame for deterministic tests.

    Includes: two valid sales, one cancellation (InvoiceNo 'C...'), one
    negative-quantity return, and one row with a null CustomerID.
    """
    rows = [
        # InvoiceNo, StockCode, Description, Quantity, UnitPrice, CustomerID, Country
        ("536365", "85123A", "WHITE HANGING HEART", 6, 2.55, 17850.0, "United Kingdom"),
        ("536366", "71053",  "WHITE METAL LANTERN", 4, 3.39, 17850.0, "United Kingdom"),
        ("C536367", "84406B", "CANCELLED ITEM",      2, 2.75, 13047.0, "France"),
        ("536368", "22960",  "RETURN ITEM",         -3, 1.25, 13047.0, "France"),
        ("536369", "22961",  "NO CUSTOMER",          5, 1.00, None,    "France"),
    ]
    cols = [
        "InvoiceNo", "StockCode", "Description",
        "Quantity", "UnitPrice", "CustomerID", "Country",
    ]
    return spark.createDataFrame(rows, cols)
