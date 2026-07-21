"""Unit tests for retail_pipeline.analysis and data loading."""

from retail_pipeline import (
    add_revenue,
    clean_data,
    aggregate_sales,
    top_products,
    RETAIL_SCHEMA,
)


def test_aggregate_sales_totals(sample_transactions):
    """Total sales must equal the sum of Revenue over the valid, clean rows."""
    clean = add_revenue(clean_data(sample_transactions))
    metrics = aggregate_sales(clean).collect()[0]
    # Valid clean rows: 536365 (6*2.55=15.30) and 536366 (4*3.39=13.56)
    assert metrics["total_sales"] == round(15.30 + 13.56, 2)  # 28.86
    assert metrics["num_transactions"] == 2
    assert metrics["total_quantity"] == 10


def test_top_products_orders_by_quantity(sample_transactions):
    """top_products returns products ranked by total quantity descending."""
    clean = add_revenue(clean_data(sample_transactions))
    top = top_products(clean, n=1).collect()
    assert len(top) == 1
    # 536365 has quantity 6 (the highest among the two valid rows).
    assert top[0]["StockCode"] == "85123A"
    assert top[0]["total_quantity"] == 6


def test_retail_schema_has_expected_columns():
    """The published schema exposes the 8 expected retail columns."""
    names = [f.name for f in RETAIL_SCHEMA.fields]
    assert names == [
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country",
    ]
