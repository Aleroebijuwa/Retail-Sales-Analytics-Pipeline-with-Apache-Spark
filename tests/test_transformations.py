"""Unit tests for retail_pipeline.transformations."""

from pyspark.sql import functions as F

from retail_pipeline import add_revenue, remove_invalid_sales, clean_data


def test_add_revenue_computes_quantity_times_price(sample_transactions):
    """Revenue must equal Quantity * UnitPrice for each row."""
    result = add_revenue(sample_transactions)
    row = result.filter(F.col("InvoiceNo") == "536365").collect()[0]
    assert row["Revenue"] == round(6 * 2.55, 2)  # 15.30


def test_remove_invalid_sales_drops_cancellations_and_negatives(sample_transactions):
    """Cancellations (C-invoices) and non-positive quantities are removed."""
    result = remove_invalid_sales(sample_transactions)
    invoices = {r["InvoiceNo"] for r in result.select("InvoiceNo").collect()}
    assert "C536367" not in invoices           # cancellation removed
    assert "536368" not in invoices            # negative quantity removed
    assert {"536365", "536366", "536369"} <= invoices  # valid sales kept


def test_clean_data_removes_missing_customer(sample_transactions):
    """clean_data drops rows without a CustomerID (and invalid rows)."""
    result = clean_data(sample_transactions)
    invoices = {r["InvoiceNo"] for r in result.select("InvoiceNo").collect()}
    assert "536369" not in invoices            # null CustomerID removed
    assert result.count() == 2                 # only the two valid, complete sales
