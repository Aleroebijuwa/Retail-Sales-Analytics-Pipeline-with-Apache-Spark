# Retail Sales Analytics Pipeline with Apache Spark

An end-to-end learning project that ingests a real retail transaction dataset
into **Apache Spark (PySpark)**, explores its structure, and computes core
business metrics using both the **DataFrame API** and **Spark SQL**.

This repository covers:
- **Step 1 – Data Ingestion and Exploration** (`data_ingestion.py`)
- **Step 2 – Data Cleaning and Feature Engineering** (`data_cleaning.py`, `feature_engineering.py`)
- **Step 3 – Model Training and Evaluation** (`train_models.py`)
- **Step 4 – Model Evaluation, Fairness & Explainability** (`model_evaluation.py`)
- **Step 5 – Performance Optimization** (`optimized_pipeline.py`, `optimization_report.md`)
- **Step 6 – Packaging, Tests & Docs** (`src/retail_pipeline/`, `setup.py`, `tests/`)

---

## 📦 Installable package: `retail_pipeline`

The reusable logic is packaged as an installable library under
[`src/retail_pipeline/`](src/retail_pipeline/) (a `src`-layout package).

### Package structure
```
src/retail_pipeline/
├── __init__.py          # public API + version
├── data_loader.py       # data processing: get_spark_session, load_sales_data, RETAIL_SCHEMA
├── transformations.py   # data processing: add_revenue, remove_invalid_sales, clean_data, cap_outliers_iqr
├── analysis.py          # analysis: aggregate_sales, top_products, sales_by_country, daily_sales
└── utils.py             # utilities: configure_spark_home
tests/                   # pytest suite (6 tests)
setup.py                 # packaging metadata + dependencies
```

### Installation
```bash
pip install -r requirements.txt
pip install -e .          # install the retail_pipeline package (editable)
```

### Usage example
```python
from retail_pipeline import (
    get_spark_session, load_sales_data, clean_data, add_revenue,
    aggregate_sales, top_products,
)

spark = get_spark_session("MyRetailApp")
df = load_sales_data(spark, "data/online_retail.csv")   # -> Spark DataFrame
clean = add_revenue(clean_data(df))                     # clean + add Revenue

aggregate_sales(clean).show()      # total_sales, total_quantity, num_transactions
top_products(clean, n=5).show()    # top 5 products by quantity
```

### Key functions
| Function | Purpose |
|----------|---------|
| `get_spark_session(app_name, master)` | Create a configured SparkSession (fixes `SPARK_HOME`). |
| `load_sales_data(spark, path, infer_schema=False)` | Load the retail CSV into a DataFrame with an explicit schema. |
| `clean_data(df, drop_missing_customer=True)` | Drop blank descriptions, missing customers and invalid sales. |
| `add_revenue(df)` | Add `Revenue = Quantity × UnitPrice`. |
| `aggregate_sales(df)` | Total sales, total quantity, number of transactions. |
| `top_products(df, n=5)` | Top *n* products by quantity sold. |

### Running the tests
```bash
pip install pytest
pytest -q          # 6 tests: loading, transformations, aggregations
```

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository — *Online Retail* Data Set](https://archive.ics.uci.edu/dataset/352/online+retail)

The dataset contains all transactions occurring between **01/12/2010 and
09/12/2011** for a UK-based, registered, non-store online retailer that sells
unique all-occasion gifts. It has **541,909 rows** across **8 columns**.

> The raw data file is **not** committed to the repo (it is large and is
> `.gitignore`d). The script downloads it automatically on first run into
> `data/`.

### Column descriptions

| Column        | Type       | Description                                                                                 |
|---------------|------------|---------------------------------------------------------------------------------------------|
| `InvoiceNo`   | string     | 6-digit invoice number. A leading **`C`** marks a **cancellation** (return).                |
| `StockCode`   | string     | 5-digit product (item) code.                                                                |
| `Description` | string     | Product name.                                                                               |
| `Quantity`    | integer    | Units per transaction line. Can be **negative** for returns.                                |
| `InvoiceDate` | timestamp  | Date and time the transaction was generated.                                                |
| `UnitPrice`   | double     | Price per unit, in **GBP (£)**.                                                              |
| `CustomerID`  | double     | 5-digit customer number (many rows are null — guest / unregistered).                        |
| `Country`     | string     | Country where the customer resides.                                                         |

A derived column **`Revenue = Quantity × UnitPrice`** is added during processing.

---

## 🧮 Metrics calculated

"Valid sales" exclude cancellations (`InvoiceNo` starting with `C`) and any rows
with a non-positive quantity or price. On that filtered set the pipeline computes:

1. **Total sales (revenue)** — `SUM(Quantity × UnitPrice)`
2. **Average transaction value** — average total revenue per invoice (order)
3. **Top 5 products by quantity sold**

Plus supporting breakdowns: **daily sales** and **sales by country**.

<!-- METRICS:START -->
**Results** (valid sales only, computed by `data_ingestion.py`):

| Metric                       | Value           |
|------------------------------|-----------------|
| Total sales (revenue)        | **£10,666,684.54** |
| Average transaction value    | **£534.40**     |
| Number of transactions       | 19,960          |
| Total quantity sold          | 5,588,376       |
| Unique products              | 3,922           |
| Unique customers             | 4,338           |
| Total line items (raw rows)  | 541,909         |

**Top 5 products by quantity sold**

| Rank | Product (StockCode)                          | Quantity | Revenue     |
|------|----------------------------------------------|----------|-------------|
| 1    | PAPER CRAFT , LITTLE BIRDIE (23843)          | 80,995   | £168,469.60 |
| 2    | MEDIUM CERAMIC TOP STORAGE JAR (23166)       | 78,033   | £81,700.92  |
| 3    | WORLD WAR 2 GLIDERS ASSTD DESIGNS (84077)    | 55,047   | £13,841.85  |
| 4    | JUMBO BAG RED RETROSPOT (85099B)             | 48,474   | £94,340.05  |
| 5    | WHITE HANGING HEART T-LIGHT HOLDER (85123A)  | 37,599   | £104,340.29 |
<!-- METRICS:END -->

---

## 📁 Outputs

All results are written to the `output/` directory:

| File                     | Contents                                                        |
|--------------------------|-----------------------------------------------------------------|
| `metrics.csv`            | **Primary deliverable** — all key metrics as `section,name,value`. |
| `top_products.csv`       | Top 5 products by quantity, with revenue.                        |
| `daily_sales.csv`        | Revenue, transactions and units sold per day.                   |
| `sales_by_country.csv`   | Total revenue per country.                                       |

---

## 🧹 Step 2 — Data Cleaning & Feature Engineering

Two scripts turn the raw transactions into an analysis-ready feature set.

### `data_cleaning.py`
- **Missing values** (documented strategy per column): drop null `CustomerID`
  (an identity key — imputation would fabricate customers), drop blank
  `Description` (non-product adjustment rows), median-impute any null
  `UnitPrice` (robust to price skew).
- **Invalid records**: remove cancellations (`InvoiceNo` starting with `C`) and
  non-positive quantity/price.
- **Outliers**: detected with the **IQR method** and **capped (winsorized)** to
  `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` on `Quantity` and `UnitPrice` — keeps every
  transaction while limiting extreme influence.

### `feature_engineering.py`
- **Temporal features (7)**: `day_of_week`, `day_name`, `month`, `quarter`,
  `year`, `hour`, `is_weekend`, `is_holiday` (UK bank holidays, via a Spark UDF).
- **Customer behaviour aggregations**: `total_purchases`, `total_quantity`,
  `total_revenue`, `unique_products`, `days_since_last_purchase` (recency),
  and `average_transaction_value` (avg revenue per invoice) — an RFM-style table.
- **Scaling**: all numeric customer features are standardised with a
  scikit-learn `StandardScaler` **Pipeline**, saved to
  `models/customer_feature_scaler.joblib` for reuse.

**Step 2 outputs:** `output/customer_features.csv`,
`output/customer_features_scaled.csv`, and the saved scaler in `models/`.

Run:
```bash
python data_cleaning.py        # clean only (writes data/cleaned_retail.csv)
python feature_engineering.py  # clean + features + scaling (writes output/ + models/)
```

---

## 🤖 Step 3 — Model Training & Evaluation (`train_models.py`)

Trains two Spark **MLlib** models on the Step 2 features:

### 1. Linear Regression — sales forecasting
Predicts **daily total revenue** from calendar features (`day_of_week`, `month`,
`quarter`, `is_weekend`, `is_holiday`). Evaluated with `RegressionEvaluator`:

| Metric | Value |
|--------|-------|
| RMSE   | ~6,258 |
| R²     | ~0.41  |

### 2. Random Forest — product demand prediction
Predicts **units sold (Quantity)** per transaction from price + calendar features.
Tuned with **`TrainValidationSplit`** over a **4-combination** parameter grid
(`maxDepth ∈ {5,10}` × `numTrees ∈ {20,50}`). Best model: `maxDepth=10, numTrees=50`.

**Feature importances** (extracted from the best model): `UnitPrice` dominates
(~0.71), followed by `hour` (~0.14) — i.e. price is by far the strongest driver
of how many units sell.

### Outputs
- `output/predictions.parquet` — actual vs predicted rows for both models
  (columns: `model`, `actual`, `predicted`).

Runs end-to-end in **under 2 minutes** (the demand model trains on a 10% sample).

Run:
```bash
python train_models.py
```

---

## ⚖️ Step 4 — Model Evaluation, Fairness & Explainability (`model_evaluation.py`)

AUC-ROC / precision / recall / F1 are **classification** metrics, so this step
defines a binary target: **customer churn** (`label = 1` if no purchase in the
last 90 days). A Spark MLlib `RandomForestClassifier` predicts churn from
customer behaviour features.

### Classification metrics
| Metric | Overall |
|--------|---------|
| AUC-ROC | 0.859 |
| Precision | 0.772 |
| Recall | 0.777 |
| F1 | 0.773 |

### Bias detection & fairness
Customers are split into two segments — **United Kingdom** vs **International**
(country is the only demographic-like attribute in the data) — and the metrics
are recomputed per segment:

| Segment | n | AUC-ROC | Precision | Recall | F1 |
|---------|---|---------|-----------|--------|----|
| United Kingdom | 772 | 0.858 | 0.774 | 0.779 | 0.774 |
| International   |  89 | 0.865 | 0.762 | 0.764 | 0.762 |

**Max disparity ≈ 0.015** across all metrics → no fairness concern; the model
performs equitably across segments.

### Explainability (SHAP)
SHAP explanations are generated for **5 individual predictions**, attributing
each churn probability to the input features (a scikit-learn surrogate mirrors
the Spark model, since SHAP has no native Spark-ML support).

### Outputs
- `evaluation_results.json` — full metrics, per-segment metrics, fairness
  disparities, and the 5 SHAP explanations.
- `output/fairness_report.csv` — flat per-segment metric table.

Run:
```bash
python model_evaluation.py
```

---

## ⚡ Step 5 — Performance Optimization (`optimized_pipeline.py`)

Runs the same analytics workload **twice** (baseline vs optimized), measures the
wall-clock time, and captures the Spark UI Stages tab for each.

| Run | Configuration | Time |
|-----|---------------|------|
| Before | 200 shuffle partitions, no cache, AQE off | **45.37 s** |
| After  | 8 partitions, cached DataFrames, AQE on | **10.79 s** |

**➡ 76.2% faster** (well above the 30% target).

**Optimizations:** repartition by `Country` into 8 partitions; cache the two
reused DataFrames (base + daily sales); tune 3 config parameters
(`shuffle.partitions` 200→8, `adaptive.enabled` on, `adaptive.coalescePartitions`
on). Full write-up in [`optimization_report.md`](optimization_report.md).

**Spark UI evidence** (`screenshots/`): the *before* capture shows 15 stages at
200 tasks each; the *after* capture shows **34 skipped stages** (cache hits) and
8 tasks per stage.

Run:
```bash
python optimized_pipeline.py
```

---

## 🚀 How to run

### Prerequisites
- **Python 3.11**
- **Java 11** (required by Spark 3.5). Set `JAVA_HOME` to your JDK, e.g.
  `C:\Program Files\Microsoft\jdk-11.0.31.11-hotspot`.

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the script
```bash
python data_ingestion.py
```

On the first run the script downloads the dataset (~23 MB), converts the Excel
workbook to CSV in `data/`, then loads it into Spark and writes the metrics to
`output/`.

### Run the notebook (optional)
The same workflow is available interactively in
[`notebooks/data_ingestion.ipynb`](notebooks/data_ingestion.ipynb).

---

## 🗂️ Project structure

```
.
├── data/                 # raw dataset (downloaded, git-ignored)
├── models/               # (future) trained models
├── notebooks/            # Jupyter notebooks
├── output/               # computed metrics (CSV) — tracked in git
├── src/                  # (future) reusable pipeline modules
├── data_ingestion.py     # Step 1 script: load → explore → metrics → CSV
├── SETUP_GUIDE.md        # Spark / Java environment setup guide
└── README.md
```

---

## 🛠️ Tech stack

- Apache Spark 3.5.3 (PySpark)
- Python 3.11
- pandas + openpyxl (dataset download / Excel→CSV conversion)
