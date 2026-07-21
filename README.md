# Retail Sales Analytics Pipeline with Apache Spark

An end-to-end learning project that ingests a real retail transaction dataset
into **Apache Spark (PySpark)**, explores its structure, and computes core
business metrics using both the **DataFrame API** and **Spark SQL**.

This repository covers **Step 1 – Data Ingestion and Exploration**.

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

## 🚀 How to run

### Prerequisites
- **Python 3.11**
- **Java 11** (required by Spark 3.5). Set `JAVA_HOME` to your JDK, e.g.
  `C:\Program Files\Microsoft\jdk-11.0.31.11-hotspot`.

### Install dependencies
```bash
pip install pyspark==3.5.3 pandas openpyxl
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
