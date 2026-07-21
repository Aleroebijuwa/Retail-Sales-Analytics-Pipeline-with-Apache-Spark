"""
Model Evaluation, Fairness & Explainability - Retail Pipeline (Step 4)
=====================================================================

Evaluates a customer-churn classifier and analyses it for bias/fairness across
customer segments, with SHAP explanations.

Task framing
------------
Steps 1-3 produced regression models, but AUC-ROC / precision / recall / F1 are
CLASSIFICATION metrics, so this step defines a natural binary target:

    label = 1  ->  customer has CHURNED  (no purchase in the last 90 days)
    label = 0  ->  customer is ACTIVE

Features (customer behaviour from Step 2, excluding the recency field that
defines the label to avoid leakage):
    total_purchases, total_revenue, average_transaction_value,
    unique_products, total_quantity

Fairness / bias
---------------
The only demographic-like attribute in the Online Retail data is the customer's
country, so we segment customers into two groups and compare model performance:

    "United Kingdom"  vs  "International"

What this script produces (Definition of Done)
----------------------------------------------
  1. AUC-ROC, precision, recall, F1 for the trained classifier (Spark MLlib).
  2. Bias detection: the same metrics computed for each of the 2 segments.
  3. A fairness report of the metric disparities between segments.
  4. SHAP explanations for 5 individual predictions.
  5. All results written to evaluation_results.json.

Run:
    python model_evaluation.py
"""

import os
import json

import numpy as np

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

from sklearn.ensemble import RandomForestClassifier as SkRandomForest
import shap

from data_ingestion import build_spark, OUTPUT_DIR
from data_cleaning import clean_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(BASE_DIR, "evaluation_results.json")

CHURN_DAYS = 90
SEED = 42
FEATURE_COLS = [
    "total_purchases",
    "total_revenue",
    "average_transaction_value",
    "unique_products",
    "total_quantity",
]


# --------------------------------------------------------------------------- #
# Build the labelled customer table (features + churn label + segment)
# --------------------------------------------------------------------------- #
def build_customer_table(spark):
    df = clean_data(spark)
    max_date = df.agg(F.max("InvoiceDate")).collect()[0][0]

    invoice_totals = df.groupBy("CustomerID", "InvoiceNo").agg(
        F.sum("Revenue").alias("invoice_revenue")
    )
    avg_txn = invoice_totals.groupBy("CustomerID").agg(
        F.round(F.avg("invoice_revenue"), 2).alias("average_transaction_value")
    )

    customer = (
        df.groupBy("CustomerID")
        .agg(
            F.countDistinct("InvoiceNo").alias("total_purchases"),
            F.sum("Quantity").alias("total_quantity"),
            F.round(F.sum("Revenue"), 2).alias("total_revenue"),
            F.countDistinct("StockCode").alias("unique_products"),
            F.max("InvoiceDate").alias("last_purchase_date"),
            F.first("Country").alias("country"),
        )
        .withColumn(
            "days_since_last_purchase",
            F.datediff(F.lit(max_date), F.col("last_purchase_date")),
        )
        .join(avg_txn, on="CustomerID", how="left")
        # Binary churn label.
        .withColumn(
            "label",
            F.when(F.col("days_since_last_purchase") > CHURN_DAYS, 1).otherwise(0),
        )
        # Customer segment (demographic proxy) for fairness analysis.
        .withColumn(
            "segment",
            F.when(F.col("country") == "United Kingdom", "United Kingdom")
            .otherwise("International"),
        )
        .drop("last_purchase_date")
    )
    return customer


# --------------------------------------------------------------------------- #
# Spark MLlib evaluation helpers
# --------------------------------------------------------------------------- #
def _compute_metrics(predictions):
    """AUC-ROC, precision, recall, F1 for a predictions DataFrame."""
    n = predictions.count()
    n_pos = predictions.filter(F.col("label") == 1).count()

    metrics = {"n": n, "n_churned": n_pos, "n_active": n - n_pos}

    # AUC-ROC needs both classes present.
    if 0 < n_pos < n:
        auc = BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        ).evaluate(predictions)
        metrics["auc_roc"] = round(auc, 4)
    else:
        metrics["auc_roc"] = None  # undefined when a segment has one class only

    multi = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction"
    )
    metrics["precision"] = round(
        multi.evaluate(predictions, {multi.metricName: "weightedPrecision"}), 4
    )
    metrics["recall"] = round(
        multi.evaluate(predictions, {multi.metricName: "weightedRecall"}), 4
    )
    metrics["f1"] = round(
        multi.evaluate(predictions, {multi.metricName: "f1"}), 4
    )
    return metrics


def train_and_evaluate(customer):
    """Train the Spark churn classifier and evaluate overall + per segment."""
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
    data = assembler.transform(customer).select(
        "features", "label", "segment", *FEATURE_COLS
    )
    train, test = data.randomSplit([0.8, 0.2], seed=SEED)

    clf = RandomForestClassifier(
        featuresCol="features", labelCol="label", numTrees=60, maxDepth=8, seed=SEED
    )
    model = clf.fit(train)
    predictions = model.transform(test).cache()

    print("\n=== Overall metrics (churn classifier) ===")
    overall = _compute_metrics(predictions)
    for k, v in overall.items():
        print(f"  {k}: {v}")

    # ----- Bias detection / fairness across segments --------------------- #
    print("\n=== Per-segment metrics (bias detection) ===")
    segments = [r["segment"] for r in predictions.select("segment").distinct().collect()]
    per_segment = {}
    for seg in sorted(segments):
        seg_pred = predictions.filter(F.col("segment") == seg)
        per_segment[seg] = _compute_metrics(seg_pred)
        print(f"  [{seg}] {per_segment[seg]}")

    return model, train, test, overall, per_segment


# --------------------------------------------------------------------------- #
# Fairness report: disparities between segments
# --------------------------------------------------------------------------- #
def build_fairness_report(per_segment):
    """Compute metric gaps between the two segments."""
    segs = list(per_segment.keys())
    report = {"segments_compared": segs, "disparities": {}}
    if len(segs) != 2:
        return report

    a, b = segs
    for metric in ["auc_roc", "precision", "recall", "f1"]:
        va, vb = per_segment[a].get(metric), per_segment[b].get(metric)
        if va is not None and vb is not None:
            report["disparities"][metric] = {
                a: va, b: vb, "difference": round(abs(va - vb), 4),
            }
    # Simple overall fairness flag: >0.10 gap on any metric is worth attention.
    gaps = [d["difference"] for d in report["disparities"].values()]
    report["max_disparity"] = round(max(gaps), 4) if gaps else None
    report["fairness_concern"] = bool(gaps and max(gaps) > 0.10)
    return report


# --------------------------------------------------------------------------- #
# SHAP explanations (sklearn surrogate on identical data)
# --------------------------------------------------------------------------- #
def shap_explanations(train, test, n_explain=5):
    """Explain individual predictions with SHAP.

    SHAP has no native Spark-ML support, so we fit an equivalent scikit-learn
    RandomForest on the SAME training rows and features, then use SHAP's fast
    TreeExplainer to attribute each prediction to the input features.
    """
    train_pdf = train.select("label", *FEATURE_COLS).toPandas()
    test_pdf = test.select("label", *FEATURE_COLS).toPandas()

    sk = SkRandomForest(n_estimators=60, max_depth=8, random_state=SEED)
    sk.fit(train_pdf[FEATURE_COLS], train_pdf["label"])

    explainer = shap.TreeExplainer(sk)
    sample = test_pdf[FEATURE_COLS].head(n_explain)
    sv = np.array(explainer.shap_values(sample))

    # Normalise SHAP output shape to (n_samples, n_features) for the churn class.
    if sv.ndim == 3:
        arr = sv[1] if sv.shape[0] == 2 else sv[:, :, 1]
    else:
        arr = sv

    explanations = []
    for i in range(len(sample)):
        contribs = {f: round(float(arr[i][j]), 4) for j, f in enumerate(FEATURE_COLS)}
        explanations.append({
            "prediction_index": int(i),
            "predicted_churn_probability": round(
                float(sk.predict_proba(sample.iloc[[i]])[0][1]), 4
            ),
            "actual_label": int(test_pdf["label"].iloc[i]),
            "feature_values": {f: float(sample.iloc[i][f]) for f in FEATURE_COLS},
            "shap_contributions": contribs,
        })

    print(f"\n=== SHAP explanations generated for {len(explanations)} predictions ===")
    for e in explanations:
        top = max(e["shap_contributions"].items(), key=lambda x: abs(x[1]))
        print(f"  #{e['prediction_index']} churn_prob={e['predicted_churn_probability']} "
              f"-> top driver: {top[0]} ({top[1]:+})")
    return explanations


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    customer = build_customer_table(spark)
    model, train, test, overall, per_segment = train_and_evaluate(customer)
    fairness = build_fairness_report(per_segment)
    explanations = shap_explanations(train, test)

    results = {
        "task": "customer_churn_classification",
        "churn_definition": f"no purchase in the last {CHURN_DAYS} days",
        "features": FEATURE_COLS,
        "model": "Spark MLlib RandomForestClassifier (numTrees=60, maxDepth=8)",
        "overall_metrics": overall,
        "segment_metrics": per_segment,
        "fairness_report": fairness,
        "shap_explanations": explanations,
    }

    with open(RESULTS_JSON, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[output] Wrote {RESULTS_JSON}")

    # Also drop a flat fairness CSV for quick viewing.
    fairness_csv = os.path.join(OUTPUT_DIR, "fairness_report.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(fairness_csv, "w") as fh:
        fh.write("segment,n,auc_roc,precision,recall,f1\n")
        for seg, m in per_segment.items():
            fh.write(f"{seg},{m['n']},{m['auc_roc']},{m['precision']},"
                     f"{m['recall']},{m['f1']}\n")
    print(f"[output] Wrote {fairness_csv}")

    print("\nModel evaluation & fairness analysis complete.")
    spark.stop()


if __name__ == "__main__":
    main()
