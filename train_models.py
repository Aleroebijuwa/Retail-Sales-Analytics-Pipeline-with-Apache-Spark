"""
Model Training & Evaluation - Retail Sales Analytics Pipeline (Step 3)
=====================================================================

Trains and evaluates two Spark MLlib models on the cleaned, feature-engineered
retail data from Step 2:

  1. Linear Regression  -> SALES FORECASTING
     Predicts daily total revenue from calendar features. Reports RMSE and R².

  2. Random Forest      -> PRODUCT DEMAND PREDICTION
     Predicts units sold (Quantity) per transaction. Tuned with
     TrainValidationSplit over a parameter grid, with feature importances
     extracted from the best model.

Predictions (actual vs predicted) are saved to a Parquet file.

Runs in well under 5 minutes by training the demand model on a sample.

Run:
    python train_models.py
"""

import os

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

from data_ingestion import build_spark, OUTPUT_DIR
from data_cleaning import clean_data
from feature_engineering import create_temporal_features

SEED = 42
PRED_PARQUET = os.path.join(OUTPUT_DIR, "predictions.parquet")


# --------------------------------------------------------------------------- #
# Shared feature frame
# --------------------------------------------------------------------------- #
def build_feature_frame(spark):
    """Cleaned transactions + temporal features (cached for reuse)."""
    df = create_temporal_features(clean_data(spark))
    return df.cache()


# --------------------------------------------------------------------------- #
# Model 1: Linear Regression -> daily sales forecasting
# --------------------------------------------------------------------------- #
def train_linear_regression(df):
    """Forecast daily total revenue from calendar features."""
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION  ->  daily sales forecasting")
    print("=" * 60)

    # Aggregate transactions to one row per day. The calendar features are
    # functionally determined by the date, so grouping by them is safe.
    daily = df.groupBy(
        F.to_date("InvoiceDate").alias("sale_date"),
        "day_of_week", "month", "quarter", "is_weekend", "is_holiday",
    ).agg(F.sum("Revenue").alias("daily_revenue"))

    feature_cols = ["day_of_week", "month", "quarter", "is_weekend", "is_holiday"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(daily)

    train, test = data.randomSplit([0.8, 0.2], seed=SEED)

    lr = LinearRegression(featuresCol="features", labelCol="daily_revenue")
    model = lr.fit(train)
    predictions = model.transform(test)

    rmse = RegressionEvaluator(
        labelCol="daily_revenue", predictionCol="prediction", metricName="rmse"
    ).evaluate(predictions)
    r2 = RegressionEvaluator(
        labelCol="daily_revenue", predictionCol="prediction", metricName="r2"
    ).evaluate(predictions)

    print(f"Training days: {train.count()},  Test days: {test.count()}")
    print(f"Linear Regression RMSE: {rmse:,.2f}")
    print(f"Linear Regression R²  : {r2:.4f}")

    return predictions.select(
        F.lit("linear_regression").alias("model"),
        F.col("daily_revenue").alias("actual"),
        F.col("prediction").alias("predicted"),
    )


# --------------------------------------------------------------------------- #
# Model 2: Random Forest -> product demand prediction (with tuning)
# --------------------------------------------------------------------------- #
def train_random_forest(df):
    """Predict units sold (Quantity) per transaction; tune with TVS."""
    print("\n" + "=" * 60)
    print("RANDOM FOREST  ->  product demand prediction")
    print("=" * 60)

    # Train on a sample so the whole script finishes well under 5 minutes.
    sample = df.sample(fraction=0.1, seed=SEED)

    feature_cols = [
        "UnitPrice", "day_of_week", "month", "quarter",
        "is_weekend", "is_holiday", "hour",
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(sample).select("features", "Quantity")

    train, test = data.randomSplit([0.8, 0.2], seed=SEED)

    rf = RandomForestRegressor(
        featuresCol="features", labelCol="Quantity", seed=SEED
    )

    # ParamGrid: 2 x 2 = 4 parameter combinations (>= 3 required).
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.maxDepth, [5, 10])
        .addGrid(rf.numTrees, [20, 50])
        .build()
    )
    print(f"TrainValidationSplit testing {len(param_grid)} parameter combinations")

    evaluator = RegressionEvaluator(
        labelCol="Quantity", predictionCol="prediction", metricName="rmse"
    )
    tvs = TrainValidationSplit(
        estimator=rf,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        trainRatio=0.8,
        seed=SEED,
    )

    tvs_model = tvs.fit(train)
    best = tvs_model.bestModel
    predictions = best.transform(test)

    rmse = evaluator.evaluate(predictions)
    r2 = RegressionEvaluator(
        labelCol="Quantity", predictionCol="prediction", metricName="r2"
    ).evaluate(predictions)

    print(f"Best model -> maxDepth={best.getMaxDepth()}, numTrees={best.getNumTrees}")
    print(f"Random Forest RMSE: {rmse:.4f}")
    print(f"Random Forest R²  : {r2:.4f}")

    # Feature importances from the best Random Forest model.
    importances = best.featureImportances.toArray()
    print("\nFeature importances (demand prediction):")
    for name, imp in sorted(
        zip(feature_cols, importances), key=lambda x: x[1], reverse=True
    ):
        print(f"  {name:<15} {imp:.4f}")

    return predictions.select(
        F.lit("random_forest").alias("model"),
        F.col("Quantity").cast("double").alias("actual"),
        F.col("prediction").alias("predicted"),
    )


# --------------------------------------------------------------------------- #
# Save predictions to Parquet
# --------------------------------------------------------------------------- #
def save_predictions_parquet(pred_df, path):
    """Write actual-vs-predicted rows to a Parquet file.

    Uses pandas/pyarrow so it works on Windows without a Hadoop/winutils
    installation (Spark's native writer requires it).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pdf = pred_df.toPandas()
    pdf.to_parquet(path, index=False)
    print(f"\n[output] Saved {len(pdf):,} predictions -> {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    df = build_feature_frame(spark)

    lr_preds = train_linear_regression(df)
    rf_preds = train_random_forest(df)

    # Combine both models' actual-vs-predicted rows and persist to Parquet.
    all_preds = lr_preds.unionByName(rf_preds)
    save_predictions_parquet(all_preds, PRED_PARQUET)

    print("\nModel training & evaluation complete.")
    spark.stop()


if __name__ == "__main__":
    main()
