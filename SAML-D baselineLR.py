from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number, when
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = (
    SparkSession.builder
    .appName("SAML-D Baseline LR (原始欄位)")
    .config("spark.driver.memory", "6g")
    .config("spark.executor.memory", "8g")
    .config("spark.executor.cores", "4")
    .config("spark.executor.instances", "4")
    .config("spark.sql.shuffle.partitions", "128")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setCheckpointDir("gs://saml-d/spark-checkpoint")

# 1. 載入資料
df = spark.read.option("header", True).option("inferSchema", True).csv("gs://saml-d/SAML-D.csv")

# 2. 時間排序加 row_id，切分 7:1.5:1.5
window = Window.orderBy("Date", "Time")
df = df.withColumn("row_id", row_number().over(window))
total = df.count()
train_cut = int(total * 0.7)
val_cut = int(total * 0.85)
train_df = df.filter(col("row_id") <= train_cut)
val_df = df.filter((col("row_id") > train_cut) & (col("row_id") <= val_cut))
test_df = df.filter(col("row_id") > val_cut)

# 3. 編碼所有類別型欄位
cat_cols = [
    "Sender_account", "Receiver_account", "Payment_currency", "Received_currency",
    "Sender_bank_location", "Receiver_bank_location", "Payment_type", "Laundering_type"
]
indexers = [StringIndexer(inputCol=colname, outputCol=f"{colname}_idx", handleInvalid="keep").fit(train_df) for colname in cat_cols]
for indexer in indexers:
    train_df = indexer.transform(train_df)
    val_df = indexer.transform(val_df)
    test_df = indexer.transform(test_df)

# 4. 建立特徵向量
feature_cols = [f"{c}_idx" for c in cat_cols] + ["Amount"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_df = assembler.transform(train_df)
val_df = assembler.transform(val_df)
test_df = assembler.transform(test_df)

# 5. 訓練 baseline LR
lr = LogisticRegression(
    featuresCol="features",
    labelCol="Is_laundering",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0,
    threshold=0.5
)
model = lr.fit(train_df)
val_pred = model.transform(val_df)
test_pred = model.transform(test_df)

# 6. 指標
def metrics(pred_df):
    tp = pred_df.filter((col("prediction") == 1) & (col("Is_laundering") == 1)).count()
    tn = pred_df.filter((col("prediction") == 0) & (col("Is_laundering") == 0)).count()
    fp = pred_df.filter((col("prediction") == 1) & (col("Is_laundering") == 0)).count()
    fn = pred_df.filter((col("prediction") == 0) & (col("Is_laundering") == 1)).count()
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return acc, prec, rec, f1, tp, fp, fn, tn

evaluator = BinaryClassificationEvaluator(labelCol="Is_laundering", rawPredictionCol="rawPrediction")
val_auc = evaluator.evaluate(val_pred)
test_auc = evaluator.evaluate(test_pred)
print(f"\n📊 驗證集 AUC: {val_auc:.4f}")
print(f"✅ 測試集 AUC: {test_auc:.4f}")

acc, prec, rec, f1, tp, fp, fn, tn = metrics(test_pred)
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

spark.stop()
