import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import (
    col, unix_timestamp, concat_ws, lag, sha2, substring, conv,
    min as spark_min, max as spark_max, avg, stddev, count, broadcast, when, monotonically_increasing_id
)
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from graphframes import GraphFrame
from sklearn.model_selection import StratifiedKFold

# 強制設定 Java/Hadoop 路徑
os.environ["JAVA_HOME"] = r"C:\Program Files\Gephi-0.10.1\jre-x64\jdk-11.0.17+8-jre"
os.environ["HADOOP_HOME"] = r"C:\Winutils"
os.environ["PATH"] = os.environ["JAVA_HOME"] + r";" + os.environ["HADOOP_HOME"] + r"\bin;" + os.environ["PATH"]

# =============================================================
# 1/10 ▶ 啟動 SparkSession
# =============================================================
print("1/10 ▶ 啟動 SparkSession…")
spark = (
    SparkSession.builder
    .appName("Aligned Stratified 10-Fold: Baseline vs TD-UF")
    .master("local[*]")                           # 自動用滿核心
    .config("spark.driver.memory", "8g")         # 主控端記憶體
    .config("spark.executor.memory", "8g")       # 執行端記憶體
    .config("spark.driver.maxResultSize", "2g")   # collect()最大值
    .config("spark.sql.shuffle.partitions", "256") # shuffle分區數，根據你機器可再小
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setCheckpointDir("file:///C:/Users/Leon/Desktop/spark-checkpoint")
print("✔️ SparkSession 建立完成\n")

# =============================================================
# 2/10 ▶ 讀取原始 SAML-D.csv 並處理欄位
# =============================================================
print("2/10 ▶ 讀取原始 SAML-D.csv 並處理欄位…")
saml_path = r"C:\Users\Leon\Desktop\程式語言資料\python\TD-UF\Anti Money Laundering Transaction Data (SAML-D)\SAML-D.csv"
df_saml = (
    spark.read.option("header", True).option("inferSchema", True)
    .csv(saml_path)
    .withColumn("tx_id", monotonically_increasing_id())
)
df_saml = (
    df_saml
    .withColumn("timestamp", col("Time").cast("long"))
    .withColumn("label", col("Is_laundering").cast("integer"))
)
print("SAML-D Schema:")
df_saml.printSchema()
print("SAML-D Example:")
df_saml.show(1, vertical=True)
print(f"   ✔️ SAML-D 共 {df_saml.count()} 筆\n")

# =============================================================
# 3/10 ▶ 讀取 PoH_Hashed_Transactions.csv 並對齊 label
# =============================================================
print("3/10 ▶ 讀取 PoH_Hashed_Transactions.csv 並對齊 label…")
hash_path = r"C:\Users\Leon\Desktop\程式語言資料\python\TD-UF\PoH_Hashed_Transactions\PoH_Hashed_Transactions.csv"
df_hash = (
    spark.read.option("header", True).option("inferSchema", True)
    .csv(hash_path)
    .withColumn("tx_id", monotonically_increasing_id())
)
df_hash = df_hash.join(broadcast(df_saml.select("tx_id", "label")), on="tx_id", how="left").na.fill({"label": 0})
print("PoH_Hashed Schema:")
df_hash.printSchema()
print("PoH_Hashed Example:")
df_hash.show(1, vertical=True)
print(f"   ✔️ PoH_Hashed 共 {df_hash.count()} 筆\n")

# =============================================================
# 4/10 ▶ 清理、檢查 Null，StratifiedKFold 分配 fold
# =============================================================
print("4/10 ▶ 檢查欄位 Null 分布…")
df_saml.select([
    count(when(col("Amount").isNull(), 1)).alias("Amount_nulls"),
    count(when(col("timestamp").isNull(), 1)).alias("timestamp_nulls"),
    count(when(col("label").isNull(), 1)).alias("label_nulls"),
    count("*").alias("total")
]).show()
print("   ▶ 正在丟棄 null 資料…")
before_drop = df_saml.count()
df_saml = df_saml.na.drop(subset=["Amount", "timestamp", "label"])
after_drop = df_saml.count()
print(f"   ▶ 清理前 {before_drop} 筆，清理後 {after_drop} 筆。")
if after_drop == 0:
    raise ValueError("na.drop 後無資料，請檢查資料格式與前處理流程！")
df_hash = df_hash.na.drop(subset=["Amount", "timestamp", "label"])
print(f"   ✔️ SAML-D 清理後還有 {df_saml.count()} 筆資料, Hash 清理後還有 {df_hash.count()} 筆資料\n")

saml_pd = df_saml.select("tx_id", "label").toPandas()
if len(saml_pd) == 0:
    raise ValueError("你的資料筆數為 0，請檢查前面欄位是否都正確！")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
saml_pd["fold"] = -1
for fold, (_, test_idx) in enumerate(skf.split(saml_pd["tx_id"], saml_pd["label"])):
    saml_pd.loc[test_idx, "fold"] = fold
folds_df = spark.createDataFrame(saml_pd[["tx_id", "fold"]])
df_saml = df_saml.join(folds_df, on="tx_id")
df_hash = df_hash.join(folds_df, on="tx_id")
print("   ✔️ fold 分配完成（StratifiedKFold），共 10 折\n")

# =============================================================
# 5/10 ▶ Baseline 特徵向量化準備
# =============================================================
print("5/10 ▶ Baseline 特徵向量化準備…")
raw_features = ["Amount", "timestamp"]
assembler_raw = VectorAssembler(inputCols=raw_features, outputCol="features_raw")
print("   ✔️ Baseline VectorAssembler 設定完成\n")

# =============================================================
# 6/10 ▶ 定義 TD-UF 特徵計算
# =============================================================
print("6/10 ▶ 定義 TD-UF 特徵計算…")
tduf_feats = [
    "tx_count", "avg_amt", "std_amt", "max_amt", "min_amt",
    "avg_gap", "std_gap", "max_gap", "min_gap"
]
assembler_tduf = VectorAssembler(inputCols=tduf_feats, outputCol="features_tduf")
print("   ✔️ TD-UF VectorAssembler 設定完成\n")

# =============================================================
# 7/10 ▶ 進行 10 折 CV
# =============================================================
print("7/10 ▶ 進行 10 折 Stratified CV…")
evaluator = BinaryClassificationEvaluator(labelCol="label")
metrics = {k: [] for k in [
    'Acc_raw', 'Prec_raw', 'Rec_raw', 'F1_raw', 'AUC_raw', 'T_raw',
    'Acc_tduf', 'Prec_tduf', 'Rec_tduf', 'F1_tduf', 'AUC_tduf', 'T_tduf'
]}
for fold in tqdm(range(10), desc="fold"):
    train_s = df_saml.filter(col("fold") != fold)
    test_s  = df_saml.filter(col("fold") == fold)
    train_h = df_hash.filter(col("fold") != fold)
    test_h  = df_hash.filter(col("fold") == fold)
    print(f"Fold {fold}: train_s={train_s.count()}, test_s={test_s.count()}, train_h={train_h.count()}, test_h={test_h.count()}")

    # Baseline
    t0 = time.time()
    tr_r = assembler_raw.transform(train_s).select("features_raw", "label")
    te_r = assembler_raw.transform(test_s).select("features_raw", "label")
    if tr_r.count() == 0 or te_r.count() == 0:
        print(f"   ⚠️ 第 {fold} 折 baseline 沒有訓練或測試資料，跳過")
        continue
    lr_r = LogisticRegression(featuresCol="features_raw", labelCol="label")
    model_r = lr_r.fit(tr_r)
    pred_r  = model_r.transform(te_r)
    t1 = time.time()
    tp = pred_r.filter((col("label") == 1) & (col("prediction") == 1)).count()
    tn = pred_r.filter((col("label") == 0) & (col("prediction") == 0)).count()
    fp = pred_r.filter((col("label") == 0) & (col("prediction") == 1)).count()
    fn = pred_r.filter((col("label") == 1) & (col("prediction") == 0)).count()
    total = tp + tn + fp + fn
    acc  = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    auc  = evaluator.evaluate(pred_r, {evaluator.metricName: "areaUnderROC"})
    metrics['Acc_raw'].append(acc)
    metrics['Prec_raw'].append(prec)
    metrics['Rec_raw'].append(rec)
    metrics['F1_raw'].append(f1)
    metrics['AUC_raw'].append(auc)
    metrics['T_raw'].append(t1 - t0)

    # TD-UF
    t2 = time.time()
    v = train_h.select(col("tx_id").alias("id")).distinct()
    e = (
        train_h.alias("a").join(
            train_h.alias("b"),
            (col("a.Receiver_account") == col("b.Sender_account")) & (col("a.timestamp") < col("b.timestamp")),
            "inner"
        )
        .select(col("a.tx_id").alias("src"), col("b.tx_id").alias("dst")).distinct()
    )
    comp = GraphFrame(v, e).connectedComponents()
    dfc = comp.join(train_h.union(test_h), comp.id == train_h.union(test_h).tx_id, "inner").select(train_h.union(test_h)["*"], comp["component"])
    w = Window.partitionBy("component").orderBy("timestamp")
    tmp = (
        dfc
        .withColumn("prev_ts", lag("timestamp", 1).over(w))
        .withColumn("gap", col("timestamp") - col("prev_ts"))
    )
    agg = (
        tmp.groupBy("component").agg(
            count("*").alias("tx_count"),
            avg("Amount").alias("avg_amt"),
            stddev("Amount").alias("std_amt"),
            spark_max("Amount").alias("max_amt"),
            spark_min("Amount").alias("min_amt"),
            avg("gap").alias("avg_gap"),
            stddev("gap").alias("std_gap"),
            spark_max("gap").alias("max_gap"),
            spark_min("gap").alias("min_gap"),
            spark_max("label").alias("label")
        )
        .na.fill({
            "tx_count": 0,
            "avg_amt": 0.0, "std_amt": 0.0, "max_amt": 0.0, "min_amt": 0.0,
            "avg_gap": 0.0, "std_gap": 0.0, "max_gap": 0.0, "min_gap": 0.0
        })
    )
    feat_df = dfc.join(agg, ["component"], "left")
    tr_t = assembler_tduf.transform(feat_df.filter(col("fold") != fold)).select("features_tduf", "label")
    te_t = assembler_tduf.transform(feat_df.filter(col("fold") == fold)).select("features_tduf", "label")
    if tr_t.count() == 0 or te_t.count() == 0:
        print(f"   ⚠️ 第 {fold} 折 TD-UF 沒有訓練或測試資料，跳過")
        continue
    lr_t = LogisticRegression(featuresCol="features_tduf", labelCol="label")
    model_t = lr_t.fit(tr_t)
    pred_t  = model_t.transform(te_t)
    t3 = time.time()
    tp = pred_t.filter((col("label") == 1) & (col("prediction") == 1)).count()
    tn = pred_t.filter((col("label") == 0) & (col("prediction") == 0)).count()
    fp = pred_t.filter((col("label") == 0) & (col("prediction") == 1)).count()
    fn = pred_t.filter((col("label") == 1) & (col("prediction") == 0)).count()
    total = tp + tn + fp + fn
    acc  = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    auc  = evaluator.evaluate(pred_t, {evaluator.metricName: "areaUnderROC"})
    metrics['Acc_tduf'].append(acc)
    metrics['Prec_tduf'].append(prec)
    metrics['Rec_tduf'].append(rec)
    metrics['F1_tduf'].append(f1)
    metrics['AUC_tduf'].append(auc)
    metrics['T_tduf'].append(t3 - t2)

# =============================================================
# 8/10 ▶ 平均指標輸出＋存CSV
# =============================================================
print("8/10 ▶ 計算各指標平均…")
summary = []
for label in ["Acc", "Prec", "Rec", "F1", "AUC"]:
    baseline = sum(metrics[f"{label}_raw"]) / len(metrics[f"{label}_raw"]) if metrics[f"{label}_raw"] else 0
    tduf = sum(metrics[f"{label}_tduf"]) / len(metrics[f"{label}_tduf"]) if metrics[f"{label}_tduf"] else 0
    summary.append({"指標": label, "Baseline": baseline, "TD-UF": tduf})
    print(f"{label:12s} Baseline: {baseline:.4f}  TD-UF: {tduf:.4f}")

df_summary = pd.DataFrame(summary)
df_summary.to_csv("fold10_result_summary.csv", index=False, encoding="utf-8-sig")
print("已儲存平均結果至 fold10_result_summary.csv")

# =============================================================
# 9/10 ▶ 長條圖比較
# =============================================================
print("9/10 ▶ 長條圖比較…")
labels = ["Acc", "Prec", "Rec", "F1", "AUC"]
idx = range(len(labels))
vals_raw = [sum(metrics[f"{m}_raw"]) / len(metrics[f"{m}_raw"]) if metrics[f"{m}_raw"] else 0 for m in labels]
vals_tduf = [sum(metrics[f"{m}_tduf"]) / len(metrics[f"{m}_tduf"]) if metrics[f"{m}_tduf"] else 0 for m in labels]
plt.figure(figsize=(7, 4))
plt.bar([i - 0.2 for i in idx], vals_raw,  width=0.4, label="Baseline LR")
plt.bar([i + 0.2 for i in idx], vals_tduf, width=0.4, label="TD-UF+LR")
plt.xticks(idx, labels)
plt.ylim(0, 1)
plt.title("10-Fold Stratified CV: Baseline vs TD-UF+LR")
plt.ylabel("Score")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("fold10_comparison_bar.png")  # 可存圖
plt.show()

# =============================================================
# 10/10 ▶ 結束
# =============================================================
print("10/10 ▶ 全部完成，Spark 停止！")
spark.stop()
