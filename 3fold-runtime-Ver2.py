import os
import time
from tqdm import trange
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lag, count, avg, stddev, min as spark_min, max as spark_max,
    broadcast, monotonically_increasing_id, row_number
)
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.model_selection import StratifiedKFold

# --- GCS儲存路徑 ---
GCS_BUCKET = "gs://saml-d/"

# --- 啟動SparkSession ---
print("▶️ 啟動 SparkSession...")
spark = SparkSession.builder.appName("Baseline vs TD-UF with Kahn").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setCheckpointDir(GCS_BUCKET + "spark-checkpoint")
print("✅ SparkSession 完成\n")

# --- 載入 SAML-D 標註資料 ---
print("▶️ 載入 SAML-D 資料...")
df_saml = (
    spark.read.option("header", True).option("inferSchema", True)
    .csv(GCS_BUCKET + "SAML-D.csv")
    .withColumn("tx_id", monotonically_increasing_id())
    .withColumn("timestamp", col("Time").cast("long"))
    .withColumn("label", col("Is_laundering").cast("integer"))
)
print("✅ SAML-D 載入完成\n")

# --- 載入交易雜湊表，補上label ---
print("▶️ 載入 PoH_Hashed_Transactions...")
df_hash = (
    spark.read.option("header", True).option("inferSchema", True)
    .csv(GCS_BUCKET + "PoH_Hashed_Transactions.csv")
    .withColumn("tx_id", monotonically_increasing_id())
)
df_hash = df_hash.join(broadcast(df_saml.select("tx_id", "label")), on="tx_id", how="left").na.fill({"label": 0})
print("✅ PoH 資料載入完成\n")

# --- 基礎清理 ---
print("▶️ 資料清理...")
df_saml = df_saml.na.drop(subset=["Amount", "timestamp", "label"])
df_hash = df_hash.na.drop(subset=["Amount", "timestamp", "label"])
print("✅ 清理完成\n")

# --- KFold 分配 ---
print("▶️ 分配 Stratified K-Fold...")
saml_pd = df_saml.select("tx_id", "label").toPandas()
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
saml_pd["fold"] = -1
for fold, (_, test_idx) in enumerate(skf.split(saml_pd["tx_id"], saml_pd["label"])):
    saml_pd.loc[test_idx, "fold"] = fold
folds_df = spark.createDataFrame(saml_pd[["tx_id", "fold"]])
df_saml = df_saml.join(folds_df, on="tx_id")
df_hash = df_hash.join(folds_df, on="tx_id")
print("✅ Fold 完成\n")

# --- 特徵組裝器 ---
assembler_raw = VectorAssembler(inputCols=["Amount", "timestamp"], outputCol="features_raw")
assembler_tduf = VectorAssembler(inputCols=[
    "tx_count", "avg_amt", "std_amt", "max_amt", "min_amt",
    "avg_gap", "std_gap", "max_gap", "min_gap"
], outputCol="features_tduf")

# --- 指標初始化 ---
print("▶️ 初始化評估器與指標容器...")
evaluator = BinaryClassificationEvaluator(labelCol="label")
metrics = {k: [] for k in [
    'Acc_raw', 'Prec_raw', 'Rec_raw', 'F1_raw', 'AUC_raw', 'T_raw',
    'Acc_tduf', 'Prec_tduf', 'Rec_tduf', 'F1_tduf', 'AUC_tduf', 'T_tduf'
]}
print("✅ 初始化完成\n")

# --- 3-Fold 交叉驗證 ---
print("▶️ 開始 3-Fold 交叉驗證\n")
for fold in trange(3, desc="3-Fold Validation"):
    train_s = df_saml.filter(col("fold") != fold)
    test_s  = df_saml.filter(col("fold") == fold)
    train_h = df_hash.filter(col("fold") != fold)
    test_h  = df_hash.filter(col("fold") == fold)

    # ===== Baseline 模型流程 =====
    print(f"\n▶️ Fold {fold}: Baseline 訓練")
    t0 = time.time()
    tr_r = assembler_raw.transform(train_s).select("features_raw", "label")
    te_r = assembler_raw.transform(test_s).select("features_raw", "label")
    if tr_r.count() == 0 or te_r.count() == 0:
        continue
    model_r = LogisticRegression(featuresCol="features_raw", labelCol="label").fit(tr_r)
    pred_r  = model_r.transform(te_r)
    t1 = time.time()
    tp = pred_r.filter((col("label") == 1) & (col("prediction") == 1)).count()
    tn = pred_r.filter((col("label") == 0) & (col("prediction") == 0)).count()
    fp = pred_r.filter((col("label") == 0) & (col("prediction") == 1)).count()
    fn = pred_r.filter((col("label") == 1) & (col("prediction") == 0)).count()
    total = tp + tn + fp + fn
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    auc  = evaluator.evaluate(pred_r, {evaluator.metricName: "areaUnderROC"})
    acc  = (tp + tn) / total if total > 0 else 0
    metrics['Acc_raw'].append(acc)
    metrics['Prec_raw'].append(prec)
    metrics['Rec_raw'].append(rec)
    metrics['F1_raw'].append(f1)
    metrics['AUC_raw'].append(auc)
    metrics['T_raw'].append(t1 - t0)

    # ===== TD-UF 模型流程（無洩漏）=====
    print(f"▶️ Fold {fold}: TD-UF 建模（防止洩漏）")
    t2 = time.time()
    # 1. 只用 train fold 構建資金流圖
    edges = train_h.alias("a").join(
        train_h.alias("b"),
        (col("a.Receiver_account") == col("b.Sender_account")) & (col("a.timestamp") < col("b.timestamp")),
        "inner"
    ).select(col("a.tx_id").alias("src"), col("b.tx_id").alias("dst"))

    from_nodes = edges.select("src").distinct()
    all_nodes = train_h.select("tx_id").distinct()
    roots = all_nodes.join(from_nodes, all_nodes.tx_id == from_nodes.src, "left_anti")
    roots = roots.withColumn("component", row_number().over(Window.orderBy("tx_id")))
    components = roots.select("tx_id", "component")

    for _ in trange(3, desc=f"Propagate Fold {fold}", leave=False):
        new_edges = edges.join(components, edges.src == components.tx_id).select(edges.dst.alias("tx_id"), "component").distinct()
        components = components.union(new_edges).dropDuplicates(["tx_id"]).checkpoint()

    # 2. 只對train fold 交易貼上component id並進行component聚合
    train_with_comp = train_h.join(components, on="tx_id", how="left")
    w = Window.partitionBy("component").orderBy("timestamp")
    tmp = train_with_comp.withColumn("prev_ts", lag("timestamp", 1).over(w)).withColumn("gap", col("timestamp") - col("prev_ts"))
    agg = tmp.groupBy("component").agg(
        count("*").alias("tx_count"),
        avg("Amount").alias("avg_amt"),
        stddev("Amount").alias("std_amt"),
        spark_max("Amount").alias("max_amt"),
        spark_min("Amount").alias("min_amt"),
        avg("gap").alias("avg_gap"),
        stddev("gap").alias("std_gap"),
        spark_max("gap").alias("max_gap"),
        spark_min("gap").alias("min_gap"),
        spark_max("label").alias("label_agg")
    ).na.fill(0)

    # 聚合欄位名稱不跟原始label重名
    feat_tr = train_with_comp.join(agg, on="component", how="left")
    # assembler_tduf.transform時保留原始label（單筆）
    tr_t = assembler_tduf.transform(feat_tr).select("features_tduf", "label")


    # 3. test fold只能查找train fold propagate出來的component id與特徵，查不到都補0
    test_with_comp = test_h.join(components, on="tx_id", how="left").fillna({"component": -1})
    feat_tr = train_with_comp.join(agg, on="component", how="left")
    feat_te = test_with_comp.join(agg, on="component", how="left")
    for colname in [
        "tx_count", "avg_amt", "std_amt", "max_amt", "min_amt",
        "avg_gap", "std_gap", "max_gap", "min_gap"
    ]:
        feat_te = feat_te.na.fill({colname: 0})

    tr_t = assembler_tduf.transform(feat_tr).select("features_tduf", "label")
    te_t = assembler_tduf.transform(feat_te).select("features_tduf", "label")
    if tr_t.count() == 0 or te_t.count() == 0:
        continue
    model_t = LogisticRegression(featuresCol="features_tduf", labelCol="label").fit(tr_t)
    pred_t  = model_t.transform(te_t)
    t3 = time.time()
    tp = pred_t.filter((col("label") == 1) & (col("prediction") == 1)).count()
    tn = pred_t.filter((col("label") == 0) & (col("prediction") == 0)).count()
    fp = pred_t.filter((col("label") == 0) & (col("prediction") == 1)).count()
    fn = pred_t.filter((col("label") == 1) & (col("prediction") == 0)).count()
    total = tp + tn + fp + fn
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    auc  = evaluator.evaluate(pred_t, {evaluator.metricName: "areaUnderROC"})
    acc  = (tp + tn) / total if total > 0 else 0
    metrics['Acc_tduf'].append(acc)
    metrics['Prec_tduf'].append(prec)
    metrics['Rec_tduf'].append(rec)
    metrics['F1_tduf'].append(f1)
    metrics['AUC_tduf'].append(auc)
    metrics['T_tduf'].append(t3 - t2)

# --- 匯出與總結 ---
print("\n📊 平均指標總結")
summary = []
for label in ["Acc", "Prec", "Rec", "F1", "AUC"]:
    summary.append({
        "指標": label,
        "Baseline": sum(metrics[f"{label}_raw"]) / len(metrics[f"{label}_raw"]),
        "TD-UF": sum(metrics[f"{label}_tduf"]) / len(metrics[f"{label}_tduf"])
    })
    print(f"{label:12s} Baseline: {summary[-1]['Baseline']:.4f}  TD-UF: {summary[-1]['TD-UF']:.4f}")

runtime_row = {
    "指標": "Runtime (sec)",
    "Baseline": sum(metrics["T_raw"]) / len(metrics["T_raw"]) if metrics["T_raw"] else 0,
    "TD-UF": sum(metrics["T_tduf"]) / len(metrics["T_tduf"]) if metrics["T_tduf"] else 0
}
summary.append(runtime_row)
print(f"Runtime     Baseline: {runtime_row['Baseline']:.2f} s   TD-UF: {runtime_row['TD-UF']:.2f} s")

pd.DataFrame(summary).to_csv("/tmp/fold3_runtimeVer2.csv", index=False, encoding="utf-8-sig")
os.system(f"gsutil cp /tmp/fold3_runtimeVer2.csv {GCS_BUCKET}")
spark.stop()
print("\n🏁 任務完成，SparkSession 結束")
