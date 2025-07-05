import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, sha2

# ====== 1. 設定環境變數（使用 Gephi 的 JRE）======
os.environ["JAVA_HOME"] = r"C:\Program Files\Gephi-0.10.1\jre-x64\jdk-11.0.17+8-jre"
os.environ["HADOOP_HOME"] = r"C:\Winutils"
os.environ["PATH"] = os.environ["JAVA_HOME"] + ";" + os.environ["HADOOP_HOME"] + r"\bin;" + os.environ["PATH"]

# ====== 2. 啟動 SparkSession ======
print("🚀 啟動 SparkSession...")
spark = SparkSession.builder \
    .appName("PoH DAG Hash Construction") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.maxResultSize", "2g") \
    .getOrCreate()
print("✔️ SparkSession 建立完成")

# ====== 3. 讀取資料 ======
input_path = r"C:\Users\Leon\Desktop\程式語言資料\python\TD-UF\Anti Money Laundering Transaction Data (SAML-D)\SAML-D.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(input_path)

# ====== 4. 建立 timestamp 與 poh_input ======
df = df.withColumn("timestamp", concat_ws(" ", col("Date"), col("Time")))
df = df.withColumn("poh_input", concat_ws("|",
    concat_ws("->", col("Sender_account").cast("string"), col("Receiver_account").cast("string")),
    col("Amount").cast("string"),
    col("timestamp")
))
df = df.withColumn("PoH_hash", sha2(col("poh_input"), 256))

# ====== 5. 設定輸出路徑並建立資料夾 ======
output_dir = r"C:\TDUF_OUTPUT"
output_path = os.path.join(output_dir, "PoH_Hashed_Transactions.csv")
os.makedirs(output_dir, exist_ok=True)

# ====== 6. 輸出為 CSV ======
df_pd = df.select("Sender_account", "Receiver_account", "Amount", "timestamp", "poh_input", "PoH_hash").toPandas()
df_pd.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"✅ 已成功輸出至：{output_path}")
print(df_pd.head())

# ====== 7. 結束 SparkSession ======
spark.stop()
