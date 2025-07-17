print("🚀 開始執行轉檔...")

import polars as pl

# ✅ 修正路徑錯誤，使用 raw string
csv_path = r"C:\Users\Leon\Desktop\程式語言資料\python\TD-UF\Anti Money Laundering Transaction Data (SAML-D)\SAML-D.csv"
parquet_path = csv_path.replace(".csv", ".parquet")

try:
    df = pl.read_csv(csv_path, infer_schema_length=1000)
    print(f"📥 CSV 讀取成功！共 {df.shape[0]} 筆資料，{df.shape[1]} 欄位")
except Exception as e:
    print("❌ 錯誤：無法讀取 CSV！")
    print(e)
    exit(1)

try:
    df.write_parquet(parquet_path, compression="zstd")
    print(f"✅ Parquet 儲存成功：{parquet_path}")
except Exception as e:
    print("❌ 錯誤：無法儲存 Parquet！")
    print(e)
