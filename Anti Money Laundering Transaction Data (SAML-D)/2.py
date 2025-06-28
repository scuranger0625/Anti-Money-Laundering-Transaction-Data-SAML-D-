import pandas as pd

# 讀取 SAML-D 資料集
file_path = "C:/Users/Leon/Desktop/程式語言資料/python/TD-UF/Anti Money Laundering Transaction Data (SAML-D)/SAML-D.csv"

# 只讀前幾列看欄位名稱與型態
df = pd.read_csv(file_path, nrows=10)  # 預設用 utf-8 編碼，如有錯誤可加 encoding='ISO-8859-1' 或 'utf-8-sig'

# 印出欄位名稱
print("🔍 欄位名稱如下：")
print(df.columns.tolist())

# 顯示前幾筆資料
print("\n📌 前 5 筆資料：")
print(df.head())
