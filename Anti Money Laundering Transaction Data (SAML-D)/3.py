import os
import pandas as pd

# === 檔案與資料夾設定 === #
base_dir = "C:/Users/Leon/Desktop/程式語言資料/python/TD-UF/Anti Money Laundering Transaction Data (SAML-D)"
csv_path = os.path.join(base_dir, "SAML-D.csv")
node_path = os.path.join(base_dir, "saml_graph_nodes", "saml_nodes.csv")
edge_path = os.path.join(base_dir, "saml_graph_edges", "saml_edges.csv")

# 確保資料夾存在
os.makedirs(os.path.dirname(node_path), exist_ok=True)
os.makedirs(os.path.dirname(edge_path), exist_ok=True)

# === 載入資料 === #
df = pd.read_csv(csv_path)

# === 建立節點資料（唯一帳戶 ID）=== #
sender_nodes = df["Sender_account"].dropna().astype(str)
receiver_nodes = df["Receiver_account"].dropna().astype(str)
all_nodes = pd.Series(pd.concat([sender_nodes, receiver_nodes]).unique(), name="account_id")
all_nodes.to_csv(node_path, index=False)

# === 建立邊資料（含交易屬性）=== #
edge_df = df[[
    "Sender_account", "Receiver_account", "Date", "Time",
    "Payment_type", "Is_laundering", "Laundering_type"
]].copy()
edge_df = edge_df.dropna(subset=["Sender_account", "Receiver_account"])  # 移除缺失
edge_df["source"] = edge_df["Sender_account"].astype(str)
edge_df["target"] = edge_df["Receiver_account"].astype(str)
edge_df["timestamp"] = edge_df["Date"].astype(str) + " " + edge_df["Time"].astype(str)

# 保留欄位順序
edge_output = edge_df[[
    "source", "target", "timestamp", "Payment_type", "Is_laundering", "Laundering_type"
]]
edge_output.to_csv(edge_path, index=False)

print("✅ 節點與邊資料輸出完成")
print(f"節點檔案：{node_path}")
print(f"邊檔案：{edge_path}")
