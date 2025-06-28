import polars as pl
import networkx as nx
import os

# === 1. 設定路徑 === #
base_dir = "C:/Users/Leon/Desktop/程式語言資料/python/TD-UF/Anti Money Laundering Transaction Data (SAML-D)"
csv_path = os.path.join(base_dir, "SAML-D.csv")

# === 2. 讀入 CSV 資料（自動推論欄位型別） === #
df = pl.read_csv(csv_path)

# === 3. 建立時間欄位（合併 Date 與 Time 為 timestamp） === #
df = df.with_columns([
    (df["Date"].cast(pl.Utf8) + " " + df["Time"].cast(pl.Utf8)).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S").alias("timestamp")
])

# === 4. 初始化有向圖 === #
G = nx.DiGraph()

# === 5. 建立節點與邊 === #
for row in df.iter_rows(named=True):
    sender = row["Sender_account"]
    receiver = row["Receiver_account"]

    # 加入節點（帳戶 ID）
    G.add_node(sender)
    G.add_node(receiver)

    # 加入邊：sender → receiver，附加屬性
    G.add_edge(sender, receiver, 
               timestamp=row["timestamp"],
               payment_type=row["Payment_type"],
               is_laundering=row["Is_laundering"],
               laundering_type=row["Laundering_type"])

# === 6. 顯示圖的大小 === #
print("✅ 圖建構完成")
print(f"圖中共有 {G.number_of_nodes()} 個帳戶節點")
print(f"圖中共有 {G.number_of_edges()} 筆交易邊")

# === 7. 輸出邊與節點成 CSV 檔案（可給後續演算法使用） === #
node_out_path = os.path.join(base_dir, "saml_nodes.csv")
edge_out_path = os.path.join(base_dir, "saml_edges.csv")

# 儲存節點（僅帳戶 ID）
pl.DataFrame({"account_id": list(G.nodes)}).write_csv(node_out_path)

# 儲存邊（來源、目標與屬性）
edge_records = []
for u, v, data in G.edges(data=True):
    edge_records.append({
        "source": u,
        "target": v,
        "timestamp": data["timestamp"],
        "payment_type": data["payment_type"],
        "is_laundering": data["is_laundering"],
        "laundering_type": data["laundering_type"]
    })

pl.DataFrame(edge_records).write_csv(edge_out_path)
print(f"✅ 邊與節點已輸出至：\n{node_out_path}\n{edge_out_path}")
