import polars as pl
import networkx as nx
from tabulate import tabulate

# 讀入交易資料
edge_file = "C:/Users/Leon/Desktop/程式語言資料/python/TD-UF/Anti Money Laundering Transaction Data (SAML-D)/saml_graph_edges/saml_edges.csv"
df = pl.read_csv(edge_file)

# 將 timestamp 轉為 datetime 並排序
df = df.with_columns(
    pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
).sort("timestamp")

# 抽取欄位成為 list
source = df["source"].to_list()
target = df["target"].to_list()
timestamps = df["timestamp"].to_list()

# 建立 DAG
G = nx.DiGraph()
for u, v, ts in zip(source, target, timestamps):
    if not G.has_edge(u, v):
        G.add_edge(u, v, timestamp=str(ts))  # ⚠ 將 timestamp 轉為字串再儲存

# 輸出圖指標
table = [
    ["Number of nodes", G.number_of_nodes()],
    ["Number of edges", G.number_of_edges()],
    ["Is DAG", int(nx.is_directed_acyclic_graph(G))],
]
print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))

# 儲存為 GraphML（無錯誤）
nx.write_graphml(G, "saml_graph_dag.graphml")
print("\n✅ DAG 建構完成，已儲存為：saml_graph_dag.graphml")
