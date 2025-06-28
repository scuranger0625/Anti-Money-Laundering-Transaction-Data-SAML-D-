import polars as pl
import networkx as nx
from tabulate import tabulate
from tqdm import tqdm  # ✅ 加入進度條

# === 讀取邊資料 === #
edge_file = "C:/Users/Leon/Desktop/程式語言資料/python/TD-UF/Anti Money Laundering Transaction Data (SAML-D)/SAML-D.csv"
df = pl.read_csv(edge_file)

# === 建立 timestamp 並排序 === #
df = df.with_columns([
    pl.concat_str(["Date", "Time"], separator=" ").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("timestamp")
]).sort("timestamp")

# === 初始化 TD-Union-Find 結構 === #
parent = {}

def find(u):
    if parent[u] != u:
        parent[u] = find(parent[u])
    return parent[u]

def union(u, v):
    pu, pv = find(u), find(v)
    if pu != pv:
        parent[pv] = pu

# === 建立 DAG === #
G_dag = nx.DiGraph()
added_edges = 0

# ✅ 加入 tqdm 顯示進度
for row in tqdm(df.iter_rows(named=True), total=df.height, desc="Building DAG"):
    u = row["Sender_account"]
    v = row["Receiver_account"]
    ts = str(row["timestamp"])

    for node in [u, v]:
        if node not in parent:
            parent[node] = node

    if find(u) != find(v):
        G_dag.add_edge(u, v, timestamp=ts)
        union(u, v)
        added_edges += 1

# === 統計輸出 === #
table = [
    ["Original nodes", df.select(["Sender_account", "Receiver_account"]).unique().height],
    ["Original edges", df.shape[0]],
    ["DAG edges retained", added_edges],
    ["Final node count", G_dag.number_of_nodes()],
    ["Is DAG", int(nx.is_directed_acyclic_graph(G_dag))],
]
print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))

# === 輸出 GraphML 檔案 === #
for u, v, d in G_dag.edges(data=True):
    d["timestamp"] = str(d["timestamp"])
nx.write_graphml(G_dag, "saml_graph_dag_td_uf.graphml")

print("\n✅ 使用 TD-UF 成功建構 DAG，檔案：saml_graph_dag_td_uf.graphml")
