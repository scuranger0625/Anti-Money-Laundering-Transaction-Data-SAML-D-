import polars as pl
import networkx as nx
from collections import deque
from tabulate import tabulate
from tqdm import tqdm
from time import time

# === DAG GraphML 檔案路徑 === #
dag_path = "C:/Users/Leon/Desktop/程式語言資料/python/TD-UF/Anti Money Laundering Transaction Data (SAML-D)/saml_graph_dag_td_uf.graphml"

# === 載入 GraphML DAG === #
print("🔄 載入 DAG 檔案中...")
G = nx.read_graphml(dag_path)

# === 處理時間戳（防錯）=== #
for u, v, d in G.edges(data=True):
    if "timestamp" in d:
        d["timestamp"] = str(d["timestamp"])

# === Kahn 拓撲排序 === #
def kahn_topo_sort(graph):
    in_deg = {u: 0 for u in graph}
    for _, v in graph.edges():
        in_deg[v] += 1
    queue = deque([u for u, deg in in_deg.items() if deg == 0])
    topo = []

    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in graph.successors(u):
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    return topo

# === Union-Find 結構 === #
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu != pv:
            self.parent[pv] = pu

    def add(self, u):
        if u not in self.parent:
            self.parent[u] = u

# === PoH 評分（越長越高風險）=== #
def poh_score(path, G):
    return len(path) - 1

# === 主程式開始 === #
start = time()
topo_order = kahn_topo_sort(G)

uf = UnionFind()
for node in topo_order:
    uf.add(node)

# === 建立疑似風險路徑 === #
risk_paths = []
print("🚧 分析風險路徑中...")
for node in tqdm(reversed(topo_order), total=len(topo_order)):
    path = [node]
    cur = node
    while True:
        succs = list(G.successors(cur))
        if not succs:
            break
        cur = succs[0]
        path.append(cur)
    if len(path) >= 5:
        risk_paths.append(path)

# === 進行 PoH 評分與排序 === #
scores = [(p, poh_score(p, G)) for p in risk_paths]
scores.sort(key=lambda x: -x[1])  # 依據分數由高到低

elapsed = time() - start

# === 輸出總結 === #
table = [
    ["Total Nodes", G.number_of_nodes()],
    ["Total Edges", G.number_of_edges()],
    ["Topological Order Length", len(topo_order)],
    ["Detected Risk Paths", len(scores)],
    ["Top PoH Score", scores[0][1] if scores else 0],
    ["Runtime (sec)", f"{elapsed:.2f}"],
]
print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))

# 顯示範例高風險路徑
if scores:
    print("\n⚠️ High Risk Path Example (max 10 nodes):")
    print(" → ".join(scores[0][0][:10]))
