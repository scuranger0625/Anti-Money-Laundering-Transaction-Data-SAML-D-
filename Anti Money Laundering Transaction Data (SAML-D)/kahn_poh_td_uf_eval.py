import polars as pl
import networkx as nx
from collections import deque
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time

# === 檔案路徑 === #
dag_path = "C:/Users/Leon/Desktop/程式語言資料/python/TD-UF/Anti Money Laundering Transaction Data (SAML-D)/saml_graph_dag_td_uf.graphml"
csv_path = "C:/Users/Leon/Desktop/程式語言資料/python/TD-UF/Anti Money Laundering Transaction Data (SAML-D)/SAML-D.csv"

# === 載入原始資料（標記非法帳戶）=== #
raw_df = pl.read_csv(csv_path)
laundering_accounts = set(
    raw_df.filter(pl.col("Is_laundering") == 1)
          .select(["Sender_account", "Receiver_account"])
          .to_series(0)
          .to_list() +
    raw_df.filter(pl.col("Is_laundering") == 1)
          .select("Receiver_account")
          .to_series()
          .to_list()
)

# === 載入 DAG === #
print("🔄 載入 DAG 檔案中...")
G = nx.read_graphml(dag_path)
for u, v, d in G.edges(data=True):
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

# === Union-Find === #
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

# === PoH 評分（可換成自訂函數）=== #
def poh_score(path):
    return len(path) - 1

# === 主流程 === #
start = time()
topo_order = kahn_topo_sort(G)
uf = UnionFind()
for node in topo_order:
    uf.add(node)

# === 掃描高風險路徑 === #
risk_paths = []
print("🚧 掃描風險路徑中...")
for node in tqdm(reversed(topo_order), total=len(topo_order)):
    path = [node]
    cur = node
    while True:
        succs = list(G.successors(cur))
        if not succs: break
        cur = succs[0]
        path.append(cur)
    if len(path) >= 5:
        risk_paths.append(path)

# === 得分排序 === #
scored = [(p, poh_score(p)) for p in risk_paths]
scored.sort(key=lambda x: -x[1])

elapsed = time() - start

# === 準確率、召回率、F1-score === #
predicted_accounts = set()
for path, _ in scored[:1000]:  # Top-K 風險路徑節點
    predicted_accounts.update(path)

TP = len(predicted_accounts & laundering_accounts)
FP = len(predicted_accounts - laundering_accounts)
FN = len(laundering_accounts - predicted_accounts)
precision = TP / (TP + FP + 1e-9)
recall = TP / (TP + FN + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

# === 輸出統計 === #
table = [
    ["Total Nodes", G.number_of_nodes()],
    ["Total Edges", G.number_of_edges()],
    ["Detected Risk Paths", len(scored)],
    ["Top PoH Score", scored[0][1] if scored else 0],
    ["Runtime (sec)", f"{elapsed:.2f}"],
    ["TP", TP], ["FP", FP], ["FN", FN],
    ["Precision", f"{precision:.4f}"],
    ["Recall", f"{recall:.4f}"],
    ["F1-score", f"{f1:.4f}"],
]
print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))

# === 畫出最高風險路徑圖（最多前10個節點）=== #
if scored:
    G_sub = nx.DiGraph()
    high_risk_path = scored[0][0][:10]
    for i in range(len(high_risk_path) - 1):
        G_sub.add_edge(high_risk_path[i], high_risk_path[i+1])
    plt.figure(figsize=(10, 5))
    nx.draw(G_sub, with_labels=True, node_size=700, font_size=8, node_color="lightcoral", edge_color="gray")
    plt.title("Top-Risk Path (PoH)")
    plt.tight_layout()
    plt.savefig("high_risk_path.png")
    print("📊 已輸出高風險路徑圖 high_risk_path.png")
