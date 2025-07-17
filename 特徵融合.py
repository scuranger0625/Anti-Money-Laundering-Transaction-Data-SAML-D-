import polars as pl
import networkx as nx
from tqdm import tqdm   # 進度條女友！

# 讀 Parquet
file_path = r"C:\Users\Leon\Desktop\程式語言資料\python\TD-UF\Anti Money Laundering Transaction Data (SAML-D)\SAML-D.parquet"
df = pl.read_parquet(file_path)

edges = df.select(["Sender_account", "Receiver_account"]).rows()
G = nx.DiGraph()
G.add_edges_from(edges)

# WCC分群
wccs = list(nx.weakly_connected_components(G))
node2group = {}
for i, group in enumerate(tqdm(wccs, desc="分配群組編號")):   # 加進度條
    for node in group:
        node2group[node] = i

group_ids = [node2group[sender] for sender, _ in edges]
df = df.with_columns(pl.Series("group_id", group_ids))

# 預算空間：存每個群組的中心性結果
group_info = {}
closeness_dict = {}
betweenness_dict = {}

for i, group in enumerate(tqdm(wccs, desc="計算中心性")):   # 這裡也加
    subG = G.subgraph(group)
    nodes = list(subG.nodes())
    edges = list(subG.edges())
    total = len(edges)
    bidirect = sum(1 for (u, v) in edges if subG.has_edge(v, u))
    ratio = bidirect / total if total else 0

    # 只計算有三個以上節點的群組（避免小群中心性不穩）
    if len(nodes) >= 3:
        undirected_subG = subG.to_undirected()
        closeness = nx.closeness_centrality(undirected_subG)
        betweenness = nx.betweenness_centrality(undirected_subG)
    else:
        closeness = {n: 0 for n in nodes}
        betweenness = {n: 0 for n in nodes}

    group_info[i] = {
        "node_count": len(nodes),
        "edge_count": total,
        "bidirect_ratio": ratio,
    }
    for n in nodes:
        closeness_dict[n] = closeness[n]
        betweenness_dict[n] = betweenness[n]

df = df.with_columns([
    pl.Series("group_node_count", [group_info[g]["node_count"] for g in group_ids]),
    pl.Series("group_edge_count", [group_info[g]["edge_count"] for g in group_ids]),
    pl.Series("group_bidirect_ratio", [group_info[g]["bidirect_ratio"] for g in group_ids]),
])

# 加入 sender/receiver 的 degree, closeness, betweenness（群組內計算）
sender_degrees = []
receiver_degrees = []
sender_closeness = []
receiver_closeness = []
sender_betweenness = []
receiver_betweenness = []

for sender, receiver in tqdm(zip(df["Sender_account"], df["Receiver_account"]),
                             total=len(df),
                             desc="節點特徵寫入"):   # 這裡也加進度條
    sender_degrees.append(G.degree(sender))
    receiver_degrees.append(G.degree(receiver))
    sender_closeness.append(closeness_dict.get(sender, 0))
    receiver_closeness.append(closeness_dict.get(receiver, 0))
    sender_betweenness.append(betweenness_dict.get(sender, 0))
    receiver_betweenness.append(betweenness_dict.get(receiver, 0))

df = df.with_columns([
    pl.Series("sender_degree", sender_degrees),
    pl.Series("receiver_degree", receiver_degrees),
    pl.Series("sender_closeness", sender_closeness),
    pl.Series("receiver_closeness", receiver_closeness),
    pl.Series("sender_betweenness", sender_betweenness),
    pl.Series("receiver_betweenness", receiver_betweenness),
])

out_path = r"C:\Users\Leon\Desktop\程式語言資料\python\TD-UF\Anti Money Laundering Transaction Data (SAML-D)\SAML-D_with_graph_centrality.parquet"
df.write_parquet(out_path)
print("✅ 中心性指標計算並儲存完畢！")

