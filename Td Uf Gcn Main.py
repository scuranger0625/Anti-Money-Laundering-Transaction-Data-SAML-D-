#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
td_uf_gcn_main.py
──────────────────────────────────────────────────────────────────────────────
• 以 PySpark 讀取 SAML-D 交易資料
• GCN (PyG) 學習節點洗錢機率
• TD-UF (Temporal-Directed Union-Find) 依機率聚合高風險邊
• 輸出 Precision / Recall / F1 與混淆矩陣
"""
# ── CLI / 系統 ───────────────────────────────────────────────────────────────
import argparse, os, time, warnings
from collections import defaultdict, deque

# ── 表格 & 進度條 ────────────────────────────────────────────────────────────
from tabulate import tabulate
from tqdm import tqdm

# ── PySpark ─────────────────────────────────────────────────────────────────
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp

# ── 科學計算 ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ── PyG & Torch ─────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ── 視覺化 ─────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

# ╭────────────────────────────────────────╮
# │               G C N 模型               │
# ╰────────────────────────────────────────╯
class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# ╭────────────────────────────────────────╮
# │                工具函式                 │
# ╰────────────────────────────────────────╯
def enforce_dag_and_mark(df: pd.DataFrame):
    """移除形成環的邊並標記，確保圖為 DAG"""
    graph, indeg, removed, seen = defaultdict(list), defaultdict(int), set(), set()
    for u, v in tqdm(zip(df["Sender_account"], df["Receiver_account"]),
                     total=len(df), desc="⛏️  檢查 DAG 結構", ncols=100):
        if (u, v) in seen:
            continue
        seen.add((u, v))
        graph[u].append(v)
        indeg[v] += 1

        # 檢測是否形成閉環 u → … → u
        q, visited, cycle = deque([v]), set(), False
        while q:
            cur = q.popleft()
            if cur == u:
                cycle = True
                break
            for nxt in graph.get(cur, []):
                if nxt not in visited:
                    visited.add(nxt); q.append(nxt)

        if cycle:                          # 發現循環 → 撤回此邊
            graph[u].pop(); indeg[v] -= 1; removed.add((u, v))

    df["CycleFlag"] = [(u, v) in removed
                       for u, v in zip(df["Sender_account"], df["Receiver_account"])]
    df["Is_laundering"] = df[["Is_laundering", "CycleFlag"]].max(axis=1)
    return graph, indeg, df


def create_graph_data(df: pd.DataFrame):
    accounts = pd.unique(df[["Sender_account", "Receiver_account"]].values.ravel("K"))
    acc_idx = {acc: i for i, acc in enumerate(accounts)}

    edge_index = torch.tensor(
        [[acc_idx[u], acc_idx[v]]
         for u, v in zip(df["Sender_account"], df["Receiver_account"])],
        dtype=torch.long).t().contiguous()

    # 三個節點特徵：累積金額 / 交易次數 / 是否屬於被刪循環
    x = np.zeros((len(accounts), 3), dtype=np.float32)
    for _, r in df.iterrows():
        i = acc_idx[r["Sender_account"]]
        x[i, 0] += r["Amount"]
        x[i, 1] += 1
        x[i, 2] += r["CycleFlag"]
    x /= (x.max(axis=0, keepdims=True) + 1e-9)

    y = torch.zeros(len(accounts), dtype=torch.long)
    for acc in df.loc[df["Is_laundering"] == 1, "Sender_account"]:
        y[acc_idx[acc]] = 1

    return Data(x=torch.tensor(x), edge_index=edge_index, y=y), acc_idx


def compute_edge_risk(df: pd.DataFrame, acc_idx: dict, node_scores: dict):
    return {(u, v): max(node_scores[acc_idx[u]], node_scores[acc_idx[v]])
            for u, v in zip(df["Sender_account"], df["Receiver_account"])}


def topological_sort(graph: dict, indeg: dict):
    order, q = [], deque([u for u in graph if indeg[u] == 0])
    while q:
        u = q.popleft(); order.append(u)
        for v in graph.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order


def td_uf(graph: dict, topo: list, threshold: float, risk: dict):
    parent = {}

    def find(x):
        if parent.get(x, x) != x:
            parent[x] = find(parent[x])
        return parent.get(x, x)

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa

    for u in tqdm(topo, desc="⚙️  TD-UF 聚合", ncols=100):
        for v in graph.get(u, []):
            if risk.get((u, v), 0) >= threshold:
                union(u, v)
    return parent


def evaluate(df: pd.DataFrame, parent: dict):
    gsend = df["Sender_account"].map(lambda a: parent.get(a, a))
    grecv = df["Receiver_account"].map(lambda a: parent.get(a, a))
    y_true = df["Is_laundering"].astype(int).values
    y_pred = (gsend == grecv).astype(int).values

    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3.6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["Normal", "Launder"],
                yticklabels=["Normal", "Launder"])
    plt.title("Confusion Matrix"); plt.tight_layout(); plt.show()
    return pre, rec, f1


# ╭────────────────────────────────────────╮
# │               主   流   程              │
# ╰────────────────────────────────────────╯
def main(args):
    spark = (
        SparkSession.builder
        .appName("TD-UF-GCN")
        .master("local[*]")
        .config("spark.driver.memory",  os.getenv("SPARK_DRIVER_MEMORY",  "6g"))
        .config("spark.executor.memory", os.getenv("SPARK_EXECUTOR_MEMORY", "6g"))
        .config("spark.sql.shuffle.partitions", str(max(os.cpu_count(), 4)))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("\n[1/5] ➜ 讀取 CSV …")
    df_spark = (spark.read.option("header", True)
                          .option("inferSchema", True)
                          .csv(args.csv)
                          .dropna(subset=["Sender_account", "Receiver_account",
                                          "Amount", "Time", "Is_laundering"]))

    # ── 將含時區字串 → epoch 整數，並移除原欄，完全避免 UnknownTimeZoneError ──
    df_spark = (df_spark
                .withColumn("Time_epoch", unix_timestamp(col("Time")))   # 轉 long
                .drop("Time")
                .withColumnRenamed("Time_epoch", "Time"))

    if args.pandas_cap:
        df_spark = df_spark.limit(args.pandas_cap)   # 抽樣避免 OOM

    df_pd = df_spark.toPandas()
    spark.stop()

    # ───────────────────────────────────────────────────────────────────────
    print("[2/5] ➜ 檢查 DAG / 標記 CycleFlag …")
    graph, indeg, df_pd = enforce_dag_and_mark(df_pd)

    print("[3/5] ➜ 構建 PyG Data & 訓練 GCN …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, acc_idx = create_graph_data(df_pd); data = data.to(device)

    idx = np.arange(data.num_nodes)
    train_idx, test_idx = train_test_split(
        idx, test_size=0.3, stratify=data.y.cpu().numpy(), random_state=42)
    train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)

    model = GCN(data.num_node_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 21):
        model.train(); optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_idx], data.y[train_idx])
        loss.backward(); optimizer.step()
        tqdm.write(f"  Epoch {epoch:02d}/20  Loss={loss.item():.4f}")

    model.eval() 
    with torch.no_grad():
        probs = F.softmax(model(data.x, data.edge_index), dim=1)[:, 1].cpu().numpy()
    
    node_score = {i: p for i, p in enumerate(probs)}

    print("[4/5] ➜ TD-UF 聚合 …")
    risk_dict = compute_edge_risk(df_pd, acc_idx, node_score)
    topo      = topological_sort(graph, indeg.copy())
    parent    = td_uf(graph, topo, args.risk_threshold, risk_dict)

    pre, rec, f1 = evaluate(df_pd, parent)
    print("\n" + tabulate([
        ["Metric",   "Score"],
        ["Precision", f"{pre:.3f}"],
        ["Recall",    f"{rec:.3f}"],
        ["F1-Score",  f"{f1:.3f}"],
    ], headers="firstrow", tablefmt="github"))

    print(f"\n⏱  Done in {time.time() - START_TIME:.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TD-UF + GCN Anti-Money-Laundering")
    parser.add_argument("--csv", required=True, help="SAML-D.csv 路徑")
    parser.add_argument("--risk-threshold", type=float, default=0.5,
                        help="邊風險閾值 (0-1)")
    parser.add_argument("--pandas-cap", type=int, default=250_000,
                        help="轉為 pandas 的最大列數 (防 OOM)")
    args = parser.parse_args()

    START_TIME = time.time()
    main(args)
