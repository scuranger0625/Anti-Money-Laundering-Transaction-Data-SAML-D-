# Union-Find 簡化版 (含路徑壓縮)
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))  # 初始每個節點的父節點指向自己
        self.size = [1] * n           # 用 size 控制合併，減少樹高度

    def find(self, x):
        # 路徑壓縮：讓所有節點直接指向根
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        # 讓小樹掛到大樹下
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
