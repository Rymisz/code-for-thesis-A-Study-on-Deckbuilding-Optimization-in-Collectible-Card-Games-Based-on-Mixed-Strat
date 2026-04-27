import numpy as np
from itertools import combinations
from scipy.optimize import linprog

# 1. 定义7个基础阵营及名称
factions = ["红", "黄", "蓝", "绿", "紫", "黑", "白"]
n_factions = len(factions)

# 2. 录入微调并放大2倍后的基准克制矩阵 B (7x7)
B = np.array([
    [ 0, -4, -2,  6,  4,  1, -4],
    [ 4,  0, -2, -2, -4, -1,  4],
    [ 2,  2,  0,  4,  4, -4, -2],
    [-6,  2, -4,  0, -4,  8,  4],
    [-4,  4, -4,  4,  0, -2,  4],
    [-1,  1,  4, -8,  2,  0, -4],
    [ 4, -4,  2, -4, -4,  4,  0]
])

# 3. 生成所有 21 种双色卡组组合
deck_combinations = list(combinations(range(n_factions), 2))
deck_names = [f"{factions[i]}{factions[j]}" for i, j in deck_combinations]
n_decks = len(deck_combinations)

# 4. 构建卡组特征向量矩阵 V (7 x 21)
V = np.zeros((n_factions, n_decks))
for idx, (i, j) in enumerate(deck_combinations):
    V[i, idx] = 0.5
    V[j, idx] = 0.5

# 5. 利用双线性映射计算 21x21 的全局收益矩阵 A
A = V.T @ B @ V

# 打印 21 套卡组的名称，方便后续对号入座
print(A)
print(deck_names)

n = A.shape[0]

# 我们要求解的是玩家的概率分布 X = [x_1, x_2, ..., x_21] 和博弈价值 v
# 将其转化为 linprog 的标准型变量向量：[x_1, x_2, ..., x_21, v]
# 目标函数：最大化 v，即最小化 -v
c = np.zeros(n + 1)
c[-1] = -1

# 不等式约束：对于对手的每一个策略 j，我方的期望收益必须 >= v
# 即 A^T * X >= v * 1  转化为标准小于等于号形式:  -A^T * X + v * 1 <= 0
A_ub = np.hstack((-A.T, np.ones((n, 1))))
b_ub = np.zeros(n)

# 等式约束：所有概率之和必须为 1
A_eq = np.ones((1, n + 1))
A_eq[0, -1] = 0  # 变量 v 不参与概率求和
b_eq = np.array([1])

# 变量边界：概率 x_i >= 0，博弈价值 v 没有上下限 (由于是反对称零和博弈，理论上 v 最终会是 0)
bounds = [(0, None)] * n + [(None, None)]

# 使用 highs 算法求解线性规划
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

if res.success:
    # 提取概率分布 (前 21 个变量)
    probabilities = res.x[:-1]
    game_value = res.x[-1]
    
    print(f"=== 天梯 MSNE 稳态求解成功 ===")
    print(f"博弈期望价值 v: {game_value:.4f} (理论应为0)")
    print("\n最终存活的 Meta 核心卡组及其出场率：")
    
    # 过滤掉出场率接近 0 的边缘卡组
    meta_decks = []
    for i, p in enumerate(probabilities):
        if p > 0.001:  # 设定 0.1% 的浮点数容差阈值
            meta_decks.append((deck_names[i], p))
            
    # 按出场率从高到低排序
    meta_decks.sort(key=lambda x: x[1], reverse=True)
    
    for name, p in meta_decks:
        print(f"{name}: {p*100:.2f}%")
else:
    print("求解失败:", res.message)



