import numpy as np
from scipy.optimize import root
from itertools import combinations
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体，防止 matplotlib 画图时中文显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

# 1. 初始化矩阵
factions = ["红", "黄", "蓝", "绿", "紫", "黑", "白"]
n_fac = len(factions)
deck_combos = list(combinations(range(n_fac), 2))
deck_names = [f"{factions[i]}{factions[j]}" for i, j in deck_combos]

# 基础收益矩阵 B
B_base = np.array([
    [ 0, -4, -2,  6,  4,  1, -4],
    [ 4,  0, -2, -2, -4, -1,  4],
    [ 2,  2,  0,  4,  4, -4, -2],
    [-6,  2, -4,  0, -4,  8,  4],
    [-4,  4, -4,  4,  0, -2,  4],
    [-1,  1,  4, -8,  2,  0, -4],
    [ 4, -4,  2, -4, -4,  4,  0]
], dtype=float)

V = np.zeros((n_fac, len(deck_combos)))
for idx, (i, j) in enumerate(deck_combos):
    V[i, idx] = 0.5
    V[j, idx] = 0.5

# 提取生成全局收益矩阵 A 的函数，方便后续加扰动
def get_A(B_matrix):
    return V.T @ B_matrix @ V 

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum()

def qre_fixed_point(x, A, lam):
    x = np.clip(x, 0, None)
    if x.sum() > 0:
        x = x / x.sum()
    else:
        x = np.ones(len(x)) / len(x)
        
    expected_payoffs = A @ x
    target_x = softmax(lam * expected_payoffs)
    return x - target_x

# ==========================================
# 任务 1: 绘制不同 lambda 下最高出场率卡组的变化
# ==========================================
lambdas = np.linspace(0.1, 15, 60) # 设置 lambda 的变化范围
probs_lambda = []
x0 = np.ones(21) / 21
A_base = get_A(B_base)

print("正在计算 lambda 变化曲线...")
for lam in lambdas:
    sol = root(qre_fixed_point, x0, args=(A_base, lam), method='hybr')
    if sol.success:
        p = np.clip(sol.x, 0, None)
        p /= p.sum()
        probs_lambda.append(p)
        x0 = p # 热启动：用当前的解作为下一步的初始猜测，提高收敛率
    else:
        probs_lambda.append(np.full(21, np.nan)) # 未收敛则用 nan 占位

probs_lambda = np.array(probs_lambda)

# 找出 lambda 最大时，出场率排名前 5 的卡组
top_5_indices_lam = np.argsort(probs_lambda[-1])[::-1][:5]

plt.figure(figsize=(10, 6))
for idx in top_5_indices_lam:
    plt.plot(lambdas, probs_lambda[:, idx], label=deck_names[idx], linewidth=2)
plt.xlabel('理性参数 (lambda)')
plt.ylabel('出场率')
plt.title('不同 lambda 下高出场率卡组的关系图')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

