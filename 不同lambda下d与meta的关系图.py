import numpy as np
from scipy.optimize import root
from itertools import combinations
import matplotlib.pyplot as plt

# 字体设置，确保论文图表中的中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

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
    return x - softmax(lam * expected_payoffs)

# --- 实验绘图函数 ---
def plot_robustness(lam_val):
    ds = np.linspace(-10, 10, 100) # 更高分辨率的 d
    probs_d = []
    
    # 初始化一个平均分布作为起点
    current_x0 = np.ones(21) / 21
    
    for d in ds:
        B_pert = B_base.copy()
        # 维持零和博弈特征：一加一减
        B_pert[3, 5] += d
        B_pert[5, 3] -= d
        A_pert = get_A(B_pert)
        
        # 使用上一步的解作为当前步的初值 (顺序热启动，保证曲线平滑不迷路)
        sol = root(qre_fixed_point, current_x0, args=(A_pert, lam_val), method='hybr')
        
        if sol.success:
            p = np.clip(sol.x, 0, None)
            p /= p.sum()
            probs_d.append(p)
            current_x0 = p # 更新热启动锚点
        else:
            probs_d.append(np.full(21, np.nan))

    probs_d = np.array(probs_d)
    
    # 提取在 d=0 附近出场率最高的5个卡组进行绘制
    mid_idx = len(ds) // 2
    top_5_indices = np.argsort(probs_d[mid_idx])[::-1][:5]

    plt.figure(figsize=(10, 6), dpi=150)
    for idx in top_5_indices:
        plt.plot(ds, probs_d[:, idx], label=deck_names[idx], linewidth=2.5)
        
    plt.xlabel('对局收益扰动 d (绿 vs 黑)', fontsize=12)
    plt.ylabel('理论出场率', fontsize=12)
    plt.title(f'QRE 模型下的天梯生态鲁棒性测试 ($\lambda$ = {lam_val})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'QRE_robustness_lambda_{lam_val}.png')
    plt.show()

# 分别绘制高、低两种理性参数下的环境演化图以作对比
plot_robustness(lam_val=20)