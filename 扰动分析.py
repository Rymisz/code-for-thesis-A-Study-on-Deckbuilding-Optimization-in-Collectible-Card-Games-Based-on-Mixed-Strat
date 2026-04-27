import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from itertools import combinations

# 设置中文字体 (根据你的系统可能需要调整，如 'SimHei' 或 'Arial Unicode MS')
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def run_sensitivity_analysis():
    factions = ["红", "黄", "蓝", "绿", "紫", "黑", "白"]
    n_fac = len(factions)
    deck_combos = list(combinations(range(n_fac), 2))
    deck_names = [f"{factions[i]}{factions[j]}" for i, j in deck_combos]
    
    # 基础特征矩阵 V
    V = np.zeros((n_fac, len(deck_combos)))
    for idx, (i, j) in enumerate(deck_combos):
        V[i, idx] = 0.5
        V[j, idx] = 0.5
        
    # 设定的微扰区间 [-4, 4]，取 40 个采样点
    deltas = np.linspace(-4, 4, 5000)
    meta_history = {name: [] for name in ["蓝紫", "蓝白", "黑白", "绿白", "紫白"]}
    
    for d in deltas:
        # 重构微扰后的矩阵 B
        B_temp = np.array([
            [ 0, -4, -2,  6,  4,  1, -4],
            [ 4,  0, -2, -2, -4, -1,  4],
            [ 2,  2,  0,  4,  4, -4, -2],
            [-6,  2, -4,  0, -4,  8+d, 4],  # 绿克黑 加入扰动 d
            [-4,  4, -4,  4,  0, -2,  4],
            [-1,  1,  4, -(8+d), 2,  0, -4], # 黑被绿克 同步扰动
            [ 4, -4,  2, -4, -4,  4,  0]
        ])
        
        # 生成新环境矩阵并求解 MSNE
        A_temp = V.T @ B_temp @ V
        n = A_temp.shape[0]
        c = np.zeros(n + 1)
        c[-1] = -1
        A_ub = np.hstack((-A_temp.T, np.ones((n, 1))))
        b_ub = np.zeros(n)
        A_eq = np.ones((1, n + 1))
        A_eq[0, -1] = 0
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.array([1]), 
                      bounds=[(0, None)] * n + [(None, None)], method='highs')
        
        if res.success:
            probs = res.x[:-1]
            # 记录 5 套核心卡组在当前扰动下的出场率
            for name in meta_history.keys():
                idx = deck_names.index(name)
                meta_history[name].append(probs[idx] * 100) # 转为百分比
                
    # 绘制灵敏度曲线图
    plt.figure(figsize=(10, 6))
    for name, history in meta_history.items():
        plt.plot(deltas, history, label=name, linewidth=2)
        
    plt.axvline(x=0, color='grey', linestyle='--', label='基准模型假设点')
    plt.title("绿黑阵营克制系数扰动对 MSNE 稳态分布的灵敏度分析", fontsize=14)
    plt.xlabel("扰动变量 d (基础值 $b_{46}=8$)", fontsize=12)
    plt.ylabel("天梯稳态出场概率 (%)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 运行分析
run_sensitivity_analysis()