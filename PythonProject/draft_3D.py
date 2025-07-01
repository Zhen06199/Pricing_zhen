import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Buyer 参数
buyer_params = np.array([
    [2.5, 2.2],
    [2.0, 2.8],
    [1.8, 2.5],
    [2.9, 2.0],
    [2.4, 1.9]
])
buyer_demand = np.array([10, 10, 15, 18, 20])
buyer_budget = np.array([2.3, 2.5, 2.4, 2.2, 2.45])

capacity0 = 40
capacity1 = 40

price0_vals = np.linspace(1.9, 2.5, 100)
price1_vals = np.linspace(1.7, 2.5, 100)
P0, P1 = np.meshgrid(price0_vals, price1_vals)

q0_total = np.zeros_like(P0)
q1_total = np.zeros_like(P1)

for (a0, a1), demand, budget in zip(buyer_params, buyer_demand, buyer_budget):
    mask0 = (P0 <= budget).astype(float)
    mask1 = (P1 <= budget).astype(float)

    U0 = a0 - P0
    U1 = a1 - P1

    exp_U0 = np.exp(U0) * mask0
    exp_U1 = np.exp(U1) * mask1
    denom = exp_U0 + exp_U1

    prob0 = np.where(denom == 0, 0, exp_U0 / denom)
    prob1 = np.where(denom == 0, 0, exp_U1 / denom)

    q0_total += prob0 * demand
    q1_total += prob1 * demand

actual_q0 = np.minimum(q0_total, capacity0)
actual_q1 = np.minimum(q1_total, capacity1)

profit0 = actual_q0 * P0
profit1 = actual_q1 * P1

# 纳什均衡查找
nash_points = []
for i in range(len(price1_vals)):
    for j in range(len(price0_vals)):
        best_col = np.argmax(profit0[i, :])
        best_row = np.argmax(profit1[:, j])
        if best_col == j and best_row == i:
            nash_points.append((price0_vals[j], price1_vals[i], profit0[i, j], profit1[i, j]))

# 输出结果
if nash_points:
    print("找到的纯策略纳什均衡点：")
    for idx, (p0, p1, π0, π1) in enumerate(nash_points, 1):
        print(f"纳什均衡 {idx}: 公司0价格 = {p0:.3f}, 利润 = {π0:.3f} | 公司1价格 = {p1:.3f}, 利润 = {π1:.3f}")
else:
    print("未找到纯策略纳什均衡。")

# ===== 更高对比度的 3D 可视化 =====
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

# Seller 0 的利润面（高亮色）
surf0 = ax.plot_surface(P0, P1, profit0, cmap='coolwarm', alpha=0.85, edgecolor='k', linewidth=0.2)

# Seller 1 的利润面（对比色）
surf1 = ax.plot_surface(P0, P1, profit1, cmap='cividis', alpha=0.45, edgecolor='k', linewidth=0.2)

# 纳什点绘制
if nash_points:
    for p0, p1, π0, π1 in nash_points:
        ax.scatter(p0, p1, π0, color='red', s=70, edgecolor='black', linewidth=0.8, label='Nash (Seller 0)')
        ax.scatter(p0, p1, π1, color='blue', s=70, edgecolor='black', linewidth=0.8, label='Nash (Seller 1)')

# 标签与图例
ax.set_title('3D Profit Surfaces of Seller 0 & 1 (with Nash Equilibrium)', fontsize=13)
ax.set_xlabel('Price 0')
ax.set_ylabel('Price 1')
ax.set_zlabel('Profit')

# 避免重复图例项
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys())

plt.tight_layout()
plt.show()
