#############区别就是加了个budet，因为budget不同

import numpy as np
import matplotlib.pyplot as plt

# 固定 5 个 buyer 的偏好参数 (a0, a1)、需求量和预算（固定值）
buyer_params = np.array([
    [2.5, 2.2],
    [2.0, 2.8],
    [1.8, 2.5],
    [2.9, 2.0],
    [2.4, 1.9]
])
buyer_demand = np.array([100, 100, 150, 50, 200])  # 需求量
buyer_budget = np.array([2.3, 2.5, 2.4, 2.2, 2.45])  # 预算，固定值

# 固定库存
capacity0 = 300
capacity1 = 300
# 离散价格空间
price0_vals = np.linspace(1.9, 2.5, 100)
price1_vals = np.linspace(1.7, 2.5, 100)
P0, P1 = np.meshgrid(price0_vals, price1_vals)

# 初始化销量
q0_total = np.zeros_like(P0)
q1_total = np.zeros_like(P1)

# 累计每个 buyer 的选择概率 × 需求量
for (a0, a1), demand, budget in zip(buyer_params, buyer_demand, buyer_budget):
    # 对价格和预算做广播比较，价格大于预算则该选择概率为0
    mask0 = (P0 <= budget).astype(float)  # 价格在预算内才可能买
    mask1 = (P1 <= budget).astype(float)

    U0 = a0 - P0
    U1 = a1 - P1

    exp_U0 = np.exp(U0) * mask0
    exp_U1 = np.exp(U1) * mask1
    denom = exp_U0 + exp_U1

    # 注意：如果 denom=0（即两个价格都超预算），设概率为0
    prob0 = np.where(denom == 0, 0, exp_U0 / denom)
    prob1 = np.where(denom == 0, 0, exp_U1 / denom)

    q0_total += prob0 * demand
    q1_total += prob1 * demand

# 应用库存限制
actual_q0 = np.minimum(q0_total, capacity0)
actual_q1 = np.minimum(q1_total, capacity1)

# 利润计算
profit0 = actual_q0 * P0
profit1 = actual_q1 * P1

# 查找纯策略纳什均衡点
nash_points = []
for i in range(len(price1_vals)):
    for j in range(len(price0_vals)):
        best_col = np.argmax(profit0[i, :])  # seller0 最优价格响应（fix seller1 price）
        best_row = np.argmax(profit1[:, j])  # seller1 最优价格响应（fix seller0 price）
        if best_col == j and best_row == i:
            nash_points.append((price0_vals[j], price1_vals[i], profit0[i, j], profit1[i, j]))

# 输出纳什均衡点信息
if nash_points:
    print("The pure strategy Nash equilibrium points (price and profit) found are:")
    for idx, (p0, p1, π0, π1) in enumerate(nash_points, 1):
        print(f"纳什均衡 {idx}:")
        print(f"  Seller0  Price = {p0:.3f}, Revenue = {π0:.3f}")
        print(f"  Seller1  Price= {p1:.3f}, Revenue = {π1:.3f}")
else:
    print("未找到纯策略纳什均衡")



# === Best Response 数据计算 ===
best_response0_x = []
best_response0_y = []
for i in range(len(price1_vals)):
    j_best = np.argmax(profit0[i, :])
    best_response0_x.append(price0_vals[j_best])
    best_response0_y.append(price1_vals[i])

best_response1_x = []
best_response1_y = []
for j in range(len(price0_vals)):
    i_best = np.argmax(profit1[:, j])
    best_response1_x.append(price0_vals[j])
    best_response1_y.append(price1_vals[i_best])

# === 开始绘图：四个子图 ===
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 销量图：Seller 0
c0 = axs[0, 0].imshow(actual_q0, origin='lower', aspect='auto',
                     extent=[price0_vals[0], price0_vals[-1], price1_vals[0], price1_vals[-1]], cmap='Blues')
axs[0, 0].set_title('Sales of Seller 0')
axs[0, 0].set_xlabel('Price 0')
axs[0, 0].set_ylabel('Price 1')
fig.colorbar(c0, ax=axs[0, 0])

# 销量图：Seller 1
c1 = axs[0, 1].imshow(actual_q1, origin='lower', aspect='auto',
                     extent=[price0_vals[0], price0_vals[-1], price1_vals[0], price1_vals[-1]], cmap='Blues')
axs[0, 1].set_title('Sales of Seller 1')
axs[0, 1].set_xlabel('Price 0')
axs[0, 1].set_ylabel('Price 1')
fig.colorbar(c1, ax=axs[0, 1])

# 利润图：Seller 0
c2 = axs[1, 0].imshow(profit0, origin='lower', aspect='auto',
                     extent=[price0_vals[0], price0_vals[-1], price1_vals[0], price1_vals[-1]], cmap='Blues')
axs[1, 0].plot(best_response0_x, best_response0_y, color='black', linewidth=1.5, label='Seller 0 Best Response')
axs[1, 0].set_title(f'Profit of Seller 0 (Capacity={capacity0})')
axs[1, 0].set_xlabel('Price 0')
axs[1, 0].set_ylabel('Price 1')
fig.colorbar(c2, ax=axs[1, 0])

# 利润图：Seller 1
c3 = axs[1, 1].imshow(profit1, origin='lower', aspect='auto',
                     extent=[price0_vals[0], price0_vals[-1], price1_vals[0], price1_vals[-1]], cmap='Blues')
axs[1, 1].plot(best_response1_x, best_response1_y, color='black', linewidth=1.5, label='Seller 1 Best Response')
axs[1, 1].set_title(f'Profit of Seller 1 (Capacity={capacity1})')
axs[1, 1].set_xlabel('Price 0')
axs[1, 1].set_ylabel('Price 1')
fig.colorbar(c3, ax=axs[1, 1])

# 纳什点绘制（红点）
for (p0, p1, _, _) in nash_points:
    for ax in axs.flatten():
        ax.scatter(p0, p1, color='red', s=30, label='Nash Equilibrium')

# 避免重复图例
for ax in axs.flatten():
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys())

plt.tight_layout()
plt.show()
