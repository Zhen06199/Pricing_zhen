import numpy as np
import matplotlib.pyplot as plt

# 离散价格空间
price0_vals = np.linspace(1.9, 2.5, 100)
price1_vals = np.linspace(1.7, 2.5, 100)
P0, P1 = np.meshgrid(price0_vals, price1_vals)

# 效用与 Logit 概率
U0 = 1 - P0
U1 = 1.75 - P1
exp_U0 = np.exp(U0)
exp_U1 = np.exp(U1)
sum_exp = exp_U0 + exp_U1
prob0 = exp_U0 / sum_exp
prob1 = exp_U1 / sum_exp

# 总需求 & 库存限制
demand = 3
capacity0 = 10
capacity1 = 20

# 实际销售数量
q0 = np.minimum(prob0 * demand, capacity0)
q1 = np.minimum(prob1 * demand, capacity1)

# 利润计算
profit0 = q0 * P0
profit1 = q1 * P1

# 最佳响应（每行/列的最优策略）
best_response0 = price0_vals[np.argmax(profit0, axis=1)]  # seller 0 针对 price1
best_response1 = price1_vals[np.argmax(profit1, axis=0)]  # seller 1 针对 price0

# 纳什均衡搜索
nash_points = []
for i in range(len(price1_vals)):
    for j in range(len(price0_vals)):
        if j == np.argmax(profit0[i, :]) and i == np.argmax(profit1[:, j]):
            nash_points.append((price0_vals[j], price1_vals[i]))


# 输出纳什均衡点及利润
if nash_points:
    print("找到的纯策略纳什均衡点及对应利润：\n")
    for p0, p1 in nash_points:
        # 找到对应索引
        i = np.argmin(np.abs(price1_vals - p1))
        j = np.argmin(np.abs(price0_vals - p0))

        prof0 = profit0[i, j]
        prof1 = profit1[i, j]

        print(f"价格组合: Price0 = {p0:.3f}, Price1 = {p1:.3f} | 利润: Seller0 = {prof0:.3f}, Seller1 = {prof1:.3f}")
else:
    print("没有找到纯策略纳什均衡。")



# 可视化：四个图（实际销量 + 利润）
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# q0 图（实际销售量）
c0 = axs[0, 0].imshow(q0, extent=[1.9, 2.5, 1.7, 2.5], origin='lower', aspect='auto', cmap='Blues')
axs[0, 0].set_title('Expected Sales of Seller 0')
axs[0, 0].set_xlabel('Price 0')
axs[0, 0].set_ylabel('Price 1')

axs[0, 0].legend()
fig.colorbar(c0, ax=axs[0, 0])

# q1 图（实际销售量）
c1 = axs[0, 1].imshow(q1, extent=[1.9, 2.5, 1.7, 2.5], origin='lower', aspect='auto', cmap='Blues')
axs[0, 1].set_title('Expected Sales of Seller 1')
axs[0, 1].set_xlabel('Price 0')
axs[0, 1].set_ylabel('Price 1')

axs[0, 1].legend()
fig.colorbar(c1, ax=axs[0, 1])

# Profit 0
c2 = axs[1, 0].imshow(profit0, extent=[1.9, 2.5, 1.7, 2.5], origin='lower', aspect='auto', cmap='Blues')
axs[1, 0].set_title('Profit of Seller 0')
axs[1, 0].set_xlabel('Price 0')
axs[1, 0].set_ylabel('Price 1')
axs[1, 0].scatter(best_response0, price1_vals, color='grey', s=20)
fig.colorbar(c2, ax=axs[1, 0])

# Profit 1
c3 = axs[1, 1].imshow(profit1, extent=[1.9, 2.5, 1.7, 2.5], origin='lower', aspect='auto', cmap='Blues')
axs[1, 1].set_title('Profit of Seller 1')
axs[1, 1].set_xlabel('Price 0')
axs[1, 1].set_ylabel('Price 1')
axs[1, 1].scatter(price0_vals, best_response1, color='grey', s=20)
fig.colorbar(c3, ax=axs[1, 1])

# 纳什点标记
for (px, py) in nash_points:
    for ax in axs.flatten():
        ax.scatter(px, py, color='red', edgecolor='red', s=30, label='Nash')

# 避免重复图例
for ax in axs.flatten():
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

plt.tight_layout()
plt.show()
