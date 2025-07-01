import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()

# 5个buyer偏好 (a0, a1, a2)，需求量，预算
buyer_params = np.array([
    [2.5, 2.2, 3.0],
    [2.0, 2.8, 3.1],
    [1.8, 2.5, 2.3],
    [2.9, 2.0, 2.0],
    [2.4, 1.9, 3.0],
    [3, 2, 4],
])
buyer_demand = np.array([10, 12, 15, 20, 20.20])
buyer_budget = np.array([2.3, 2.5, 2.4, 2.2, 2.45,2.45])

# 三家公司库存
capacity = np.array([50, 50, 30])

# 离散价格空间 (为了演示维度减少，这里用20个点)
price_vals = np.linspace(1.7, 2.5, 25)
P0, P1, P2 = np.meshgrid(price_vals, price_vals, price_vals)

# 初始化销量
q_total = np.zeros((3,) + P0.shape)

for (a0, a1, a2), demand, budget in zip(buyer_params, buyer_demand, buyer_budget):
    # 价格矩阵与预算比较（广播），超预算概率置0
    mask0 = (P0 <= budget).astype(float)
    mask1 = (P1 <= budget).astype(float)
    mask2 = (P2 <= budget).astype(float)

    U0 = a0 - P0
    U1 = a1 - P1
    U2 = a2 - P2

    exp_U0 = np.exp(U0) * mask0
    exp_U1 = np.exp(U1) * mask1
    exp_U2 = np.exp(U2) * mask2

    denom = exp_U0 + exp_U1 + exp_U2
    denom[denom == 0] = 1  # 防止除零

    prob0 = exp_U0 / denom
    prob1 = exp_U1 / denom
    prob2 = exp_U2 / denom

    q_total[0] += prob0 * demand
    q_total[1] += prob1 * demand
    q_total[2] += prob2 * demand

# 应用库存限制
actual_q = np.minimum(q_total, capacity[:, None, None, None])

# 利润计算
profit = actual_q * np.array([P0, P1, P2])

# 找纯策略纳什均衡（暴力搜索）
nash_points = []
len_p = len(price_vals)

for i in range(len_p):
    for j in range(len_p):
        for k in range(len_p):
            # seller0对固定p1,p2的最佳p0位置
            best_p0 = np.argmax(profit[0, i, :, k])
            # seller1对固定p0,p2的最佳p1位置
            best_p1 = np.argmax(profit[1, :, j, k])
            # seller2对固定p0,p1的最佳p2位置
            best_p2 = np.argmax(profit[2, :, j, k])
            if best_p0 == j and best_p1 == i and best_p2 == k:
                nash_points.append((price_vals[j], price_vals[i], price_vals[k],
                                    profit[0, i, j, k], profit[1, i, j, k], profit[2, i, j, k]))

if nash_points:
    print("Pure strategy Nash equilibrium points (price and profit) found：")
    for idx, (p0, p1, p2, π0, π1, π2) in enumerate(nash_points, 1):
        print(f"Nash equilibrium {idx}:")
        print(f"  Seller0 Price = {p0:.3f}, Revenue = {π0:.3f}")
        print(f"  Seller1 Price = {p1:.3f}, Revenue = {π1:.3f}")
        print(f"  Seller2 Price = {p2:.3f}, Revenue = {π2:.3f}")
else:
    print("No pure strategy Nash equilibrium found")



end_time = time.time()  # 记录结束时间

elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.4f} 秒")

from mpl_toolkits.mplot3d import Axes3D  # 3D画图

# 假设你已经有了nash_points列表，格式为：
# (price0, price1, price2, profit0, profit1, profit2)

if nash_points:
    # 3D散点图表示所有纳什均衡点价格组合
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    p0_vals = [p[0] for p in nash_points]
    p1_vals = [p[1] for p in nash_points]
    p2_vals = [p[2] for p in nash_points]
    ax.scatter(p0_vals, p1_vals, p2_vals, c='r', marker='o', s=50)
    ax.set_xlabel('Price 0')
    ax.set_ylabel('Price 1')
    ax.set_zlabel('Price 2')
    ax.set_title('Nash Equilibria Price Points (3 Sellers)')
    plt.show()

    # 固定price2为第一个纳什点的price2，画price0-price1利润等高线及均衡点投影
    fixed_p2 = nash_points[0][2]
    idx_fixed_p2 = np.argmin(np.abs(price_vals - fixed_p2))  # 找最接近索引

    # 利润二维切片
    profit0_slice = profit[0, :, :, idx_fixed_p2]
    profit1_slice = profit[1, :, :, idx_fixed_p2]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    c0 = axs[0].contourf(price_vals, price_vals, profit0_slice, levels=20, cmap='plasma')
    fig.colorbar(c0, ax=axs[0])
    axs[0].set_title(f'Seller 0 Profit at Price2={fixed_p2:.2f}')
    axs[0].set_xlabel('Price 0')
    axs[0].set_ylabel('Price 1')

    c1 = axs[1].contourf(price_vals, price_vals, profit1_slice, levels=20, cmap='inferno')
    fig.colorbar(c1, ax=axs[1])
    axs[1].set_title(f'Seller 1 Profit at Price2={fixed_p2:.2f}')
    axs[1].set_xlabel('Price 0')
    axs[1].set_ylabel('Price 1')

    # 标出纳什均衡点投影
    for p0, p1, p2, _, _, _ in nash_points:
        if np.isclose(p2, fixed_p2, atol=0.05):
            axs[0].scatter(p0, p1, color='red', marker='o')
            axs[1].scatter(p0, p1, color='red', marker='o')

    plt.tight_layout()
    plt.show()
else:
    print("There is no Nash equilibrium point to draw")
