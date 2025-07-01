import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 价格空间
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

# 需求和容量
demand = 29
capacity0 = 10
capacity1 = 19

q0 = np.minimum(prob0 * demand, capacity0)
q1 = np.minimum(prob1 * demand, capacity1)

profit0 = q0 * P0
profit1 = q1 * P1

# ===== Best Response 曲线 =====
# Seller 0 best response: 对每个 P1，找最优 P0
best_resp0 = [price0_vals[np.argmax(profit0[i, :])] for i in range(len(price1_vals))]
# Seller 1 best response: 对每个 P0，找最优 P1
best_resp1 = [price1_vals[np.argmax(profit1[:, j])] for j in range(len(price0_vals))]

# ===== 图 1：选择概率 - Seller 0 =====
fig1 = plt.figure(figsize=(7, 5))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(P0, P1, prob0, cmap='viridis', alpha=0.95, rstride=1, cstride=1)
ax1.set_title('Choice Probability - Seller 0')
ax1.set_xlabel('Price 0')
ax1.set_ylabel('Price 1')
ax1.set_zlabel('Probability')
ax1.grid(False)

# ===== 图 2：选择概率 - Seller 1 =====
fig2 = plt.figure(figsize=(7, 5))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(P0, P1, prob1, cmap='plasma', alpha=0.95, rstride=1, cstride=1)
ax2.set_title('Choice Probability - Seller 1')
ax2.set_xlabel('Price 0')
ax2.set_ylabel('Price 1')
ax2.set_zlabel('Probability')
ax2.grid(False)

# ===== 图 3：利润图 - Seller 0 + best response (黑线) =====
fig3 = plt.figure(figsize=(7, 5))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(P0, P1, profit0, cmap='Blues', alpha=0.95, rstride=1, cstride=1)
ax3.plot(best_resp0, price1_vals, profit0[np.arange(len(price1_vals)), [np.argmax(profit0[i, :]) for i in range(len(price1_vals))]],
         color='black', linewidth=2, label='Best Response 0')
ax3.set_title('Profit Surface - Seller 0')
ax3.set_xlabel('Price 0')
ax3.set_ylabel('Price 1')
ax3.set_zlabel('Profit')
ax3.grid(False)
ax3.legend()

# ===== 图 4：利润图 - Seller 1 + best response (黑线) =====
fig4 = plt.figure(figsize=(7, 5))
ax4 = fig4.add_subplot(111, projection='3d')
ax4.plot_surface(P0, P1, profit1, cmap='Oranges', alpha=0.95, rstride=1, cstride=1)
ax4.plot(price0_vals, best_resp1, profit1[[np.argmax(profit1[:, j]) for j in range(len(price0_vals))], np.arange(len(price0_vals))],
         color='black', linewidth=2, label='Best Response 1')
ax4.set_title('Profit Surface - Seller 1')
ax4.set_xlabel('Price 0')
ax4.set_ylabel('Price 1')
ax4.set_zlabel('Profit')
ax4.grid(False)
ax4.legend()

plt.show()
