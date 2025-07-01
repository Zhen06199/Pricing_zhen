import numpy as np
import matplotlib.pyplot as plt

# 支持区间
a, b = 1.0, 3.0
c = 0.5  # 成本

# 离散点用于展示整个策略空间（支持区间外也展示）
x_full = np.linspace(0.5, 3.5, 300)

# 假设混合策略CDF F(x) 只在[a,b]支持，且满足无差异条件
# 简单用线性CDF，表示均匀分布混合策略
def F(x):
    if x < a:
        return 0
    elif x > b:
        return 1
    else:
        return (x - a) / (b - a)

# 向量化
F_vec = np.vectorize(F)

# 计算期望收益 U(x) = (x-c)*(1 - F(x))
U = (x_full - c) * (1 - F_vec(x_full))

# 支持区间内的收益（无差异，理应是常数，这里近似）
U_in_support = (x_full[(x_full>=a) & (x_full<=b)] - c) * (1 - F_vec(x_full[(x_full>=a) & (x_full<=b)]))
mean_U_in = np.mean(U_in_support)

# 画图
plt.figure(figsize=(8,5))
plt.plot(x_full, U, label='Payoff $U(x)$')
plt.axvline(a, color='gray', linestyle='--', label='Support interval start')
plt.axvline(b, color='gray', linestyle='--', label='Support interval end')
plt.hlines(mean_U_in, x_full[0], x_full[-1], colors='red', linestyles='dashed', label='Constant payoff in support')
plt.fill_between(x_full, mean_U_in, U, where=(U < mean_U_in), color='red', alpha=0.2)

plt.xlabel('Strategy $x$')
plt.ylabel('Expected Payoff $U(x)$')
plt.title('Demonstration of the No-Profitable-Deviation Condition')
plt.legend()
plt.grid(True)
plt.show()

print(f"Mean payoff in support interval [{a}, {b}]: {mean_U_in:.4f}")
print(f"Max payoff outside support interval: {max(U[(x_full < a) | (x_full > b)]) :.4f}")
