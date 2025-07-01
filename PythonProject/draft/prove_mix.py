import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 支持区间
a, b = 1.0, 3.0
c = 0.5  # 成本

# 离散点数
n = 100
x = np.linspace(a, b, n)

# 定义混合策略分布的参数：离散点上的CDF值 F(x_i)
# 约束：单调递增，F(a)=0, F(b)=1

# 初始猜测：线性增长
F_init = np.linspace(0, 1, n)

def payoff(F):
    # 计算期望收益
    U = (x - c) * (1 - F)
    return U

def objective(F):
    U = payoff(F)
    # 目标：让收益在区间尽量平坦，最小化收益的方差
    return np.var(U)

# 约束条件
constraints = [
    {'type': 'eq', 'fun': lambda F: F[0]},        # F(a) = 0
    {'type': 'eq', 'fun': lambda F: F[-1] - 1},   # F(b) = 1
]

# 单调递增约束，保证CDF有效
def monotone(F):
    return np.diff(F)  # 差值>=0

constraints.append({'type':'ineq', 'fun': monotone})

# 优化
result = minimize(objective, F_init, constraints=constraints, method='SLSQP')

F_opt = result.x
U_opt = payoff(F_opt)

# 画图
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(x, F_opt, label='Optimized CDF $F(x)$')
plt.title('Mixed Strategy Distribution')
plt.xlabel('Strategy $x$')
plt.ylabel('$F(x)$')
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.plot(x, U_opt, label='Payoff $(x - c)(1 - F(x))$', color='orange')
plt.hlines(np.mean(U_opt), a, b, colors='red', linestyles='dashed', label='Constant Payoff')
plt.title('Expected Payoff under Mixed Strategy')
plt.xlabel('Strategy $x$')
plt.ylabel('Payoff')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
