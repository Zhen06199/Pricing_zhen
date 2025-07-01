import numpy as np
import matplotlib.pyplot as plt

# 效用差的范围
delta_V = np.linspace(-20, 20, 300)

# 敏感度参数 λ（控制决策的“果断程度”）
lambdas = [0.5, 1, 2]

plt.figure(figsize=(8, 5))

# 分别画出不同 λ 值下的选择概率曲线
for lam in lambdas:
    P_A = np.exp(lam * delta_V) / (np.exp(lam * delta_V) + 1)
    plt.plot(delta_V, P_A)

plt.title('λ ')
plt.xlabel('V = V_A - V_B')
plt.ylabel('P(A)')
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.axhline(0.5, color='gray', linestyle='--', linewidth=1)
plt.legend()
plt.grid(True)
plt.show()
