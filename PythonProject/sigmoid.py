import numpy as np
import matplotlib.pyplot as plt

# 定义 Sigmoid 变体函数
def custom_sigmoid(x):
    return 0.2 * np.tanh(20 * x)

# 生成 x 轴数据
x = np.linspace(-0.2, 0.2, 300)
y = custom_sigmoid(x)

# 绘制图像
plt.plot(x, y, label=r'$f(x) = 0.2 \cdot \tanh(2x)$', color='blue')
plt.axhline(0.2, linestyle="--", color="red", label="Upper Bound: 0.2")
plt.axhline(-0.2, linestyle="--", color="green", label="Lower Bound: -0.2")
plt.axvline(0.2, linestyle=":", color="black", label="x=±1")
plt.axvline(-0.2, linestyle=":", color="black")
plt.title("Modified Sigmoid Function")
plt.legend()
plt.show()
