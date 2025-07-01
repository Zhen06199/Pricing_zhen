import numpy as np
import matplotlib.pyplot as plt

# 定义 Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义修改过的 Sigmoid 变体函数
def custom_sigmoid(x):
    return 0.2 * np.tanh(20 * x)

# 生成 x 轴数据
x = np.linspace(-0.2, 0.2, 300)

# 计算两种函数的 y 值
y_custom_sigmoid = custom_sigmoid(x)
y_sigmoid = 0.4 * sigmoid(40 * x) - 0.2  # 转换后的标准 Sigmoid

# 绘制两个函数的图像
plt.plot(x, y_custom_sigmoid, label=r'$f(x) = 0.2 \cdot \tanh(20x)$', color='blue')
plt.plot(x, y_sigmoid, label=r'$f(x) = 0.4 \cdot \sigma(40x) - 0.2$', color='orange', linestyle='--')

# 绘制辅助线（上下边界和 x=±0.2）
plt.axhline(0.2, linestyle="--", color="red", label="Upper Bound: 0.2")
plt.axhline(-0.2, linestyle="--", color="green", label="Lower Bound: -0.2")
plt.axvline(0.2, linestyle=":", color="black", label="x=±0.2")
plt.axvline(-0.2, linestyle=":", color="black")

# 图形标题和图例
plt.title("Comparison of Modified Sigmoid Function and Standard Sigmoid")
plt.legend()

# 显示图像
plt.show()
