import matplotlib.pyplot as plt
import re

# 设置字体加粗
plt.rcParams.update({
    "font.size": 12,
    "axes.labelweight": "bold",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11
})

# 读取 runtime_log.txt
with open("runtime_log.txt", "r", encoding="utf-8") as f:
    log_content = f.read()

# 提取数据函数
def extract_all_runtime_data(log_text):
    blocks = re.findall(r"(#+\s*K=(\d+)[^#]*#+)(.*?)(?=#+\s*K=|\Z)", log_text, re.DOTALL)
    data_by_k = {}
    for header, k_value, content in blocks:
        customers = [int(c) for c in re.findall(r"Customers:\s*(\d+)", content)]
        runtimes = [float(r) for r in re.findall(r"Runtime:\s*([\d.]+)", content)]
        k = int(k_value)
        if customers and runtimes:
            data_by_k[k] = (customers, runtimes)
    return data_by_k

# 获取数据
data_dict = extract_all_runtime_data(log_content)

# 选择前两个 K 值
selected_ks = sorted(data_dict.keys())[:2]

# 创建两个独立 Y 轴的子图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, k in zip(axes, selected_ks):
    customers, runtimes = data_dict[k]
    ax.plot(customers, runtimes, marker='o', color='blue', linewidth=2, markersize=6, label=f"K = {k}")
    ax.set_title(f"Runtime vs Customers (K={k})", fontweight='bold')
    ax.set_xlabel("Number of Customers", fontweight='bold')
    ax.set_ylabel("Runtime (seconds)", fontweight='bold')
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig("runtime_subplots_independent_y.png", dpi=300)
plt.show()

print("✅ Saved as 'runtime_subplots_independent_y.png'")
