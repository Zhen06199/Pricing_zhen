import matplotlib.pyplot as plt
import re

# 读取文件内容
with open("Pricing-strategy-pruning-4providers.txt", "r", encoding="utf-8") as f:
    data = f.read()

# 提取每组 customer number 和 demand 作为横坐标标签
header_blocks = re.findall(r'# customer number: (\d+), customer demand: (\d+)', data)
blocks = re.split(r'# customer number: \d+, customer demand: \d+\n# provider number: \d+, customer demand: \d+\n# Running time: [\d.]+ seconds\n', data.strip())

provider_profits = {i: [] for i in range(5)}  # 初始化 5 个 provider
x_labels = []

# 从第二个 block 开始，第一个是文件前面的杂项
for header, block in zip(header_blocks, blocks[1:]):
    customer_number, customer_demand = header
    x_labels.append(f"<{customer_number},{customer_demand}>")

    lines = block.strip().splitlines()
    for line in lines:
        if line.startswith("#") or line.strip() == "":
            continue
        pid, _, profit = line.strip().split(',')
        provider_profits[int(pid)].append(float(profit))

# 画图
plt.figure(figsize=(16, 6))
colors = ['b', 'g', 'r', 'c', 'm']
for pid in range(4):
    plt.plot(x_labels, provider_profits[pid], label=f'Provider {pid}', marker='o', color=colors[pid])

plt.xlabel("Customer Info (<number,demand>)")
plt.ylabel("Profit")
plt.title("Profit of Each Provider vs Customer Demand")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
