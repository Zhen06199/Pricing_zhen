import numpy as np
import load_data as ld
import matplotlib.pyplot as plt

# 原始数据加载
filename_c = "customer_comparision.txt"
filename_p = "provider_comparison.txt"
demand, resource, budget, lamda, cost, supply = ld.load_customer(filename_c, filename_p)

buyer_params = lamda.T
buyer_demand = demand
buyer_budget = budget

seller_cost = cost
seller_supply = supply
num_sellers = len(seller_cost)

# === 新增：读取所有 Nash 策略价格的函数 ===
def load_nash_prices_by_customer_count(filename):
    nash_price_dict = {}
    current_prices = [0.0] * 5
    count = 0

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("ID"):
                sid, price, _ = line.split(",")
                current_prices[int(sid)] = float(price)
                count += 1
            elif line.startswith("# customer number:"):
                customer_count = int(line.split(":")[1].split(",")[0].strip())
                if count == 5:  # 确保读取了 5 个 provider 的数据
                    nash_price_dict[customer_count] = current_prices.copy()
                    current_prices = [0.0] * 5
                    count = 0
    return nash_price_dict

# === 加载 Nash 策略 ===
nash_price_table = load_nash_prices_by_customer_count("Pricing-strategy-pruning-4providers.txt")

# 利润计算函数
def compute_profit(price_test, prices_others, buyer_params, buyer_demand, buyer_budget, seller_supply, test_id):
    total_q = 0.0
    for param, d, b in zip(buyer_params, buyer_demand, buyer_budget):
        utilities = []
        masks = []
        for k in range(len(seller_supply)):
            price_k = price_test if k == test_id else prices_others[k]
            util = param[k] - price_k
            mask = price_k <= b
            utilities.append(util)
            masks.append(mask)

        exp_utils = [np.exp(u) * m for u, m in zip(utilities, masks)]
        denom = sum(exp_utils)
        prob = 0 if denom == 0 else exp_utils[test_id] / denom
        total_q += prob * d

    actual_q = min(total_q, seller_supply[test_id])
    return actual_q * (price_test - seller_cost[test_id])

# 要测试的 customer 数量
test_customer_counts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 85, 90, 100]
test_customer_total_demand = [sum(buyer_demand[:c]) for c in test_customer_counts]
providers_to_test = range(5)

results = {}

for test_id in providers_to_test:
    nash_profits = []
    fixed_profits = []
    random_best_profits = []
    random_avg_profits = []

    for idx, c_count in enumerate(test_customer_counts):
        bp = buyer_params[:c_count]
        bd = buyer_demand[:c_count]
        bb = buyer_budget[:c_count]

        # === 这里取对应 customer 数下的 Nash 策略 ===
        if c_count not in nash_price_table:
            print(f"[警告] 缺失 customer={c_count} 的 Nash 价格数据，跳过。")
            continue
        nash_prices = nash_price_table[c_count]
        price_nash = nash_prices[test_id]

        price_fixed = seller_cost[test_id] + 0.54
        prices_others = {k: nash_prices[k] for k in range(num_sellers) if k != test_id}

        profit_nash = compute_profit(price_nash, prices_others, bp, bd, bb, seller_supply, test_id)
        profit_fixed = compute_profit(price_fixed, prices_others, bp, bd, bb, seller_supply, test_id)

        # 随机定价策略
        price_min = seller_cost[test_id]
        price_max = max(bb)
        num_trials = 50
        rand_results = []
        for _ in range(num_trials):
            p = np.random.uniform(price_min, price_max)
            prof = compute_profit(p, prices_others, bp, bd, bb, seller_supply, test_id)
            rand_results.append((p, prof))

        best_random_price, best_random_profit = max(rand_results, key=lambda x: x[1])
        avg_random_profit = np.mean([x[1] for x in rand_results])

        nash_profits.append(profit_nash)
        fixed_profits.append(profit_fixed)
        random_best_profits.append(best_random_profit)
        random_avg_profits.append(avg_random_profit)

    results[test_id] = {
        "customers": test_customer_counts,
        "nash": nash_profits,
        "fixed": fixed_profits,
        "random_best": random_best_profits,
        "random_avg": random_avg_profits
    }


#####################画图##########################################
import matplotlib.pyplot as plt
import math

providers_to_plot = [0, 1, 2, 3, 4]
num_providers = len(providers_to_plot)

cols = 3
rows = math.ceil(num_providers / cols)

# 去掉 sharey=True，这样每个子图的 y 轴范围就不会统一
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()

x_labels = [f"<{c},{d}>" for c, d in zip(test_customer_counts, test_customer_total_demand)]

for i, test_id in enumerate(providers_to_plot):
    ax = axes[i]
    data = results[test_id]
    ax.grid(True)

    ax.plot(test_customer_total_demand, data["nash"], color='blue', marker='o', linestyle='-', label='Nash', linewidth=2)
    ax.plot(test_customer_total_demand, data["fixed"], color='green', marker='s', linestyle='--', label='Fixed', linewidth=2)
    ax.plot(test_customer_total_demand, data["random_avg"], color='purple', marker='x', linestyle=':', label='Random Avg', linewidth=2)

    ax.set_title(f"Provider {test_id}")
    ax.set_xlabel("Customer Info <Number, Total Demand>")
    if i % cols == 0:
        ax.set_ylabel("Profit / Euro")

    ax.legend(fontsize=8)
    ax.set_xticks(test_customer_total_demand)
    ax.set_xticklabels(x_labels, rotation=45, fontsize=7)
    ax.tick_params(axis='x', which='both', length=0)

# 删除多余的空图
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Profit Comparison Strategies by Provider", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
