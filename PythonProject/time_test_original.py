import time
import numpy as np
import load_data as ld
from itertools import product

def run_once():
    start_time = time.time()

    # Load data
    filename_c = "customers.txt"
    filename_p = "providers.txt"
    demand, resource, budget, lamda, cost, supply = ld.load_customer(filename_c, filename_p)

    buyer_params = lamda.T
    buyer_demand = demand
    total_demand = np.sum(buyer_demand)
    buyer_budget = budget

    seller_supply = supply
    seller_cost = cost
    num_sellers = len(seller_cost)

    price_vals = [
        np.linspace(c, max(buyer_budget), 20)
        for c in seller_cost
    ]

    price_grids = np.meshgrid(*price_vals, indexing='ij')
    grid_shape = price_grids[0].shape

    q_total = [np.zeros(grid_shape) for _ in range(num_sellers)]

    for param, demand, budget in zip(buyer_params, buyer_demand, buyer_budget):
        utilities = [param[k] - price_grids[k] for k in range(num_sellers)]
        masks = [(price_grids[k] <= budget).astype(float) for k in range(num_sellers)]
        exp_utils = [np.exp(u) * m for u, m in zip(utilities, masks)]

        denom = sum(exp_utils)
        probs = [np.where(denom == 0, 0, eu / denom) for eu in exp_utils]

        for k in range(num_sellers):
            q_total[k] += probs[k] * demand

    actual_q = [np.minimum(q_total[k], seller_supply[k]) for k in range(num_sellers)]
    profit = [actual_q[k] * price_grids[k] for k in range(num_sellers)]

    nash_points = []
    price_axes = [vals for vals in price_vals]

    for idx in np.ndindex(grid_shape):
        is_nash = True
        for k in range(num_sellers):
            fixed_idx = list(idx)
            best_response = None
            best_profit = -np.inf
            for i in range(len(price_axes[k])):
                test_idx = fixed_idx.copy()
                test_idx[k] = i
                profit_at_idx = profit[k][tuple(test_idx)]
                if profit_at_idx > best_profit:
                    best_profit = profit_at_idx
                    best_response = i
            if best_response != idx[k]:
                is_nash = False
                break

        if is_nash:
            prices = [price_axes[k][idx[k]] for k in range(num_sellers)]
            profits = [profit[k][idx] for k in range(num_sellers)]
            nash_points.append((prices, profits))

    end_time = time.time()
    elapsed = end_time - start_time

    return elapsed, nash_points


# ------------------------ 多次运行主函数 -----------------------------
num_runs = 5
all_times = []

for i in range(num_runs):
    print(f"\n--- Run {i+1} ---")
    elapsed, nash_pts = run_once()
    all_times.append(elapsed)
    print(f"⏱️ 运行时间：{elapsed:.4f} 秒")
    if nash_pts:
        print(f"找到 {len(nash_pts)} 个纳什点")
    else:
        print("⚠️ 未找到纳什均衡点")

# ------------------------ 输出统计信息 -----------------------------
print("\n=== 运行时间统计 ===")
print("每次运行时间（秒）：", [f"{t:.4f}" for t in all_times])
print(f"平均时间：{np.mean(all_times):.4f} 秒")
print(f"最大时间：{np.max(all_times):.4f} 秒")
print(f"最小时间：{np.min(all_times):.4f} 秒")
result_file = "results-5.txt"

for i in range(num_runs):
    print(f"\n--- Run {i+1} ---")
    elapsed, nash_pts = run_once()
    all_times.append(elapsed)

    with open(result_file, "a") as f:
        if nash_pts:
            for idx, (prices, profits) in enumerate(nash_pts):
                for seller_id, (p, pr) in enumerate(zip(prices, profits)):
                    f.write(f"{i+1},{seller_id},{p:.4f},{pr:.4f},{elapsed:.4f}\n")
        else:
            f.write(f"{i+1},-1,-1,-1,{elapsed:.4f}\n")  # 表示未找到 Nash 均衡
