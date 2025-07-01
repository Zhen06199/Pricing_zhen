import numpy as np
import load_data as ld
import time

# ==== Set the seller ID to test ====
test_id = 2  # 可改成任意 seller，比如 1、2、3...

# ==== Load data ====
filename_c = "customer_comparision.txt"
filename_p = "provider_comparison.txt"
demand, resource, budget, lamda, cost, supply = ld.load_customer(filename_c, filename_p)

buyer_params = lamda.T
buyer_demand = demand
buyer_budget = budget

seller_cost = cost
seller_supply = supply
num_sellers = len(seller_cost)

# ==== Load Nash equilibrium prices ====
nash_prices = [0.0] * num_sellers
with open("Pricing-strategy.txt", "r") as f:
    for line in f:
        if line.startswith("#") or line.startswith("ID"):
            continue
        parts = line.strip().split(",")
        sid, price = int(parts[0]), float(parts[1])
        nash_prices[sid] = price

# ==== Define function to compute profit for test seller ====
def compute_profit(price_test, prices_others):
    total_q = 0.0
    for param, d, b in zip(buyer_params, buyer_demand, buyer_budget):
        utilities = []
        masks = []

        for k in range(num_sellers):
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
    return actual_q * price_test

# ==== Prepare test prices ====
price_nash = nash_prices[test_id]
price_fixed = seller_cost[test_id] + 0.5            #cost based
prices_others = {k: nash_prices[k] for k in range(num_sellers) if k != test_id}


# Nash Strategy
profit_nash = compute_profit(price_nash, prices_others)

# Fixed Strategy
profit_fixed = compute_profit(price_fixed, prices_others)

# ==== 构造 price_vals 范围 ====
price_min = seller_cost[test_id]
price_max = max(buyer_budget)

# ==== Run tests ====
num_trials = 50
rand_results = []
for _ in range(num_trials):
    p = np.random.uniform(price_min, price_max)  # 连续区间内均匀采样
    prof = compute_profit(p, prices_others)
    rand_results.append((p, prof))

best_random_price, best_random_profit = max(rand_results, key=lambda x: x[1])
avg_random_profit = np.mean([x[1] for x in rand_results])



# ==== Output results ====
print(f"\n📊 Comparison for Seller {test_id}:")
print(f"Nash Strategy:   Price = {price_nash:.4f}, Profit = {profit_nash:.4f}")
print(f"Fixed Strategy:  Price = {price_fixed:.4f}, Profit = {profit_fixed:.4f}")
print(f"Random Strategy: Best Price = {best_random_price:.4f}, Best Profit = {best_random_profit:.4f}")
print(f"Random Strategy: Avg  Profit = {avg_random_profit:.4f}")

# ==== Save results ====
with open(f"seller{test_id}_strategy_comparison.txt", "w") as f:
    f.write("Strategy,Price,Profit\n")
    f.write(f"Nash,{price_nash:.4f},{profit_nash:.4f}\n")
    f.write(f"Fixed,{price_fixed:.4f},{profit_fixed:.4f}\n")
    f.write(f"RandomBest,{best_random_price:.4f},{best_random_profit:.4f}\n")
    f.write(f"RandomAvg,-,{avg_random_profit:.4f}\n")

