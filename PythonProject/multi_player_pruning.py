#############区别就是加了个budet，因为budget不同
import time
import numpy as np
import load_data as ld
import Neighbor
import os
from itertools import product
K = 5
filename_p_total = "provider_comparison.txt"
target_ids =[0]
TOTAL_NUM_PROVIDER=Neighbor.generate_provider_neighbors(filename_p_total,target_ids,K)


start_time = time.time()

# Load data
filename_c = "customer_comparision.txt"
filename_p = os.path.join("./neighbor", f"provider_0_subset.txt")

demand, resource, budget, lamda, cost, supply = ld.load_customer(filename_c, filename_p)
# lamda需要转置

#####################BUYER PARAMETERS#############################################
#  buyer 的偏好参数 (a0, a1)、需求量和预算（固定值）

buyer_params = lamda.T  #Performance Utility
buyer_demand = demand  # 需求量
num_buyers = len(buyer_demand)
total_demand = np.sum(buyer_demand)

print(f" the customer total demand is  {total_demand}")

buyer_budget = budget # 预算，固定值
print(buyer_budget)


#####################SELLER PARAMETERS#############################################
# 固定库存
seller_supply = supply
total_supply = np.sum(seller_supply)
seller_cost = cost
num_sellers = len(seller_cost)
# 离散价格空间
price_vals = [
    np.linspace(c, max(buyer_budget), 20)
    for c in seller_cost
]


# 构造价格网格
price_grids = np.meshgrid(*price_vals, indexing='ij')
grid_shape = price_grids[0].shape


# 初始化销量
q_total = [np.zeros(grid_shape) for _ in range(num_sellers)]



for param, demand, budget in zip(buyer_params, buyer_demand, buyer_budget):
    # 计算所有卖家的效用和预算掩码
    utilities = [param[k] - price_grids[k] for k in range(num_sellers)]
    masks = [(price_grids[k] <= budget).astype(float) for k in range(num_sellers)]
    exp_utils = [np.exp(u) * m for u, m in zip(utilities, masks)]

    denom = sum(exp_utils)
    # 如果 denom==0（所有价格都超预算），概率设0
    probs = [np.divide(eu, denom, out=np.zeros_like(eu), where=denom != 0) for eu in exp_utils]

    for k in range(num_sellers):
        q_total[k] += probs[k] * demand

# 应用库存限制
actual_q = [np.minimum(q_total[k], seller_supply[k]) for k in range(num_sellers)]




# 利润计算
# 通用利润计算
#profit = [actual_q[k] * price_grids[k] for k in range(num_sellers)]
profit = [actual_q[k] * (price_grids[k] - seller_cost[k]) for k in range(num_sellers)]



nash_points = []
grid_shape = price_grids[0].shape
price_axes = [vals for vals in price_vals]  # 每个 seller 的价格候选列表

# ====== 🚀 优化开始：只检查高利润区域 ======

threshold_ratio = 0  # 只考虑利润 ≥ 90% 最大值的组合
max_profit = [np.max(p) for p in profit]

# 每个 seller 的高利润组合索引
high_profit_indices = [
    np.argwhere(p >= threshold_ratio * max_p)
    for p, max_p in zip(profit, max_profit)
]

# 转换为 set of tuple 索引
high_profit_sets = [set(map(tuple, idx_list)) for idx_list in high_profit_indices]
valid_combinations = set.intersection(*high_profit_sets)

print(f"🔍 Only checking {len(valid_combinations)} high-profit combinations (out of {np.prod(grid_shape)})")

# ====== 纳什均衡判断逻辑（不变） ======

for idx in valid_combinations:
    is_nash = True
    for k in range(num_sellers):
        fixed_idx = list(idx)
        best_response = None
        best_profit = -np.inf
        for i in range(len(price_vals[k])):
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
        prices = [price_vals[k][idx[k]] for k in range(num_sellers)]
        profits = [profit[k][idx] for k in range(num_sellers)]
        nash_points.append((prices, profits))


# 输出纳什均衡点信息
if nash_points:
    print("🎯 Found NASH Equilibrium：")
    for idx, (price_list, profit_list) in enumerate(nash_points, 1):
        print(f"Nash Equilibrium {idx}:")
        for k in range(num_sellers):
            print(f"  Seller{k}: Price = {price_list[k]:.3f}, Revenue = {profit_list[k]:.3f}")
else:
    print("⚠️ Can not found NASH Equilibrium")


end_time = time.time()  # 记录结束时间

elapsed_time = end_time - start_time
print(f"Running time：{elapsed_time:.4f} second(s)")

#########################################Comparison output (fixed random price)###########################
import os

file_path = "Pricing-strategy-pruning-4providers.txt"

# 判断文件是否存在，决定是否写表头
write_header = not os.path.exists(file_path)

with open(file_path, "a") as f:
    if write_header:
        f.write("ID,Price,Profit\n")
    # 追加写入nash_points内容
    for _, (price_list, profit_list) in enumerate(nash_points):
        for seller_id, (price, profit) in enumerate(zip(price_list, profit_list)):
            f.write(f"{seller_id},{price:.4f},{profit:.4f}\n")
    f.write(f"# customer number: {num_buyers}, customer demand: {total_demand}\n")
    f.write(f"# provider number: {num_sellers}, customer demand: {total_supply}\n")
    f.write(f"# Running time: {elapsed_time:.4f} seconds\n")


print("✅ Pricing strategy saved")
#########################################################################################

#########################################Runing time output###########################
#
# # ===== ✅ Runtime 日志记录 =====
# runtime_log_path = "runtime_log.txt"
#
# with open(runtime_log_path, "a") as f:
#     # f.write(f"TOTAL_NUM_PROVIDER: {TOTAL_NUM_PROVIDER}, K: {K}, Providers: {num_sellers},Providers_supply: {total_supply}, Customers: {num_buyers}, Customers_demand: {total_demand}, Runtime: {elapsed_time:.4f} seconds\n")
#     # f.write(
#     #     f"Customers: {num_buyers}, Runtime: {elapsed_time:.4f} seconds, threshold_ratio: {threshold_ratio}\n")    #这个是给customer变化进行的输出
#
#     f.write(
#             f"TOTAL_NUM_PROVIDER: {TOTAL_NUM_PROVIDER}, Runtime: {elapsed_time:.4f} seconds, threshold_ratio: {threshold_ratio}\n")    #这个是给K变化进行的输出
# print("📝 Runtime log saved to 'runtime_log.txt'")
