#############区别就是加了个budet，因为budget不同
import time
import numpy as np
import load_data as ld
from itertools import product


start_time = time.time()

# Load data
filename_c = "customer_comparision.txt"
filename_p = "provider_comparison.txt"
demand, resource, budget, lamda, cost, supply = ld.load_customer(filename_c, filename_p)
# lamda需要转置

#####################BUYER PARAMETERS#############################################
#  buyer 的偏好参数 (a0, a1)、需求量和预算（固定值）

buyer_params = lamda.T  #Performance Utility
buyer_demand = demand  # 需求量
total_demand = np.sum(buyer_demand)

print(f" the customer total demand is  {total_demand}")

buyer_budget = budget # 预算，固定值
print(buyer_budget)


#####################SELLER PARAMETERS#############################################
# 固定库存
seller_supply = supply
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
    probs = [np.where(denom == 0, 0, eu / denom) for eu in exp_utils]

    for k in range(num_sellers):
        q_total[k] += probs[k] * demand

# 应用库存限制
actual_q = [np.minimum(q_total[k], seller_supply[k]) for k in range(num_sellers)]




# 利润计算
# 通用利润计算
profit = [actual_q[k] * price_grids[k] for k in range(num_sellers)]



nash_points = []
grid_shape = price_grids[0].shape
price_axes = [vals for vals in price_vals]  # 每个 seller 的价格候选列表

# 遍历所有价格组合索引
for idx in np.ndindex(grid_shape):
    is_nash = True
    for k in range(num_sellers):
        # 固定其他卖家的价格索引
        fixed_idx = list(idx)

        # seller k 在其所有价格上找最大利润
        best_response = None
        best_profit = -np.inf
        for i in range(len(price_axes[k])):
            test_idx = fixed_idx.copy()
            test_idx[k] = i
            profit_at_idx = profit[k][tuple(test_idx)]
            if profit_at_idx > best_profit:
                best_profit = profit_at_idx
                best_response = i

        # 当前组合不是 seller k 的最优响应
        if best_response != idx[k]:
            is_nash = False
            break

    if is_nash:
        prices = [price_axes[k][idx[k]] for k in range(num_sellers)]
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
    print("⚠️ 未找到纯策略纳什均衡点")


end_time = time.time()  # 记录结束时间

elapsed_time = end_time - start_time
print(f"Running time：{elapsed_time:.4f} second(s)")

##############################纳什均衡输出保存############################################

# # 写入每个 seller 的 ID, Price, Profit 到 txt
# with open("Pricing-strategy.txt", "w") as f:
#     f.write("ID,Price,Profit\n")  # 表头
#
#     for _, (price_list, profit_list) in enumerate(nash_points):
#         for seller_id, (price, profit) in enumerate(zip(price_list, profit_list)):
#             f.write(f"{seller_id},{price:.4f},{profit:.4f}\n")
#     f.write(f"# Running time: {elapsed_time:.4f} seconds\n")
# print("✅ Pricing strategy saved")

#######################################################################################