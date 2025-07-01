######带缩放功能， 无供需关系，2player完成


import numpy as np
import matplotlib.pyplot as plt
import nashpy as nash

def custom_sigmoid(x):
    return 0.2 * np.tanh(20 * x)


def load_customer(filename_c, filename_p):
    demand = []
    resource = []
    budget = []

    with open(filename_c, "r") as f:
        next(f)
        for line in f:
            fields = line.strip().split(",")
            capacity_demand = int(fields[1])
            demand.append(capacity_demand)  # 直接存整数

            # 解析 a, b, c
            abc = [round(float(fields[i]), 3) for i in range(2, 5)]
            resource.append(abc)

            # 计算预算
            budget_0 = round(0.1 * (6 * abc[0] + 3 * abc[1] + 4 * abc[2]), 3)
            budget.append(budget_0*0.50  )

    num_customer = len(resource)
    lamda = []
    cost = []
    supply = []

    with open(filename_p, "r") as f:
        next(f)
        for line in f:
            fields = line.strip().split(",")
            supply_a = np.array(int(fields[1]))
            supply.append(supply_a)
            abc = np.array([round(float(fields[i]), 3) for i in range(2, 5)])

            # 计算 lambda
            lamm = [round(np.sum(custom_sigmoid(abc - np.array(resource[x]))), 3) for x in range(num_customer)]
            lamda.append(lamm)

            # 计算成本
            cost0 = round(0.03 * (3 * abc[0] + 3 * abc[1] + 4 * abc[2]), 3)
            cost.append(cost0)

    return np.array(demand), np.array(resource), np.array(budget), np.array(lamda), np.array(cost), np.array(supply)

def print_nonzero_strategy(strategy, price_range, player_name,cost):
    print(f"\n{player_name}'s strategy distribution:")
    for price, prob in zip(price_range, strategy):
        if prob > 0:  # 只打印非0概率
            print(f"Price: {price:.3f} | Probability: {prob*100:.1f}%")
    print(f"{player_name}'s cost:{cost}")

##----------------------------------------------------------------------------
def probability(U_0, U_1, demand, P0, P1,budget):
    p_0 = np.exp(U_0 - demand * P0) if P0<=budget else 0
    p_1 = np.exp(U_1 - demand * P1) if P0<=budget else 0
    if p_1+p_0==0:
        pr_0=0
        pr_1=0
    else:
        denominator = p_0 + p_1
        pr_0 = p_0 / denominator
        pr_1 = p_1 / denominator

    return [round(pr_0, 3), round(pr_1, 3)]


##-------读取数据---------------------------------------------------------
filename_c= "/customer_comparision.txt"
filename_p= "/provider_comparison.txt"
demand,resource,budget, lamda,cost, supply = load_customer(filename_c,filename_p)

lamda_0 = lamda[0]
lamda_1 = lamda[1]

Utility_0 = lamda_0 * demand
Utility_1 = lamda_1 * demand

P0_range = np.arange(cost[0], np.max(budget), 0.02)
P1_range = np.arange(cost[1], np.max(budget), 0.02)

# Initialize a matrix to store the choice_0 probabilities
choice_0_matrix = []
choice_1_matrix = []


# Loop through P0 and P1 and calculate choice_0 probabilities
customer =10   #当前的客户编号
total_supply = np.sum(supply)
total_demand = np.sum(demand[:customer])
max_iterations = 15  # 限制迭代次数
tolerance = 0.005  # 收敛阈值，防止无限循环


for iteration in range(max_iterations):

    revenue0 = np.zeros((len(P0_range), len(P1_range)))
    revenue1 = np.zeros((len(P0_range), len(P1_range)))

    for c in range(customer):
        revenue_0_matrix = []
        revenue_1_matrix = []

        for P0 in P0_range:
            money_0 = []
            money_1 = []

            for P1 in P1_range:
                choice_0 = probability(Utility_0[c], Utility_1[c], demand[c], P0, P1,budget[c])
                revenue_0 = round(P0*demand[c]*choice_0[0],4)
                revenue_1 = round(P1 * demand[c] * choice_0[1],4)

                money_0.append(revenue_0)
                money_1.append(revenue_1)


            revenue_0_matrix.append(money_0)
            revenue_1_matrix.append(money_1)
        a =np.array(revenue_0_matrix)
        revenue0 = np.round(revenue0 + np.array(revenue_0_matrix),3)
        revenue1 = np.round(revenue1 + np.array(revenue_1_matrix),3)

    revenue0 = (revenue0 - np.mean(revenue0)) / np.std(revenue0)
    revenue1 = (revenue1 - np.mean(revenue1)) / np.std(revenue1)

    game = nash.Game(revenue0, revenue1)
    equilibria = list(game.lemke_howson_enumeration())

    print(f"Player 1's mixed strategy: {equilibria[1][0]}")
    print(f"Player 1's mixed strategy: {P0_range}")
    print(f"Player 2's mixed strategy: {equilibria[1][1]}")
    print(f"Player 2's mixed strategy: {P1_range}")
    print("--------------------------------------------------------------------")

    if not equilibria:
        print("No Nash equilibrium found.")
        break


    # 获取非零策略索引
    p0_nonzero_idx = np.nonzero(equilibria[1][0])[0]
    p1_nonzero_idx = np.nonzero(equilibria[1][1])[0]

    # 确保索引不为空，否则终止迭代
    if len(p0_nonzero_idx) == 0 or len(p1_nonzero_idx) == 0:
        print("Nash equilibrium did not yield valid indexes.")
        break

    # 取均衡价格范围
    new_P0_min, new_P0_max = np.min(P0_range[p0_nonzero_idx]), np.max(P0_range[p0_nonzero_idx])
    new_P1_min, new_P1_max = np.min(P1_range[p1_nonzero_idx]), np.max(P1_range[p1_nonzero_idx])

    # 生成新的更精细的搜索范围
    step_size = max((new_P0_max - new_P0_min) / 8, tolerance)  # 防止步长过小
    P0_range_new = np.arange(new_P0_min, new_P0_max + step_size, step_size)

    step_size = max((new_P1_max - new_P1_min) / 8, tolerance)  # 防止步长过小
    P1_range_new = np.arange(new_P1_min, new_P1_max + step_size, step_size)

    if np.array_equal(P0_range, P0_range_new) and np.array_equal(P1_range, P1_range_new):
        print("Search range did not change, stopping iterations.")
        break
    P0_range, P1_range = P0_range_new, P1_range_new



print_nonzero_strategy(equilibria[1][0],P0_range,"player 1",cost[0])
print_nonzero_strategy(equilibria[1][1],P1_range,"player 2",cost[1])