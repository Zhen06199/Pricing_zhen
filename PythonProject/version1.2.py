import numpy as np
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
            demand.append(capacity_demand)

            abc = [round(float(fields[i]), 3) for i in range(2, 5)]
            resource.append(abc)

            budget_0 = round(0.1 * (6 * abc[0] + 3 * abc[1] + 4 * abc[2]), 3)
            budget.append(budget_0 * 0.50)

    num_customer = len(resource)
    lamda = []
    cost = []
    supply = []

    with open(filename_p, "r") as f:
        next(f)
        for line in f:
            fields = line.strip().split(",")
            supply.append(int(fields[1]))

            abc = np.array([round(float(fields[i]), 3) for i in range(2, 5)])

            lamm = [round(np.sum(custom_sigmoid(abc - np.array(resource[x]))), 3) for x in range(num_customer)]
            lamda.append(lamm)

            cost0 = round(0.03 * (3 * abc[0] + 3 * abc[1] + 4 * abc[2]), 3)
            cost.append(cost0)

    return np.array(demand), np.array(resource), np.array(budget), np.array(lamda), np.array(cost), np.array(supply)

def probability(U_0, U_1, demand, P0, P1, budget):
    p_0 = np.exp(U_0 - demand * P0) if P0 <= budget else 0
    p_1 = np.exp(U_1 - demand * P1) if P1 <= budget else 0
    if p_0 + p_1 == 0:
        return [0, 0]
    return [round(p_0 / (p_0 + p_1), 3), round(p_1 / (p_0 + p_1), 3)]
# 读取数据
filename_c = "C:/Users/yuzhe/PycharmProjects/PythonProject/customer.txt"
filename_p = "C:/Users/yuzhe/PycharmProjects/PythonProject/provider.txt"
demand, resource, budget, lamda, cost, supply = load_customer(filename_c, filename_p)

lamda_0, lamda_1 = lamda[0], lamda[1]
Utility_0, Utility_1 = lamda_0 * demand, lamda_1 * demand

# 初始搜索范围
P0_range = np.arange(cost[0], np.max(budget), 0.02)
P1_range = np.arange(cost[1], np.max(budget), 0.02)

customer = 10
max_iterations = 15  # 限制迭代次数
tolerance = 0.005  # 收敛阈值，防止无限循环

for iteration in range(max_iterations):
    revenue0 = np.zeros((len(P0_range), len(P1_range)))
    revenue1 = np.zeros((len(P0_range), len(P1_range)))

    for c in range(customer):
        for i, P0 in enumerate(P0_range):
            for j, P1 in enumerate(P1_range):
                choice_0 = probability(Utility_0[c], Utility_1[c], demand[c], P0, P1, budget[c])
                revenue0[i, j] += P0 * demand[c] * choice_0[0]
                revenue1[i, j] += P1 * demand[c] * choice_0[1]

    # 归一化，避免数值过大
    revenue0 = (revenue0 - np.mean(revenue0)) / np.std(revenue0)
    revenue1 = (revenue1 - np.mean(revenue1)) / np.std(revenue1)

    # 计算纳什均衡
    game = nash.Game(revenue0, revenue1)
    equilibria = list(game.lemke_howson_enumeration())

    print(f"Final Player 1's mixed strategy: {equilibria[1][0]}")
    print(f"Final Reduced P0_range: {P0_range}")
    print(f"Final Player 2's mixed strategy: {equilibria[1][1]}")
    print(f"Final Reduced P1_range: {P1_range}")

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
    step_size = max((new_P0_max - new_P0_min) / 5, tolerance)  # 防止步长过小
    P0_range = np.arange(new_P0_min, new_P0_max + step_size, step_size)

    step_size = max((new_P1_max - new_P1_min) / 5, tolerance)  # 防止步长过小
    P1_range = np.arange(new_P1_min, new_P1_max + step_size, step_size)

    # 终止条件：若 P0_range 和 P1_range 变化幅度很小，认为收敛
    if (new_P0_max - new_P0_min < tolerance) and (new_P1_max - new_P1_min < tolerance):
        print("Search converged.")
        break

# 输出最终结果
print(f"Final Player 1's mixed strategy: {equilibria[1][0]}")
print(f"Final Reduced P0_range: {P0_range}")
print(f"Final Player 2's mixed strategy: {equilibria[1][1]}")
print(f"Final Reduced P1_range: {P1_range}")
print("Finish")