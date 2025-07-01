import numpy as np
import matplotlib.pyplot as plt
import nashpy as nash

def custom_sigmoid(x):
    return 0.2 * np.tanh(20 * x)

import numpy as np

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
            budget.append(budget_0)

    num_customer = len(resource)
    lamda = []
    cost = []

    with open(filename_p, "r") as f:
        next(f)
        for line in f:
            fields = line.strip().split(",")
            supply = int(fields[1])
            abc = np.array([round(float(fields[i]), 3) for i in range(2, 5)])

            # 计算 lambda
            lamm = [round(np.sum(custom_sigmoid(abc - np.array(resource[x]))), 3) for x in range(num_customer)]
            lamda.append(lamm)

            # 计算成本
            cost0 = round(0.05 * (3 * abc[0] + 3 * abc[1] + 4 * abc[2]), 3)
            cost.append(cost0)

    return np.array(demand), np.array(resource), np.array(budget), np.array(lamda), np.array(cost)




##----------------------------------------------------------------------------
def probability(U_0, U_1, demand, P0, P1):
    p_0 = np.exp(U_0 - demand * P0)
    p_1 = np.exp(U_1 - demand * P1)
    denominator = p_0 + p_1
    pr_0 = p_0 / denominator
    pr_1 = p_1 / denominator

    return [round(pr_0, 3), round(pr_1, 3)]


##----------------------------------------------------------------------------
filename_c= "/customer_comparision.txt"
filename_p= "/provider_comparison.txt"
demand,resource,budget, lamda,cost = load_customer(filename_c,filename_p)


lamda_0 = lamda[0]
lamda_1 = lamda[1]
#demand = np.array([30, 70, 74,50])

Utility_0 = lamda_0 * demand
Utility_1 = lamda_1 * demand

P0_range = np.arange(0.3, 0.8, 0.06)
P1_range = np.arange(0.3, 0.8, 0.06)

# Initialize a matrix to store the choice_0 probabilities
choice_0_matrix = []
choice_1_matrix = []
revenue0 = np.zeros((len(P0_range),len(P1_range)))
revenue1 = np.zeros((len(P0_range),len(P1_range)))

# Loop through P0 and P1 and calculate choice_0 probabilities
customer =10   #当前的客户编号
for c in range(customer):
    revenue_0_matrix = []
    revenue_1_matrix = []

    for P0 in P0_range:
        money_0 = []
        money_1 = []
        # row_0 = []
        # row_1 = []
        for P1 in P1_range:
            choice_0 = probability(Utility_0[c], Utility_1[c], demand[c], P0, P1)
            revenue_0 = round(P0*demand[c]*choice_0[0],4)
            revenue_1 = round(P1 * demand[c] * choice_0[1],4)
            #print(f"P0: {P0:.2f}, P1: {P1:.2f}, choice_0: {choice_0},revenue_0: {revenue_0}, revenue_1: {revenue_1}" )

            money_0.append(revenue_0)
            money_1.append(revenue_1)

        #     row_0.append(choice_0[0])  # Store the first probability (choice_0)
        #     row_1.append(choice_0[1])
        # choice_0_matrix.append(row_0)
        # choice_1_matrix.append(row_1)

        revenue_0_matrix.append(money_0)
        revenue_1_matrix.append(money_1)
    a =np.array(revenue_0_matrix)
    revenue0 = np.round(revenue0 + np.array(revenue_0_matrix),3)
    revenue1 = np.round(revenue1 + np.array(revenue_1_matrix),3)

revenue0 = (revenue0 - np.mean(revenue0)) / np.std(revenue0)
revenue1 = (revenue1 - np.mean(revenue1)) / np.std(revenue1)

print(revenue0)
print(revenue1)

game = nash.Game(revenue0, revenue1)
equilibria = list(game.lemke_howson_enumeration())

print(f"Player 1's mixed strategy: {equilibria[0][0]}")
print(f"Player 2's mixed strategy: {equilibria[0][1]}")
print("finish")

