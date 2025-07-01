######带缩放功能， 无供需关系，2player完成


import numpy as np
import matplotlib.pyplot as plt
import nashpy as nash
from numpy.ma.core import append


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
#demand--buyer的需求，resource--buyer的每个quality的需求， budget--buyer对于1个单位的预算，lamda--除了price的utility且参数都是1.
#cost--seller1个单位的成本， supplu--seller的supply
def print_nonzero_strategy(strategy, price_range, player_name,cost):
    print(f"\n{player_name}'s strategy distribution:")
    for price, prob in zip(price_range, strategy):
        if prob > 0:  # 只打印非0概率
            print(f"Price: {price:.3f} | Probability: {prob*100:.1f}%")
    print(f"{player_name}'s cost:{cost}")

def print_average_price(strategy, price_range):
    # 计算平均价格
    average_price = np.dot(strategy, price_range)  # 点积计算平均价格
    return average_price

def print_supply_price(price1, price2, supply, revenue0, revenue1):
    # 创建图形，设置为2行1列的布局
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    # 绘制价格和供应的关系
    ax[0].plot(supply, price1, label="Price 0", color='blue', marker='o')
    ax[0].plot(supply, price2, label="Price 1", color='green', marker='x')

    # 添加标题和标签
    ax[0].set_title("Price vs Supply", fontsize=14)
    ax[0].set_xlabel("Supply", fontsize=12)
    ax[0].set_ylabel("Price", fontsize=12)

    # 显示图例
    ax[0].legend()
    ax[0].grid(True)

    # 绘制revenue与价格的关系
    ax[1].plot(supply, revenue0, label="Revenue 0", color='red', marker='o')
    ax[1].plot(supply, revenue1, label="Revenue 1", color='purple', marker='x')

    # 添加标题和标签
    ax[1].set_title("Revenue vs Price", fontsize=14)
    ax[1].set_xlabel("Supply", fontsize=12)
    ax[1].set_ylabel("Revenue", fontsize=12)

    # 显示图例
    ax[1].legend()
    ax[1].grid(True)

    # 显示图形
    plt.tight_layout()  # 确保图形不会重叠
    plt.show()
def calculate_expected_revenue(revenue_matrix, prob_player0, prob_player1):

    expected_revenue = 0
    for i in range(revenue_matrix.shape[0]):  # 遍历行
        for j in range(revenue_matrix.shape[1]):  # 遍历列
            expected_revenue += revenue_matrix[i, j] * prob_player0[i] * prob_player1[j]

    return expected_revenue
##----------------------------------------------------------------------------
def probability(U_0, U_1, demand, P0, P1,budget, index):  #index就是价格前面的系数
    p_0 = np.exp(U_0 - index*demand * P0) if P0<=budget else 0
    p_1 = np.exp(U_1 - index*demand * P1) if P1<=budget else 0
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

sd_index = np.sum(supply)/np.sum(demand)

lamda_0 = lamda[0]
lamda_1 = lamda[1]

Utility_0 = lamda_0 * demand
Utility_1 = lamda_1 * demand



P0_range = np.arange(cost[0],  np.max(budget), 0.025)
P1_range = np.arange(cost[1], np.max(budget), 0.025)


# Initialize a matrix to store the choice_0 probabilities
choice_0_matrix = []
choice_1_matrix = []


# Loop through P0 and P1 and calculate choice_0 probabilities
customer =len(demand)   #当前的客户编号
total_supply = np.sum(supply)
total_demand = np.sum(demand[:customer])
max_iterations = 15  # 限制迭代次数
tolerance = 0.005  # 收敛阈值，防止无限循环

supp = np.arange(50, 1501, 50)  # 生成 0 到 700，步长为 50
price0=[]
price1=[]
eq_revenue0 =[]
eq_revenue1 =[]
for s in supp:

    for iteration in range(max_iterations):

        quantity_0 = np.zeros((len(P0_range), len(P1_range)))
        quantity_1 = np.zeros((len(P0_range), len(P1_range)))

        for c in range(customer):

            choice_c0 = []
            choice_c1 = []

            for P0 in P0_range:
                choice_p0 = []
                choice_p1 = []

                for P1 in P1_range:
                    choice_0 = probability(Utility_0[c], Utility_1[c], demand[c], P0, P1,budget[c],1)
                    choice_p0.append(choice_0[0])
                    choice_p1.append(choice_0[1])


                choice_c0.append(choice_p0)
                choice_c1.append(choice_p1)
            quantity_0 =  np.round(quantity_0 + np.array(choice_c0) * demand[c])
            quantity_1 = np.round(quantity_1 + np.array(choice_c1) * demand[c])
        quantity_0 = np.minimum(quantity_0, supply[0])
        quantity_1 = np.minimum(quantity_1, s)
        revenue0 = quantity_0 * P0_range[:, np.newaxis]
        revenue1 = quantity_1 * P1_range
        revenue0_nor = (revenue0 - np.mean(revenue0)) / np.std(revenue0)
        revenue1_nor = (revenue1 - np.mean(revenue1)) / np.std(revenue1)

        game = nash.Game(revenue0_nor, revenue1_nor)
        equilibria = list(game.lemke_howson_enumeration())




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
        #如果有纯策略
        if len(p0_nonzero_idx) == 1 and len(p1_nonzero_idx) == 1:

            break
        # 取均衡价格范围
        new_P0_min, new_P0_max = np.min(P0_range[p0_nonzero_idx]), np.max(P0_range[p0_nonzero_idx])
        new_P1_min, new_P1_max = np.min(P1_range[p1_nonzero_idx]), np.max(P1_range[p1_nonzero_idx])

        # 生成新的更精细的搜索范围
        step_size = max((new_P0_max - new_P0_min) / 5, tolerance)  # 防止步长过小
        P0_range_new = np.arange(new_P0_min, new_P0_max + step_size, step_size)

        step_size = max((new_P1_max - new_P1_min) /5, tolerance)  # 防止步长过小
        P1_range_new = np.arange(new_P1_min, new_P1_max + step_size, step_size)

        if np.array_equal(P0_range, P0_range_new) and np.array_equal(P1_range, P1_range_new):
            print("Search range did not change, stopping iterations.")
            break
        P0_range, P1_range = P0_range_new, P1_range_new

    print_nonzero_strategy(equilibria[1][0], P0_range, "player 1", cost[0])
    print_nonzero_strategy(equilibria[1][1], P1_range, "player 2", cost[1])

    eq_revenue0.append(calculate_expected_revenue(revenue0, equilibria[1][0], equilibria[1][1]))
    eq_revenue1.append(calculate_expected_revenue(revenue1, equilibria[1][0], equilibria[1][1]))


    price0.append(print_average_price(equilibria[1][0],P0_range))
    price1.append(print_average_price(equilibria[1][1], P1_range))

print(f"\nMarket's Supply and Demand distribution:")
print(f"Buyer Demand: {np.sum(demand)}")


print_supply_price(price0,price1,supp,eq_revenue0,eq_revenue1)
