import numpy as np

#论文中地形式 a=1,b=7.6
def custom_sigmoid_a(x):
    return 1.5 * np.tanh(3.8 * x)
#论文中的形式， a=1,b=76
def custom_sigmoid_b(x):
    return 1.5 * np.tanh(38 * x)


def load_customer(filename_c, filename_p):
    demand = []
    resource = []
    budget = []

    # 读取 customer 数据（a 和 b）
    with open(filename_c, "r") as f:
        next(f)
        for line in f:
            fields = line.strip().split(",")
            demand.append(int(fields[1]))
            ab = [round(float(fields[2]), 3), round(float(fields[3]), 3)]  # 只取 a 和 b
            resource.append(ab)
            budget.append(np.array(float(fields[4])))

    weights = [6, 3]  # 固定预算权重


    lamda = []
    cost = []
    supply = []

    # 读取 provider 数据
    with open(filename_p, "r") as f:
        next(f)
        for line in f:
            fields = line.strip().split(",")
            supply.append(int(fields[1]))
            ab = [round(float(fields[2]), 3), round(float(fields[3]), 3)]  # 只取 a 和 b

            # 计算 lambda
            lamm = [
                round(
                    custom_sigmoid_a(-ab[0] + r[0]) +  # 对 a 单独应用 custom_sigmoid_a
                    custom_sigmoid_b(-(ab[1] - r[1])),  # 对 b 单独应用 custom_sigmoid_b
                    3
                )
                for r in resource
            ]

            lamda.append(lamm)

            # 成本计算

            cost.append(np.array(float(fields[4])))

    return (
        np.array(demand),
        np.array(resource),
        np.array(budget),
        np.array(lamda),
        np.array(cost),
        np.array(supply)
    )

if __name__ == "__main__":
    filename_c = "customer_comparision.txt"
    filename_p = "provider_comparison.txt"
    demand, resource, budget, lamda, cost, supply = load_customer(filename_c, filename_p)

    print("✅ 加载成功")
    print("样例预算：", budget[:2])
    print("样例 lambda：", lamda[:2])
