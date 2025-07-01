import numpy as np
import matplotlib.pyplot as plt
import nashpy as nash


##----------------------------------------------------------------------------
def probability(U_0, U_1, demand, P0, P1):
    p_0 = np.exp(U_0 - demand * P0)
    p_1 = np.exp(U_1 - demand * P1)
    denominator = p_0 + p_1
    pr_0 = p_0 / denominator
    pr_1 = p_1 / denominator

    return [round(pr_0, 3), round(pr_1, 3)]


lamda_0 = np.array([0.5, 0.43, 0.1,0.6])
lamda_1 = np.array([0.4, 0.5, 0.55,0.1])
demand = np.array([30, 70, 74,50])

Utility_0 = lamda_0 * demand
Utility_1 = lamda_1 * demand

P0_range = np.arange(0.5, 0.71, 0.05)
P1_range = np.arange(0.5, 0.71, 0.05)

# Initialize a matrix to store the choice_0 probabilities
choice_0_matrix = []
choice_1_matrix = []
revenue0 = np.zeros((len(P0_range),len(P0_range)))
revenue1 = np.zeros((len(P1_range),len(P1_range)))

# Loop through P0 and P1 and calculate choice_0 probabilities
customer =4   #当前的客户编号
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
            print(f"P0: {P0:.2f}, P1: {P1:.2f}, choice_0: {choice_0},revenue_0: {revenue_0}, revenue_1: {revenue_1}" )

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

revenue0 += np.random.uniform(-0.001, 0.001, revenue0.shape)
revenue1 += np.random.uniform(-0.001, 0.001, revenue1.shape)


game = nash.Game(revenue0, revenue1)
equilibria = list(game.lemke_howson_enumeration())
for eq in equilibria:
    print(f"Player 1's mixed strategy: {eq[0]}")
    print(f"Player 2's mixed strategy: {eq[1]}")
print("finish")

#
# # Convert the list of rows into a NumPy array
# choice_0_matrix = np.array(choice_0_matrix)
# choice_1_matrix = np.array(choice_1_matrix)
#
#
# # Create a heatmap
# plt.figure(figsize=(8, 6))
# plt.imshow(revenue_0_matrix, cmap='viridis', origin='lower', extent=[0.5, 0.6, 0.5, 0.6])
# plt.colorbar(label="Probability of choice 0")
#
# # Add labels and title
# plt.xlabel("P0")
# plt.ylabel("P1")
# plt.title("Heatmap of choice_0 Probability")
#
# # Show the plot
# plt.show()
