import numpy as np
import matplotlib.pyplot as plt

def probability(U_0, U_1, demand, P0, P1):
    p_0 = np.exp(U_0 - demand * P0)
    p_1 = np.exp(U_1 - demand * P1)
    denominator = p_0 + p_1
    pr_0 = p_0 / denominator
    pr_1 = p_1 / denominator

    return [round(pr_0, 3), round(pr_1, 3)]

lamda_0 = np.array([0.5, 0.4, 0.3])       # 供应商 0对于每个客户的utility
lamda_1 = np.array([0.4, 0.5, 0.35])      # 供应商 1对于每个客户的utility
demand = np.array([57, 68, 74])           # 每个客户的要求

Utility_0 = lamda_0 * demand
Utility_1 = lamda_1 * demand

P_range = np.arange(0.5, 0.71, 0.005)   #价格区间

# Initialize a matrix to store the choice_0 probabilities
choice_0_matrix = []
choice_1_matrix = []

# Loop through P0 and P1 and calculate choice_0 probabilities
for P0 in P_range:
    row_0 = []
    row_1 = []
    for P1 in P_range:
        choice_0 = probability(Utility_0[2], Utility_1[2], demand[2], P0, P1)
        print(f"P0: {P0:.2f}, P1: {P1:.2f}, choice_0: {choice_0}")
        row_0.append(choice_0[0])  # Store the first probability (choice_0)
        row_1.append(choice_0[1])
    choice_0_matrix.append(row_0)
    choice_1_matrix.append(row_1)

# Convert the list of rows into a NumPy array
choice_0_matrix = np.array(choice_0_matrix)
choice_1_matrix = np.array(choice_1_matrix)

# Create a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(choice_1_matrix, cmap='viridis', origin='lower', extent=[0.5, 0.7, 0.5, 0.7])
plt.colorbar(label="Probability of choice 0")

# Add labels and title
plt.xlabel("P0")
plt.ylabel("P1")
plt.title("Heatmap of choice_0 Probability")

# Show the plot
plt.show()
