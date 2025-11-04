import numpy as np
import matplotlib.pyplot as plt
import time

# Start the timer
start_time = time.time()
k1 = k2 = k3 = 4200

m_values = [0.83, 0.83, 0.83]
k_values = [4200, 4200, 4200]
f_values = [1, 1, 1]

l_values = np.arange(0, 5, 1)

w_values = np.arange(0, 150, 1)

undamped_y1 = []
undamped_y2 = []
undamped_y3 = []

min_l1 = float('inf')
min_l2 = float('inf')
min_l3 = float('inf')
min_l1_l2 = float('inf')
min_l1_l3 = float('inf')
min_l2_l3 = float('inf')
min_l1_l2_l3 = float('inf')

max_y1_l1 = max_y2_l2 = max_y3_l3 = max_y1_l1_l2 = max_y2_l1_l2 = max_y3_l1_l2 = max_y1_l2_l3 = max_y2_l2_l3 = max_y3_l2_l3 = max_y1_l1_l2_l3 = max_y2_l1_l2_l3 = max_y3_l1_l2_l3 = 0
max_y1_l1 = max_y2_l1 = max_y3_l1 = 0
max_y1_l2 = max_y2_l2 = max_y3_l2 = 0
max_y1_l3 = max_y2_l3 = max_y3_l3 = 0
max_y1_l1_l2 = max_y2_l1_l2 = max_y3_l1_l2 = 0
max_y1_l1_l3 = max_y2_l1_l3 = max_y3_l1_l3 = 0
max_y1_l2_l3 = max_y2_l2_l3 = max_y3_l2_l3 = 0
max_y1_l1_l2_l3 = max_y2_l1_l2_l3 = max_y3_l1_l2_l3 = 0
for l1 in l_values:
    for l2 in l_values:
        for l3 in l_values:
            L = np.array([[l1 + l2, -l2, 0], [-l2, l2, -l3], [0, -l2, l3]])
            K = np.array([[k1 + k2, -k2, 0], [-k2, k2, -k3], [0, -k2, k3]])
            F = np.array(f_values)

            results = []

            for w in w_values:
                Inv = np.linalg.inv(-w**2 * np.diag([m_values[0], m_values[1], m_values[2]]) + 1j * w * L + K)
                X = np.dot(F, Inv)
                results.append(abs(X))

            y_values = np.array(results).T

            if l1 == 0 and l2 == 0 and l3 == 0:
                undamped_y1 = y_values[0]
                undamped_y2 = y_values[1]
                undamped_y3 = y_values[2]

            y1_percentage = np.max(y_values[0]) / np.max(undamped_y1)
            y2_percentage = np.max(y_values[1]) / np.max(undamped_y2)
            y3_percentage = np.max(y_values[2]) / np.max(undamped_y3)

            if y1_percentage <= 0.1 and y2_percentage <= 0.1 and y3_percentage <= 0.1:
                sum_l_values = l1 + l2 + l3

                if l1 < min_l1 and l2 == 0 and l3 == 0:
                    min_l1 = l1
                    max_y1_l1 = np.max(y_values[0])
                    max_y2_l1 = np.max(y_values[1])
                    max_y3_l1 = np.max(y_values[2])
                if l2 < min_l2 and l1 == 0 and l3 == 0:
                    min_l2 = l2
                    max_y1_l2 = np.max(y_values[0])
                    max_y2_l2 = np.max(y_values[1])
                    max_y3_l2 = np.max(y_values[2])
                if l3 < min_l3 and l1 == 0 and l2 == 0:
                    min_l3 = l3
                    max_y1_l3 = np.max(y_values[0])
                    max_y2_l3 = np.max(y_values[1])
                    max_y3_l3 = np.max(y_values[2])
                if sum_l_values < min_l1_l2 and l1 > 0 and l2 > 0 and l3 == 0:
                    min_l1_l2 = sum_l_values
                    max_y1_l1_l2 = np.max(y_values[0])
                    max_y2_l1_l2 = np.max(y_values[1])
                    max_y3_l1_l2 = np.max(y_values[2])
                    min_l1_l2_l1 = l1
                    min_l1_l2_l2 = l2
                if sum_l_values < min_l1_l3 and l1 > 0 and l3 > 0 and l2 == 0:
                    min_l1_l3 = sum_l_values
                    max_y1_l1_l3 = np.max(y_values[0])
                    max_y2_l1_l3 = np.max(y_values[1])
                    max_y3_l1_l3 = np.max(y_values[2])
                    min_l1_l3_l1 = l1
                    min_l1_l3_l3 = l3
                if sum_l_values < min_l2_l3 and l2 > 0 and l3 > 0 and l1 == 0:
                    min_l2_l3 = sum_l_values
                    max_y1_l2_l3 = np.max(y_values[0])
                    max_y2_l2_l3 = np.max(y_values[1])
                    max_y3_l2_l3 = np.max(y_values[2])
                    min_l2_l3_l2 = l2
                    min_l2_l3_l3 = l3
                if sum_l_values < min_l1_l2_l3 and l1 > 0 and l2 > 0 and l3 > 0:
                    min_l1_l2_l3 = sum_l_values
                    max_y1_l1_l2_l3 = np.max(y_values[0])
                    max_y2_l1_l2_l3 = np.max(y_values[1])
                    max_y3_l1_l2_l3 = np.max(y_values[2])
                    min_l1_l2_l3_l1 = l1
                    min_l1_l2_l3_l2 = l2
                    min_l1_l2_l3_l3 = l3

print(f"Minimum l1 with l2=0, l3=0: l1 = {min_l1}, l2 = 0, l3 = 0")
print(f"Maximum y1, y2, and y3 for l1: {max_y1_l1} {max_y2_l1} {max_y3_l1}")
print(f"Minimum l2 with l1=0, l3=0: l1 = 0, l2 = {min_l2}, l3 = 0")
print(f"Maximum y1, y2, and y3 for l2: {max_y1_l2} {max_y2_l2} {max_y3_l2}")
print(f"Minimum l3 with l1=0, l2=0: l1 = 0, l2 = 0, l3 = {min_l3}")
print(f"Maximum y1, y2, and y3 for l3: {max_y1_l3} {max_y2_l3} {max_y3_l3}")
print(f"Minimum l1 + l2 with l3=0: l1 = {min_l1_l2_l1}, l2 = {min_l1_l2_l2}, l3 = 0")
print(f"Maximum y1, y2, and y3 for l1 + l2: {max_y1_l1_l2} {max_y2_l1_l2} {max_y3_l1_l2}")
print(f"Minimum l1 + l3 with l2=0: l1 = {min_l1_l3_l1}, l3 = {min_l1_l3_l3}, l2 = 0")
print(f"Maximum y1, y2, and y3 for l1 + l3: {max_y1_l1_l3} {max_y2_l1_l3} {max_y3_l1_l3}")
print(f"Minimum l2 + l3 with l1=0: l2 = {min_l2_l3_l2}, l3 = {min_l2_l3_l3}, l1 = 0")
print(f"Maximum y1, y2, and y3 for l2 + l3: {max_y1_l2_l3} {max_y2_l2_l3} {max_y3_l2_l3}")
print(f"Minimum l1 + l2 + l3: l1 = {min_l1_l2_l3_l1}, l2 = {min_l1_l2_l3_l2}, l3 = {min_l1_l2_l3_l3}")
print(f"Maximum y1, y2, and y3 for l1 + l2 + l3: {max_y1_l1_l2_l3} {max_y2_l1_l2_l3} {max_y3_l1_l2_l3}")
# Assuming you have already run the previous code

# Define the 7 minimum combinations
combinations = [
    (min_l1, 0, 0),
    (0, min_l2, 0),
    (0, 0, min_l3),
    (min_l1_l2_l1, min_l1_l2_l2, 0),
    (min_l1_l3_l1, 0, min_l1_l3_l3),
    (0, min_l2_l3_l2, min_l2_l3_l3),
    (min_l1_l2_l3_l1, min_l1_l2_l3_l2, min_l1_l2_l3_l3)
]

# Create a list to store the corresponding y values
y_values_for_combinations = []

# Calculate and store the y values for each of the 7 combinations
for l1, l2, l3 in combinations:
    L = np.array([[l1 + l2, -l2, 0], [-l2, l2, -l3], [0, -l2, l3]])
    K = np.array([[k1 + k2, -k2, 0], [-k2, k2, -k3], [0, -k2, k3]])
    F = np.array(f_values)

    results = []

    for w in w_values:
        Inv = np.linalg.inv(-w**2 * np.diag([m_values[0], m_values[1], m_values[2]]) + 1j * w * L + K)
        X = np.dot(F, Inv)
        results.append(abs(X))

    y_values = np.array(results).T
    y_values_for_combinations.append(y_values)

print("Undamped Case (All l values are zero):")
print(f"Maximum y1 for undamped case: {max(undamped_y1)}")
print(f"Maximum y2 for undamped case: {max(undamped_y2)}")
print(f"Maximum y3 for undamped case: {max(undamped_y3)}")

# Create plots for each of the 7 combinations
for i, (l1, l2, l3) in enumerate(combinations):
    plt.figure(figsize=(10, 5))
    plt.title(f"Response for Combination {i + 1}: l1={l1}, l2={l2}, l3={l3}")
    plt.xlabel('Frequency (w)')
    plt.ylabel('Amplitude')
    plt.plot(w_values, y_values_for_combinations[i][0], label='y1')
    plt.plot(w_values, y_values_for_combinations[i][1], label='y2')
    plt.plot(w_values, y_values_for_combinations[i][2], label='y3')

    # Annotate with maximum y values
    max_y1, max_y2, max_y3 = 0, 0, 0
    if i == 0:
        max_y1, max_y2, max_y3 = max_y1_l1, max_y2_l1, max_y3_l1
    elif i == 1:
        max_y1, max_y2, max_y3 = max_y1_l2, max_y2_l2, max_y3_l2
    elif i == 2:
        max_y1, max_y2, max_y3 = max_y1_l3, max_y2_l3, max_y3_l3
    elif i == 3:
        max_y1, max_y2, max_y3 = max_y1_l1_l2, max_y2_l1_l2, max_y3_l1_l2
    elif i == 4:
        max_y1, max_y2, max_y3 = max_y1_l1_l3, max_y2_l1_l3, max_y3_l1_l3
    elif i == 5:
        max_y1, max_y2, max_y3 = max_y1_l2_l3, max_y2_l2_l3, max_y3_l2_l3
    elif i == 6:
        max_y1, max_y2, max_y3 = max_y1_l1_l2_l3, max_y2_l1_l2_l3, max_y3_l1_l2_l3

# Assuming max_y1, max_y2, and max_y3 are the maximum values for the respective y-values
    plt.annotate(f'Max y1: {max_y1:.6f}', (w_values[np.argmax(y_values_for_combinations[i][0])], max_y1))
    plt.annotate(f'Max y2: {max_y2:.6f}', (w_values[np.argmax(y_values_for_combinations[i][1])], max_y2))
    plt.annotate(f'Max y3: {max_y3:.6f}', (w_values[np.argmax(y_values_for_combinations[i][2])], max_y3))


    plt.legend()
    plt.grid()
    plt.show()

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")
