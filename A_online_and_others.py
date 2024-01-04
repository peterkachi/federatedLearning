import numpy as np
import random
from scipy.optimize import linprog

N = 20 # clients
I = 4 # edges
T = N + 20
class_num = 10

Umax = 50
Umin = 1
Li = 1
Ui = 50

x = np.zeros((N,I))
x_first = np.zeros((N,I))
x_random = np.zeros((N,I))
x_greedy = np.zeros((N,I))
y = np.zeros((N,I))
y_first = np.zeros((N,I))
y_random = np.zeros((N,I))
y_greedy = np.zeros((N,I))
deadline = np.zeros(N, dtype=int)
obj = 0
obj_without_reward = 0
obj_first = 0
obj_random = 0
obj_greedy = 0

u = np.zeros(N)
u_first = np.zeros(N)
u_greedy = np.zeros(N)
r = np.zeros((I,T))
r_first = np.zeros((I,T))
r_random = np.zeros((I,T))
r_greedy = np.zeros((I,T))
rt = np.zeros((I,T))
rt_first = np.zeros((I,T))
rt_random = np.zeros((I,T))
rt_greedy = np.zeros((I,T))
p = np.ones((I,T))

f = np.zeros((N,I))
f_random = np.zeros((N,I))
REWARD = np.zeros((N,I))
REWARD_random = np.zeros((N,I))
WW = np.zeros(N, dtype=int) # each bid consumes w slots
WT = np.zeros(N, dtype=int) # number of time slots participate in local training
pay = np.zeros(N)
#pay_random = np.zeros(N)
payment_greedy = np.zeros((N,I))

gamma = 0.6 # threshold
alpha = 0.5 # ratio of reward for the clients

MAX_CONNECTION_NUM = 3
en = 5.0
d_alpha = 0.5 # Dirichlet Distribution

# generate D_i under Dirichlet distribution
D_i = []
for i in range(I):
    composition_ratio = np.random.dirichlet((d_alpha, d_alpha)*(class_num//2))
    D_i.append(composition_ratio)
D_n = []
cos_sim = np.zeros((N,I))
E_n = np.ones(N) * 5.0

def custom_sort(a, b, c):
    # Get sorting indices of a, b, c
    a_argsort = np.argsort(a)[::-1]  # in descending order
    b_argsort = np.argsort(b)[::-1]
    c_argsort = np.argsort(c)[::-1]

    # Create new arrays to store the sorted elements of b and c
    new_b = np.zeros_like(b)
    new_c = np.zeros_like(c)

    # Reorder b and c based on the sorting indices of a
    for i, index in enumerate(a_argsort):
        new_b[index] = b[b_argsort[i]]
        new_c[index] = c[c_argsort[i]]

    return new_b, new_c

def cosine_similarity(a, b):
    # First, ensure the inputs have the same length
    if a.shape[0] != b.shape[0]:
        return "Arrays have different lengths!"

    # Normalize the vectors
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)

    # Calculate cosine similarity
    similarity = np.dot(a_norm, b_norm)

    return similarity

def k_smallest_indices(sub_arr, k, start_index):
    # Find the indices of the k smallest elements in the sub-array
    smallest_indices = np.argpartition(sub_arr, kth=k-1)[:k]
    # Sort these indices to maintain their order in the original array
    sorted_indices = smallest_indices[np.argsort(sub_arr[smallest_indices])]
    # Map these indices back to the indices in the original array
    return sorted_indices + start_index

def ratio_of_rows_with_one(x):
    # Calculate the sum for each row
    row_sums = np.sum(x, axis=1)

    # Count the number of rows that have a sum of 1 (i.e., rows with one '1')
    rows_with_one = np.sum(row_sums == 1)

    # Compute the ratio
    ratio = rows_with_one / x.shape[0]

    return ratio

def random_traverse(N):
    # Create a list of integers from 0 to N-1
    numbers = list(range(N))

    # Randomly shuffle the list
    random.shuffle(numbers)

    # Iterate through the list in order
    for number in numbers:
        yield number

for n in range(N):
    dead = 0
    bid = np.zeros(I)
    ubid = np.zeros(I)
    pay_greedy = np.zeros(I)
    upay_greedy = np.zeros(I)
    reward = np.zeros(I)
    ureward = np.zeros(I)

    for i in range(I):
        w = random.randint(10,20)
        WW[n] = w

    wt = 1000
    while wt > WW[n]:
        wt = random.randint(0,20)
    WT[n] = wt

    composition_ratio = np.random.dirichlet((d_alpha, d_alpha)*(class_num//2))  # randomly generate
    D_n.append(composition_ratio)

    for i in range(I):
       ubid[i] = Umin + ((Umax - Umin) * random.random())
       bid[i] = ubid[i] * WW[n]

    for i in range(I):
        ureward[i] = 1000
        while ureward[i] > ubid[i]:
            ureward[i] = Umin + ((Umax - Umin) * random.random())
        reward[i] = ureward[i] * WT[n]

    for i in range(I):
        upay_greedy[i] = Umin + ((Umax - Umin) * random.random())
        pay_greedy[i] = upay_greedy[i] * WW[n]

    for i in range(I):
        cos_sim[n][i] = cosine_similarity(D_i[i], composition_ratio)

    bid_random = bid
    reward_random = reward
    bid, reward = custom_sort(cos_sim[n], bid, reward)
    #bid, pay_greedy = custom_sort(cos_sim[n], bid, pay_greedy)
    deadline[n] = n + 20
    f[n] = bid
    REWARD[n] = reward
    f_random[n] = bid_random
    REWARD_random[n] = reward_random
    payment_greedy[n] = pay_greedy

# our online method
for n in range(N):
    c_temp = np.zeros((I,T))+10000 # record satisfied i t's price
    u_max = -np.zeros(T) # find i on each slot t with maximum utility
    u_max_index = -1 # according i index
    i_edge = np.zeros((I,T), dtype=int)
    #max_utility = 0
    schedule_start_time_temp = -1
    schedule_edges_temp = -1

    # select all satisfied v on each slot t
    for t in range((n),deadline[n]):
        for i in range(I):
            if r[i][t] + 1 <= MAX_CONNECTION_NUM:
                i_edge[i][t] = 1

    # find maximum utility (our method)
    max_utility = -10000.0
    max_utility_start_time = -1
    max_utility_edge = -1
    for i in range(I):
        for t in range((n+WW[n]), deadline[n]):
            if not 0 in i_edge[i][t-WW[n]: t]:
                utility_temp = f[n][i] - np.sum(p[i][t-WW[n]: t])
                if max_utility < utility_temp:
                    max_utility = utility_temp
                    max_utility_start_time = t-WW[n]
                    max_utility_edge = i


    if max_utility > 0.0:
        u[n] = max_utility
        x[n][max_utility_edge] = 1
        pay[n] = np.sum(p[max_utility_edge][max_utility_start_time: max_utility_start_time+WW[n]])
        for t in range(max_utility_start_time, max_utility_start_time+WW[n]):
            r[max_utility_edge][t] = r[max_utility_edge][t] + 1
            p[max_utility_edge][t] = Li * ((Ui/Li) ** (r[max_utility_edge][t] / MAX_CONNECTION_NUM))

        if alpha * REWARD[n][max_utility_edge] > E_n[n] * WT[n] and cos_sim[n][max_utility_edge] > gamma:
            y[n][max_utility_edge] = 1
            allocate_least = k_smallest_indices(rt[max_utility_edge][max_utility_start_time: max_utility_start_time+WW[n]], WT[n], max_utility_start_time)
            for t in range(max_utility_start_time, max_utility_start_time+WW[n]):
                if t in allocate_least:
                    rt[max_utility_edge][t] = rt[max_utility_edge][t] + 1

        obj = obj + f[n][max_utility_edge] + y[n][max_utility_edge] * REWARD[n][max_utility_edge]
        obj_without_reward = obj_without_reward + f[n][max_utility_edge]

'''
# bid first method
for n in range(N):
    max_bid_value = -10000
    capable_edge_index = []
    i_edge_first = np.zeros((I,T), dtype=int)

    # select all satisfied v on each slot t
    for t in range((n),deadline[n]):
        for i in range(I):
            if r_first[i][t] + 1 <= MAX_CONNECTION_NUM:
                i_edge_first[i][t] = 1

    # find maximum bid value (bid first)
    max_bid_start_time = -1
    max_bid_edge = -1
    for i in range(I):
        for t in range((n+WW[n]), deadline[n]):
            if not 0 in i_edge_first[i][t-WW[n]: t]:
                bid_temp = f[n][i]
                if max_bid_value < bid_temp:
                    max_bid_value = bid_temp
                    max_bid_start_time = t-WW[n]
                    max_bid_edge = i

    if max_bid_value > 0:
        x_first[n][i] = 1
        for t in range(max_bid_start_time, max_bid_start_time+WW[n]):
            r_first[max_bid_edge][t] = r_first[max_bid_edge][t] + 1
        if alpha * REWARD[n][max_bid_edge] > E_n[n] * WT[n] and cos_sim[n][max_bid_edge] > gamma:
            y_first[n][max_bid_edge] = 1
            for j in range(len(k_smallest_indices(rt_first[max_bid_edge][max_bid_start_time: max_bid_start_time+WW[n]], WT[n]))):
                rt_first[max_bid_edge][j] = rt_first[max_bid_edge][j] + 1

        obj_first = obj_first + f[n][max_bid_edge] + y_first[n][max_bid_edge] * REWARD[n][max_bid_edge]
'''

# random method (give a single bid, do not consider distribution)
for n in range(N):
    i_edge_random = np.zeros((I,T), dtype=int)

    # select all satisfied v on each slot t
    for t in range((n),deadline[n]):
        for i in range(I):
            if r_random[i][t] + 1 <= MAX_CONNECTION_NUM:
                i_edge_random[i][t] = 1

    # find random schedule
    random_start_time = -1
    break_out = False
    for random_edge in random_traverse(I):
        for t in range((n+WW[n]), deadline[n]):
            if not 0 in i_edge_random[random_edge][t-WW[n]: t] and f[n][random_edge]>payment_greedy[n][random_edge]:
                random_start_time = t-WW[n]
                break_out = True
                break

        if break_out:
            break

    if random_start_time != 1:
        x_random[n][random_edge] = 1
        for t in range(random_start_time, random_start_time+WW[n]):
            r_random[random_edge][t] = r_random[random_edge][t] + 1
        if alpha * REWARD[n][random_edge] > E_n[n] * WT[n] and cos_sim[n][random_edge] > gamma:
            allocate_least = k_smallest_indices(rt_random[random_edge][random_start_time: random_start_time+WW[n]], WT[n], random_start_time)
            for t in range(random_start_time, random_start_time+WW[n]):
                if t in allocate_least:
                    rt_random[random_edge][t] = rt_random[random_edge][t] + 1

        obj_random = obj_random + f_random[n][random_edge] + y_random[n][random_edge] * REWARD[n][random_edge]


# unit greedy
for n in range(N):
    i_edge_greedy = np.zeros((I,T), dtype=int)
    max_unit_utility = 0

    # select all satisfied v on each slot t
    for t in range((n),deadline[n]):
        for i in range(I):
            if r_greedy[i][t] + 1 <= MAX_CONNECTION_NUM:
                i_edge_greedy[i][t] = 1

    # find maximum unit utility (greedy)
    max_unit_utility_start_time = -1
    max_unit_utility_edge = -1
    for i in range(I):
        for t in range((n+WW[n]), deadline[n]):
            if not 0 in i_edge_greedy[i][t-WW[n]: t]:
                max_unit_utility_temp = f[n][i]  - payment_greedy[n][i]
                if max_unit_utility < max_unit_utility_temp:
                    max_unit_utility = max_unit_utility_temp
                    max_unit_utility_start_time = t-WW[n]
                    max_unit_utility_edge = i

    if max_unit_utility > 0:
        x_greedy[n][i] = 1
        for t in range(max_unit_utility_start_time, max_unit_utility_start_time+WW[n]):
            r_greedy[max_unit_utility_edge][t] = r_greedy[max_unit_utility_edge][t] + 1
        if alpha * REWARD[n][max_unit_utility_edge] > E_n[n] * WT[n] and cos_sim[n][max_unit_utility_edge] > gamma:
            y_greedy[n][max_unit_utility_edge] = 1
            allocate_least = k_smallest_indices(rt_greedy[max_unit_utility_edge][max_unit_utility_start_time: max_unit_utility_start_time+WW[n]], WT[n], max_unit_utility_start_time)
            for t in range(max_unit_utility_start_time, max_unit_utility_start_time+WW[n]):
                if t in allocate_least:
                    rt_greedy[max_unit_utility_edge][t] = rt_greedy[max_unit_utility_edge][t] + 1

        obj_greedy = obj_greedy + f[n][max_unit_utility_edge] + y_greedy[n][max_unit_utility_edge] * REWARD[n][max_unit_utility_edge]


print("total obj: " + str(obj))
print("bids accpentance ratio: " + str(ratio_of_rows_with_one(x)))
print("local training accpentance ratio: " + str(ratio_of_rows_with_one(y)))
print("obj without reward: " + str(obj_without_reward))
print("obj / obj without reward: " + str(obj/obj_without_reward))
#print("total obj first: " + str(obj_first))
print("total obj random: " + str(obj_random))
print("total obj greedy: " + str(obj_greedy))