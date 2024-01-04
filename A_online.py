import numpy as np
import random
from scipy.optimize import linprog

N = 30 # clients
I = 6 # edges
T = N + 20
class_num = 10

Umax = 50
Umin = 1
Li = 1
Ui = 50

x = np.zeros((N,I))
y = np.zeros((N,I))
deadline = np.zeros(N, dtype=int)
obj = 0

u = np.zeros(N)
r = np.zeros((I,T))
rt = np.zeros((I,T))
p = np.ones((I,T))

f = np.zeros((N,I))
REWARD = np.zeros((N,I))
WW = np.zeros(N, dtype=int) # each bid consumes w slots
WT = np.zeros(N, dtype=int) # number of time slots participate in local training
pay = np.zeros(N)

gamma = 0.6 # threshold
alpha = 0.5 # ratio of reward for the clients
d_alpha = 0.5 # Dirichlet Distribution
MAX_CONNECTION_NUM = 3
en = 5.0

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

for n in range(N):
    dead = 0
    bid = np.zeros(I)
    ubid = np.zeros(I)
    reward = np.zeros(I)
    ureward = np.zeros(I)

    for i in range(I):
        w = random.randint(10,20)
        WW[n] = w

    wt = 1000
    while wt > WW[n]:
        wt = random.randint(0,20)
    WT[n] = wt

    composition_ratio = np.random.dirichlet((d_alpha, d_alpha)*(class_num//2))
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
        cos_sim[n][i] = cosine_similarity(D_i[i], composition_ratio)

    bid, reward = custom_sort(cos_sim[n], bid, reward)
    deadline[n] = n + 20
    f[n] = bid
    REWARD[n] = reward

for n in range(N):
    c_temp = np.zeros((I,T))+10000 # record satisfied i t's price
    u_max = -np.zeros(T) # find i on each slot t with maximum utility
    u_max_index = -1 # according i index
    i_edge = np.zeros((I,T), dtype=int)
    max_utility = 0
    schedule_start_time_temp = -1
    schedule_edges_temp = -1

    # select all satisfied v on each slot t
    for t in range((n),deadline[n]):
        for i in range(I):
            if r[i][t] + 1 <= MAX_CONNECTION_NUM:
                i_edge[i][t] = 1

    # find maximum utility
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

print("total obj: " + str(obj))
print("bids accpentance ratio: " + str(ratio_of_rows_with_one(x)))
print("local training accpentance ratio: " + str(ratio_of_rows_with_one(y)))

'''
ff = f.flatten()
ff_reward = REWARD.flatten()
E = np.zeros(N*I*T)
E_reward = np.zeros(N*I*T)
ff = np.concatenate((ff, ff_reward, E, E_reward))

sizeoff = len(ff)

# contraint a
A1 = np.zeros((N*I*T, sizeoff))
b1 = np.zeros(N*I*T)
for n in range(N):
    for i in range(I):
        for t in range(T):
            A1[n*I*T+i*T+t][n*I+i] = -deadline[n]
            if t >= n:
                A1[n*I*T+i*T+t][N*I+N*I+n*I*T+i*T+t] = t

# constraint b
A2 = np.zeros((N*I, sizeoff))
b2 = np.zeros(N*I)
for n in range(N):
    for i in range(I):
        A2[n*I+i][n*I+i] = WW[n]
        for t in range(T):
            if t >= n:
                A2[n*I+i][N*I+N*I+n*I*T+i*T+t] = -1

# constraint c
A3 = np.zeros((N*I, sizeoff))
b3 = np.zeros(N*I)
for n in range(N):
    for i in range(I):
        A3[n*I+i][N*I+n*I+i] = WT[n]
        for t in range(T):
            if t>=n:
                A3[n*I+i][N*I+N*I+N*I*T+n*I*T+i*T+t] = -1

# constraint d
A4 = np.zeros((I*T,sizeoff))
b4 = np.ones(I*T) * MAX_CONNECTION_NUM
for i in range(I):
    for t in range(T):
        for n in range(N):
            A4[i*T+t][N*I+N*I+n*I*T+i*T+t] = 1

# constraint e
A5 = np.zeros((N*I,sizeoff))
b5 = cos_sim.flatten()
for n in range(N):
    for i in range(I):
        A5[n*I+i][N*I+n*I+i] = gamma

# constraint f
A6 = np.zeros((N*I*T,sizeoff))
b6 = np.zeros(N*I*T)
for n in range(N):
    for i in range(I):
        for t in range(T):
            A6[n*I*T+i*T+t][N*I+N*I+n*I*T+i*T+t] = -1
            A6[n*I*T+i*T+t][N*I+N*I+N*I*T+n*I*T+i*T+t] = 1

# constraint g
A7 = np.zeros((N*T,sizeoff))
b7 = np.ones(N*T)
for n in range(N):
    for t in range(T):
        for i in range(I):
            A7[n*T+t][N*I+N*I+n*I*T+i*T+t] = 1

# constraint h
A8 = np.zeros((N*I,sizeoff))
b8 = REWARD.flatten()
for n in range(N):
    for i in range(I):
        for t in range(T):
            if t >= n:
                A8[n*I+i][N*I+N*I+N*I*T+n*I*T+i*T+t] = en


lb = np.zeros(sizeoff)
ub = np.ones(sizeoff)
A = np.concatenate((A1, A2, A3, A4, A5, A6, A7, A8))
b = np.concatenate((b1, b2, b3, b4, b5, b6, b7, b8))
res = linprog(-ff, A_ub=A, b_ub=b, A_eq=None, b_eq=None, bounds=list(zip(lb, ub)), method='interior-point')
fval = -res.fun
print(fval)
print(fval/obj)
'''