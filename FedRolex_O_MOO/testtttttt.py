import numpy as np
import quadprog
import torch as torch

"""
    U is a list of gradients (stored as state_dict()) from n users
"""
U = [[1,0,0],[2,3,0],[6,6,0],[0,0,0]]
U = U + 1e-6*np.ones(np.shape(U))
print(U)
U = torch.tensor(U)
epsilon = 0.05
n = len(U)
K = np.eye(n, dtype=float)
for i in range(n):
    for j in range(n):
        K[i, j] = 0
        K[i, j] = torch.mul(U[i], U[j]).sum()
Q = 0.5 * (K + K.T)
I = np.eye(Q.shape[0])
Q = Q + 1e-6 * I
p = np.zeros(n, dtype=float)
a = np.ones(n, dtype=float).reshape(-1, 1)
Id = np.eye(n, dtype=float)
neg_Id = -1. * np.eye(n, dtype=float)
lower_b = (1. / n - epsilon) * np.ones(n, dtype=float)
upper_b = (-1. / n - epsilon) * np.ones(n, dtype=float)
A = np.concatenate((a, Id, Id, neg_Id), axis=1)
b = np.zeros(n + 1)
b[0] = 1.
b_concat = np.concatenate((b, lower_b, upper_b))
alpha = quadprog.solve_qp(Q, p, A, b_concat, meq=1)[0]
print(alpha)