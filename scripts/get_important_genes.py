import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.io
import cvxpy as cp
import os
import sys
import pickle

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
from functions.control_helpers import opt_genes
from scipy.optimize import minimize

genes = scipy.io.loadmat('./Brain_data_400/gene_coexpression/ParcellatedGeneExpressionLRHemiSchaefer17Network400.mat')
names = [x[0][0] for x in genes['gene_names']]
# save names as csv
names_df = pd.DataFrame({'ID': names})
names_df.to_csv('./data/gene_ids.csv')

# get coexp
exp = genes['LeftHemiParcelExpression']
# remove missing values from brain data
data_idx = [not x for x in np.isnan(sum(exp.T))]
# demean so its easier to get covariance matrix
exp = exp[data_idx, :]
A = np.corrcoef(exp)

# intitialize
top_genes = pd.DataFrame()
n = np.shape(A)[0]
m = np.shape(exp)[1]
nVec = np.ones((n, 1))
mVec = np.ones((m, 1))

# note from Peter Mucha
#  let b_i be the coefficient weighting the data of the ith gene and
#  B the correlation matrix you get computing the co-expression from the corresponding weighted-by-b gene expression.
#  Then define an objective function expressing the squared differences between B and the true A co-expression matrix
#  plus an l1 penalty to promote sparsity on b. That is, something like
#     Phi(b) = ||B(b)-A||_2 + lambda||b||_1

# Create two scalar optimization variables.
b = cp.Variable((m, 1), nonneg=True)
B = cp.Variable((n, n), PSD=True)

# set lambda
lam = cp.Parameter(nonneg=True)
lams = np.logspace(-6, -1)

# set up eqs for correlation matrix
# D = np.diag( np.sqrt( np.mean((exp - np.mean(exp, axis=0))**2, axis=0) ) )
X = cp.multiply(exp.T, b)
mu = cp.reshape(cp.sum(X, axis=0) / m, (n, 1)).T
D = cp.diag(cp.sqrt(cp.sum((X - mu) ** 2, axis=0) / m))
C = np.eye(m) - (1 / m) * mVec * mVec.T
Xs = C @ X @ D ** -1

# constraints
constraints = [b <= 1, lam <= 1, B == (1 / m) * Xs.T @ Xs]

# Form objective.
error = cp.norm2(B - A)
obj = cp.Minimize(error + lam * cp.norm1(b))

# Form and solve problem.
prob = cp.Problem(obj, constraints)
sq_penalty = []
l1_penalty = []
b_values = []
for val in lams:
    lam.value = val
    prob.solve()
    # Use expr.value to get the numerical value of
    # an expression in the problem.
    sq_penalty.append(error.value)
    l1_penalty.append(cp.norm(b, 1).value)
    b_values.append(b.value)

prob.status

# try with scipy
lams = np.logspace(1, 6)
# make array of starting conditions
x0 = np.ones((m, 1)) * 0.1
idx = np.random.choice(list(range(m)), 5000)
x0[idx, :] = 0
res = minimize(opt_genes, x0, args=(1000, exp, A), constraints=({'type': 'ineq', "fun": lambda x: x}), tol=3)

pickle.dump(res, "./data/subgrp_coexp/min_res.pkl")

