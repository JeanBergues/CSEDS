import numpy as np
import math

def create_A_B_matrix(a, b, N):
    # Creates first block
    a_block = a * np.eye(N)

    # Create secopnd block
    b_block = b * np.eye(N)

    # Combine the blocks to form matrix
    matrix = np.block([[a_block, np.zeros((N, N))],
                  [np.zeros((N, N)), b_block]])

    return matrix

def pdf_bp(x, y, alpha_it, alpha_jt, beta_jt, beta_it, lambda3, delta):
# Function to calculate the pdf of a bivariate poisson dist
    lambda1 = np.exp(delta + alpha_it - beta_jt)
    lambda2 = np.exp(alpha_jt - beta_it)
    product_component = np.exp(-(lambda1+lambda2+lambda3)) * ((lambda1**x)/(math.factorial(x))) * ((lambda2**y)/(math.factorial(y)))

    sum_component = 0
    for k in range(min(x, y)):
        sum_component += math.comb(x, k) * math.comb(y, k) * math.factorial(k) * (((lambda3)/(lambda1*lambda2))**k)

    return product_component * sum_component

def ll_biv_poisson(X, Y, a1, a2, b1, b2, lambda3, delta):
# Function for calculating the log likelihood of bivariate poisson dist
    # Length of data
    N = len(X)
    T = len(X[0])

    # Setting log likelihood to 0 first
    ll = 0

    # Defining f_t, A, B
    f = []
    A = create_A_B_matrix(a1,a2, N)
    B = create_A_B_matrix(b1,b1, N)

    # Calculating likelihood
    for t in range(T):
        if t == 0:
            f[t] = calc_ini_f
            w = np.multiply(f[t], (np.ones(len(f[t])) - np.diagonal(B)))
        else:
            f[t] = w + B * f[t] + A * s[t]

        sum_1 = 0
        for i in range(N/2):
            sum_1 += np.log(pdf_bp(X[t, i], Y[t, i], f[t][i], alpha_jt, beta_jt, beta_it, delta, lambda3)) # (X, Y, alpha_it, alpha_jt, beta_jt, beta_it, delta, lambda3)
        ll += sum_1

    return ll

def calc_ini_f():
    return

def train_model_bp(X, Y):
    f = []
    f.append(calc_ini_f())




