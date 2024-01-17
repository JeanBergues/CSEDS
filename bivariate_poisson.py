import numpy as np
import math

def pdf_bp(x, y, Lambda):
# Function to calculate the pdf of a bivariate poisson dist
    product_component = np.exp(-(np.sum(Lambda))) * ((Lambda[0]**x)/(math.factorial(x))) * ((Lambda[1]**y)/(math.factorial(y)))

    sum_component = 0
    for k in range(min(x, y)):
        sum_component += math.comb(x, k) * math.comb(y, k) * math.factorial(k) * (((Lambda[3])/(Lambda[1]*Lambda[2]))**k)

    return product_component * sum_component

def ll_biv_poisson(X, Y, Lambda, T, N):
# Function for calculating the log likelihood of bivariate poisson dist
    ll = 0
    for t in range(T):
        sum_1 = 0
        for i in range(N/2):
            sum_1 += np.log(pdf_bp(X[t, i], Y[t, i], Lambda[i,:]))
        ll += sum_1

    return ll
