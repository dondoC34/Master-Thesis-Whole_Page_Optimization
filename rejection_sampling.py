import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad
import time as time

"""
This python file is just a draft i used to perform rejection sampling before adding it in the learner
"""

def target(x):
    return x**10 * (1 - x)**12


def proposal(x, weight):
    return weight * st.beta(2, 3).pdf(x)


def rejection_sampling(iter=100000):
    result = []
    file = open("ciao.txt", "w")

    for k in range(iter):
        if k % 10000 == 0:
            print(k)
        a = np.random.beta(a=2, b=4)
        b = np.random.uniform(0, proposal(a, 1.1))

        if b < target(a):
            result.append(a)
            file.write(str(a) + "\n")

    file.close()
    return result


if __name__ == "__main__":

    a = [[1, 2, 3], [5, 1, 3], [6, 4, 2], [7, 2, 5]]
    a.sort(key=lambda x: x[2] * x[1], reverse=False)
    print(a)
