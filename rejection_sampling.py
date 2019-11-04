import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad

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

    def integrand(x, a, b):
        return a * x ** 2 + b


    a = 2
    b = 1
    I = quad(integrand, 0, 1, args=(a, b))
    print(I)

