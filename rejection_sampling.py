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

    file = open("clicks_decay_LP_frequentfrequent.txt")
    res = file.read().split(",")
    res = list(map(float, res))
    file.close()
    file = open("reward_decay_LP_frequentfrequent.txt")
    res2 = file.read().split(",")
    res2 = list(map(float, res2))
    plt.plot(res, "r")
    # plt.plot(res2)
    plt.xlabel("Interaction Number")
    plt.title("Average Page Clicks - Interest Decay - High Frequency User")
    plt.legend(["LP Allocation"])
    plt.show()
