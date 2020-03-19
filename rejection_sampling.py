import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from news_learner import *
from Line_Smoother import *
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

    categories = ["cibo", "gossip", "politic", "scienza", "sport", "tech"]
    real_slot_promenances = [0.156, 0.53, 0.245, 0.16, 0.23, 0.22, 0.654, 0.1, 0.3]

    learner_rand_1 = NewsLearner(categories=categories, layout_slots=len(real_slot_promenances),
                                 real_slot_promenances=real_slot_promenances,
                                 allocation_approach="LP",
                                 lp_rand_technique="rand_1",)
    learner_rand_1.read_weighted_beta_matrix_from_file(indexes=[[0, 0], [0, 2], [1, 2]],
                                                desinences=["0-2test1_10kiter_1kusers"]*3,
                                                folder="Saved-News_W-Beta/")

    learner_rand_1.weighted_betas_matrix[0][0].plot_distribution("sport")
    learner_rand_1.weighted_betas_matrix[0][2].plot_distribution("sport")
    learner_rand_1.weighted_betas_matrix[1][2].plot_distribution("sport")

