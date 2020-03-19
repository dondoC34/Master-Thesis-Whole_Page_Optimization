import scipy.optimize as opt
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys


class News:

    def __init__(self, news_id, news_name):

        self.news_id = news_id
        self.news_name = news_name
        self.news_category = news_name.split("-")[0]
        self.sampled_quality = 0
        self.image_path = "News-AdsApp-Copia/" + news_name + ".gif"
        self.slot_promenance_cumsum = 0
        self.click_sum = 0
        self.prova = [1, 2, 3]
        self.doubled_news_indexes = [-1, -1]

    def set_sampled_quality(self, value):
        self.sampled_quality = value


class Ad:

    def __init__(self, ad_id, ad_name, exclude_competitors=False):

        self.ad_id = ad_id
        self.ad_name = ad_name
        self.buyer = 0
        self.sampled_quality = 0
        self.exclude_competitors = exclude_competitors
        self.ad_category = ad_name.split("-")[0]
        self.image_path = "Ads-AdsApp/" + ad_name + ".gif"
        self.bid = np.random.uniform(0, 1)

    def set_sampled_quality(self, value):
        self.sampled_quality = value

    def set_as_buyer(self):
        self.buyer = 1

    def exclude_competitors(self):
        if self.exclude_competitors:
            return 1
        else:
            return 0

    def is_buyer(self):
        if self.buyer == 0:
            return False
        else:
            return True


if __name__ == "__main__":

    file = open("regret_M_27_C3_N300_L5_FI0.txt", "r")
    res = file.read().splitlines()
    best = float(res[1])
    res = res[0].split(",")
    res = list(map(float, res))
    res = np.cumsum(best - np.array(res))
    logg = [0]
    for i in range(1, len(res)):
        logg.append(1.5 * np.log(i))
    logg = np.array(logg) + res[0]
    plt.plot(res, "r")
    plt.plot(logg)
    plt.show()






