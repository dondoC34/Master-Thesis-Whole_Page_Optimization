import scipy.optimize as opt
import numpy as np


class News:

    def __init__(self, news_id, news_name):

        self.news_id = news_id
        self.news_name = news_name
        self.news_category = news_name.split("-")[0]
        self.sampled_quality = 0
        self.image_path = "News-AdsApp - Copia/" + news_name + ".gif"
        self.slot_promenance_cumsum = 0
        self.click_sum = 0
        self.prova = [1, 2, 3]

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

    a = [np.random.normal(0, 20, size=30)]
    mean = np.mean(a)
    var = np.sqrt(np.var(a))
    n = len(a)
    mean0 = 0

    T = (mean - mean0) / (var / np.sqrt(n))
    print(T)





