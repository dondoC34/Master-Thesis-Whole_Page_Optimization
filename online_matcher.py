import numpy as np
from ads_news import *


class OnlineMatcher:

    def __init__(self, num_of_ad_in_a_page, target_ads_positions, ads_pool):

        if num_of_ad_in_a_page != len(target_ads_positions):
            raise RuntimeError("The number of ads in a page is differs from the number of target positions for ads")
        self.num_of_ad_in_a_page = num_of_ad_in_a_page
        self.target_ads_positions = target_ads_positions
        if not isinstance(ads_pool, list):
            raise RuntimeError("a list of ads is expected as Ads_Pool parameter")
        self.ads_pool = ads_pool
        self.links = []

    def remove_ad_from_pool(self, ad):
        target_index = self.ads_pool.index(ad)
        self.ads_pool.__delitem__(target_index)

    def find_best_ads(self, slot_promenances):

        result = []
        # Matcher reset
        self.links.clear()

        # Links creation
        for ad in self.ads_pool:
            for slot in range(len(slot_promenances)):
                self.links.append([ad, slot, slot_promenances[slot] * ad.sampled_quality])

        best_ads = self.solve_offline_matching_problem()
        best_ads.sort(key=lambda x: x[1], reverse=False)

        for link in best_ads:
            if link[0].is_buyer():
                result.append(link[0])
                self.remove_ad_from_pool(link[0])
                continue

            outcome = np.random.binomial(1, link[0].sampled_quality)
            if outcome == 1:
                result.append(link[0])
                self.remove_ad_from_pool(link[0])
            else:
                link[0].set_as_buyer()

        return result

    def solve_offline_matching_problem(self, approach="greedy"):

        if approach == "greedy":

            matched_slots = []
            matched_ads = []
            result = []
            self.links.sort(key=lambda x: x[2], reverse=True)
            tmp_links = self.links.copy()

            if len(self.ads_pool) < self.num_of_ad_in_a_page:
                raise RuntimeError("Not enough ads to fill a page")

            while len(result) < self.num_of_ad_in_a_page:
                matching_link = tmp_links.pop(0)

                if (matching_link[1] not in matched_slots) and (matching_link[0] not in matched_ads):
                    result.append(matching_link)
                    matched_slots.append(matching_link[1])
                    matched_ads.append(matching_link[0])

            return result

        elif approach == "linear_problem":
            pass

    def fill_ads_pool(self, ads_list, append=True):

        if append:
            for ad in ads_list:
                self.ads_pool.append(ad)
        else:
            self.ads_pool = ads_list.copy()


if __name__ == "__main__":

    ad1 = Ad(1, "politic-1-ad")
    ad2 = Ad(2, "politic-2-ad")
    ad3 = Ad(3, "politic-3-ad")
    ad4 = Ad(6, "tech-1-ad")
    ad5 = Ad(6, "tech-2-ad")
    ad6 = Ad(6, "tech-3-ad")
    ad7 = Ad(6, "tech-4-ad")
    ad8 = Ad(6, "cibo-1-ad")
    ad9 = Ad(6, "cibo-2-ad")
    ad10 = Ad(6, "cibo-3-ad")
    ad11 = Ad(6, "cibo-4-ad")
    ad12 = Ad(6, "cibo-5-ad")

    ad_list = [ad1, ad2, ad3, ad4, ad5, ad6, ad7, ad8, ad9, ad10]

    for ad in ad_list:
        ad.set_sampled_quality(0.1)

    matcher = OnlineMatcher(4, [1, 2, 3, 4], ad_list)
    result_ad = matcher.find_best_ads([0.9, 0.7, 0.5, 0.3])
    for ad in result_ad:
        print(ad.ad_name)
    print(len(matcher.ads_pool))

    for ad in matcher.ads_pool:
        if ad.buyer == 1:
            print(ad.ad_name)


