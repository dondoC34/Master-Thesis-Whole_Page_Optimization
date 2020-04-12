from news_learner import *
from synthetic_user import *
from ads_news import *
from tqdm import tqdm


class PageCreator:
    def __init__(self, attributes_1, attributes_2, real_slot_promenances, layout_slots, allocation_approach, categories):
        self.categories = categories
        self.real_slot_promenances = real_slot_promenances
        self.learners_list = []
        # THE FOLLOWING TWO PARAMETERS INCLUDE THE FEATURES OF THE USERS
        self.attribute_1 = attributes_1
        self.attribute_2 = attributes_2
        self.fraction_of_ads_clicks = []
        self.total_ads_clicks = 0
        self.total_ads_allocations = 0
        self.total_ads_assignment = []
        self.average_reward = []
        self.learner_matrix = []
        # CREATES A MATRIX OF LEARNERS, THE MATRIX SIZE IS NxM, WHERE N IS THE DIMENSION OF THE FIRST CLASS OF FEATURES,
        # M THE DIMENSION OF THE SECOND ONE.
        for _ in self.attribute_1:
            attribute_row = []
            for _ in attributes_2:
                attribute_row.append(NewsLearner(real_slot_promenances=real_slot_promenances,
                                                 layout_slots=layout_slots,
                                                 allocation_approach=allocation_approach,
                                                 categories=categories,
                                                 allocation_diversity_bounds=(0.4, 0.4, 0.4, 0.4, 0.4, 0.4),
                                                 news_column_pivot=[0.01, 2],
                                                 ads_allocation=True,
                                                 ads_allocation_technique="res_LP",
                                                 ads_allocation_approach="pdda",
                                                 ))

            self.learner_matrix.append(attribute_row.copy())

        for row in self.learner_matrix:
            for learner in row:
                self.learners_list.append(learner)

        for i in range(len(self.learners_list) - 1):
            self.learners_list[i].other_classes_learners = self.learners_list[0:i] + self.learners_list[i+1::]

        self.learners_list[-1].other_classes_learners = self.learners_list[0:len(self.learners_list) - 1]

    def fill_all_news_pool(self, pool, append=False):
        """
        Fills the news pool of each learner in the learners matrix.
        :param pool: List of news to fill the news pools
        :return: Nothing
        """
        for learner_row in self.learner_matrix:
            for learner in learner_row:
                learner.fill_news_pool(pool, append=append)

    def read_all_ads_weighted_beta(self, folder):

        for i in range(len(self.learner_matrix)):
            for j in range(len(self.learner_matrix[i])):
                self.learner_matrix[i][j].read_ads_weighted_beta_matrix_from_file(desinence="_" + str(i)+"-"+str(j),
                                                                                  folder=folder)

    def fill_all_ads_pool(self, pool, append=False):
        for learner_row in self.learner_matrix:
            for learner in learner_row:
                learner.fill_ads_pool(pool, append=append)

    def refresh_all_ads_list(self, ads_list):
        for learner_row in self.learner_matrix:
            for learner in learner_row:
                learner.refresh_ads_buffer(ads_list=ads_list)

    def user_interaction(self, user, debug=False):
        """
        Handles an user interaction. Basing on the user features, classifies the user and assign it to the corresponding
        learner in the learners matrix.
        :param user: The user ifself.
        :return: Nothing
        """
        index_1 = self.attribute_1.index(user.genre)
        index_2 = self.attribute_2.index(user.age_slot)
        self.learner_matrix[index_1][index_2].user_arrival(user=user, interest_decay=True, debug=debug)
        self.average_reward.append(self.learner_matrix[index_1][index_2].multiple_arms_avg_reward[-1])
        self.total_ads_clicks += self.learner_matrix[index_1][index_2].total_ads_clicks_and_displays[-1][0]
        self.total_ads_allocations += self.learner_matrix[index_1][index_2].total_ads_clicks_and_displays[-1][1]
        self.total_ads_assignment.append(self.total_ads_allocations)
        try:
            self.fraction_of_ads_clicks.append(self.total_ads_clicks / self.total_ads_allocations)
        except ZeroDivisionError:
            self.fraction_of_ads_clicks.append(0)


if __name__ == "__main__":
    user_pool = []
    news_pool = []
    ads_pool = []
    user_genres = ["M", "F"]
    user_age = [j for j in range(10, 91)]

    num_of_users = 1000
    num_of_news_per_category = 3
    num_of_ads_per_category = 1005
    num_of_interaction = 500

    # USE WHICHEVER SLOT PROMENANCE VALUE, FEASIBLE OF COURSE (>0 AND <1)
    real_slot_promenances = [0.9, 0.8, 0.7]

    for i in range(num_of_users):
        # FILL THE USER POOL
        user_pool.append(SyntheticUser(i, np.random.choice(user_genres), np.random.choice(user_age), "C"))

    k = 0
    # CREATE A BOUNCH OF NEWS
    for category in ["cibo", "gossip", "politic", "scienza", "sport", "tech"]:
        for id in range(num_of_news_per_category):
            news_pool.append(News(news_id=k,
                                  news_name=category + "-" + str(id)))
            k += 1

    k = 0
    # CREATE A BOUNCH OF ADS
    for category in ["sport"]:
        for id in range(num_of_ads_per_category):
            ads_pool.append(Ad(k, category + "-" + str(id), np.random.choice([False])))
            k += 1
    support_ads_list = []
    ads_refill = False

    # SELECT A RANDOM USER AND MAKE IT INTERACT WITH THE SITE. REPEAT FOR NUM_OF_INTERACTIONS TIMES.
    debug = False
    click_result = []
    ads_assign = []
    site_avg_reward = []
    for w in tqdm(range(2500)):

        site = PageCreator(attributes_1=["M", "F"],
                           attributes_2=["LOW", "MEDIUM", "HIGH"],
                           real_slot_promenances=real_slot_promenances,
                           layout_slots=3,
                           allocation_approach="standard",
                           categories=["cibo", "gossip", "politic", "scienza", "sport", "tech"])

        # FILL ALL THE NEWS POOLS
        site.fill_all_news_pool(news_pool)
        site.fill_all_ads_pool(ads_pool.copy())
        site.read_all_ads_weighted_beta("Saved-Ads-W-Beta/")
        for user in user_pool:
            user.last_news_clicked.clear()
            user.last_news_in_allocation.clear()
            user.viewed_but_not_clicked_news.clear()

        for o in range(1, num_of_interaction):

            if ads_refill and ((o % 10) == 0):
                for category in ["cibo", "gossip", "politic", "scienza", "sport", "tech"]:
                    for id in range(5):
                        support_ads_list.append(Ad(k, category + "-" + str(id), np.random.choice([True, False])))
                        k += 1
                ads_pool += support_ads_list
                site.refresh_all_ads_list(support_ads_list)
                support_ads_list.clear()

            user = np.random.choice(user_pool)
            site.user_interaction(user=user)

        click_result.append(site.fraction_of_ads_clicks)
        ads_assign.append(site.total_ads_assignment)
        site_avg_reward.append(site.average_reward)

    # file = open("site-performances/site_avg_reward.txt", "w")
    # site_avg_reward = np.mean(site_avg_reward, axis=0)
    # file.write(str(site_avg_reward[0]))
    # for i in range(1, len(site_avg_reward)):
    #     file.write("," + str(site_avg_reward[i]))
    # file.close()

    result = []
    for i in [2 * k for k in range(1, 1001)]:
        tmp = []
        for j in range(len(click_result)):
            for m in range(len(click_result[j])):
                if ads_assign[j][m] == i:
                    if click_result[j][m] not in tmp:
                        tmp.append(click_result[j][m])

        if len(tmp) > 0:
            result.append(np.mean(tmp))
        else:
            result.append(-1)

    click_result = np.mean(result, axis=0)
    print(ads_assign[0])
    file = open("Ads-wpdda-perf/PDDA", "w")
    file.write(str(result[0]))
    for i in range(1, len(result)):
        file.write("," + str(result[i]))
    file.close()


