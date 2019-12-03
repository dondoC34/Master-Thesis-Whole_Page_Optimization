from news_learner import *
from synthetic_user import *
from ads_news import *
from tqdm import tqdm


class PageCreator:
    def __init__(self, attribute_1, attribute_2, real_slot_promenances, layout_slots, allocation_approach, categories):
        self.categories = categories
        self.real_slot_promenances = real_slot_promenances
        self.attribute_1 = attribute_1
        self.attribute_2 = attribute_2
        self.learner_matrix = []
        for _ in self.attribute_1:
            attribute_row = []
            for _ in attribute_2:
                attribute_row.append(NewsLearner(real_slot_promenances=real_slot_promenances,
                                                 layout_slots=layout_slots,
                                                 allocation_approach=allocation_approach,
                                                 categories=categories))

            self.learner_matrix.append(attribute_row.copy())

    def fill_all_news_pool(self, pool):
        for learner_row in self.learner_matrix:
            for learner in learner_row:
                learner.fill_news_pool(pool, append=True)

    def user_interaction(self, user):
        index_1 = self.attribute_1.index(user.genre)
        index_2 = self.attribute_2.index(user.age_slot)
        self.learner_matrix[index_1][index_2].user_arrival(user=user, interest_decay=True)


if __name__ == "__main__":
    user_pool = []
    news_pool = []
    user_genres = ["M", "F"]
    user_age = [j for j in range(10, 91)]

    num_of_users = 30
    num_of_news_per_category = 1000
    num_of_interaction = 8000

    for i in range(num_of_users):
        user_pool.append(SyntheticUser(i, np.random.choice(user_genres), np.random.choice(user_age), "C"))
    k = 0
    for category in ["cibo", "gossip", "politic", "scienza", "sport", "tech"]:
        for id in range(num_of_news_per_category):
            news_pool.append(News(news_id=k,
                                  news_name=category + "-" + str(id)))
            k += 1
    site = PageCreator(attribute_1=["M", "F"],
                       attribute_2=["LOW", "MEDIUM", "HIGH"],
                       real_slot_promenances=[0.7, 0.8, 0.3, 0.5, 0.3],
                       layout_slots=5,
                       allocation_approach="LP",
                       categories=["cibo", "gossip", "politic", "scienza", "sport", "tech"])

    site.fill_all_news_pool(news_pool)

    # for i in range(len(site.learner_matrix)):
    #     for j in range(len(site.learner_matrix[i])):
    #         site.learner_matrix[i][j].read_weighted_beta_matrix_from_file(indexes=[(0, 0), (0, 1), (0, 2), (1, 1),
    #                                                                                (1, 2)],
    #                                                                       desinences=[str(i) + "-" + str(j)] * 5)
    #
    # site.learner_matrix[1][0].weighted_betas_matrix[0][0].plot_distribution("politic")
    # site.learner_matrix[1][2].weighted_betas_matrix[1][1].plot_distribution("tech")
    # exit(34)

    for _ in tqdm(range(num_of_interaction)):
        user = np.random.choice(user_pool)
        site.user_interaction(user=user)

    for i in range(len(site.learner_matrix)):
        for j in range(len(site.learner_matrix[i])):
            site.learner_matrix[i][j].save_weighted_beta_matrices(desinence=str(i) + "--" + str(j))





