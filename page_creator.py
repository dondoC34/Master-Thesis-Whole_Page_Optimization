from news_learner import *
from synthetic_user import *
from ads_news import *
from tqdm import tqdm


class PageCreator:
    def __init__(self, attributes_1, attributes_2, real_slot_promenances, layout_slots, allocation_approach, categories):
        self.categories = categories
        self.real_slot_promenances = real_slot_promenances
        # THE FOLLOWING TWO PARAMETERS INCLUDE THE FEATURES OF THE USERS
        self.attribute_1 = attributes_1
        self.attribute_2 = attributes_2
        self.learner_matrix = []
        # CREATES A MATRIX OF LEARNERS, THE MATRIX SIZE IS NxM, WHERE N IS THE DIMENSION OF THE FIRST CLASS OF FEATURES,
        # M THE DIMENSION OF THE SECOND ONE.
        for _ in self.attribute_1:
            attribute_row = []
            for _ in attributes_2:
                attribute_row.append(NewsLearner(real_slot_promenances=real_slot_promenances,
                                                 layout_slots=layout_slots,
                                                 allocation_approach=allocation_approach,
                                                 categories=categories))

            self.learner_matrix.append(attribute_row.copy())

    def fill_all_news_pool(self, pool):
        """
        Fills the news pool of each learner in the learners matrix.
        :param pool: List of news to fill the news pools
        :return: Nothing
        """
        for learner_row in self.learner_matrix:
            for learner in learner_row:
                learner.fill_news_pool(pool, append=True)

    def user_interaction(self, user):
        """
        Handles an user interaction. Basing on the user features, classifies the user and assign it to the corresponding
        learner in the learners matrix.
        :param user: The user ifself.
        :return: Nothing
        """
        index_1 = self.attribute_1.index(user.genre)
        index_2 = self.attribute_2.index(user.age_slot)
        self.learner_matrix[index_1][index_2].user_arrival(user=user, interest_decay=True)


if __name__ == "__main__":
    user_pool = []
    news_pool = []
    user_genres = ["M", "F"]
    user_age = [j for j in range(10, 91)]

    num_of_users = 30
    num_of_news_per_category = 400
    num_of_interaction = 2000

    # USE WHICHEVER SLOT PROMENANCE VALUE, FEASIBLE OF COURSE (>0 AND <1)
    real_slot_promenances = [0.7, 0.8, 0.7, 0.7, 0.6, 0.5, 0.5, 0.4, 0.3, 0.2]

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
    site = PageCreator(attributes_1=["M", "F"],
                       attributes_2=["LOW", "MEDIUM", "HIGH"],
                       real_slot_promenances=real_slot_promenances,
                       layout_slots=10,
                       allocation_approach="LP",
                       categories=["cibo", "gossip", "politic", "scienza", "sport", "tech"])

    # FILL ALL THE NEWS POOLS
    site.fill_all_news_pool(news_pool)

    # SELECT A RANDOM USER AND MAKE IT INTERACT WITH THE SITE. REPEAT FOR NUM_OF_INTERACTIONS TIMES.
    for _ in tqdm(range(num_of_interaction)):
        user = np.random.choice(user_pool)
        site.user_interaction(user=user)

    # SAVE THE TRAINED WEIGHTED BETAS MATRIX OF EACH LEARNER.
    for i in range(len(site.learner_matrix)):
        for j in range(len(site.learner_matrix[i])):
            site.learner_matrix[i][j].save_weighted_beta_matrices(desinence=str(i) + "-" + str(j) + "de_rand_testing")





