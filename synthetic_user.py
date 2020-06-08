import numpy as np
from news_learner import *

categories = ["cibo", "gossip", "politic", "scienza", "sport", "tech"]
age_classes = ["LOW", "MEDIUM", "HIGH"]
genre_classes = ["M", "F"]
# THE FOLLOWING ARE THE MEANS AND THE VARIANCE OF NORMAL DISTRIBUTIONS USED, FOR EACH COMBINATION OF FEATURES, TO
# RANDOMLY EXTRACT VALUES OF THE PARAMETERS FOR EACH CREATED USER
quality_per_age_values = [[0.2, 0.4, 0.2, 0.4, 0.8, 0.6],
                          [0.4, 0.3, 0.6, 0.5, 0.7, 0.5],
                          [0.5, 0.5, 0.7, 0.4, 0.6, 0.3]]
quality_per_genre_values = [[0.6, 0.2, 0.7, 0.5, 0.8, 0.6],
                            [0.5, 0.7, 0.5, 0.5, 0.3, 0.5]]
quality_per_age_var = [[0.2, 0.1, 0.1, 0.2, 0.2, 0.2],
                       [0.1, 0.2, 0.2, 0.15, 0.1, 0.25],
                       [0.1, 0.1, 0.2, 0.1, 0.2, 0.3]]
quality_per_genre_var = [[0.1, 0.1, 0.1, 0.2, 0.1, 0.1],
                         [0.2, 0.1, 0.2, 0.2, 0.2, 0.1]]

quality_per_age_values_ads = [[0.1, 0.15, 0.2, 0.15, 0.05, 0.1],
                              [0.1, 0.15, 0.05, 0.1, 0.1, 0.05],
                              [0.03, 0.03, 0.035, 0.015, 0.1, 0.025]]
quality_per_genre_values_ads = [[0.01, 0.01, 0.015, 0.02, 0.19, 0.15],
                                [0.1, 0.2, 0.15, 0.02, 0.015, 0.01]]
quality_per_age_var_ads = [[0.02, 0.01, 0.01, 0.02, 0.02, 0.02],
                           [0.01, 0.02, 0.02, 0.015, 0.01, 0.025],
                           [0.01, 0.01, 0.02, 0.01, 0.02, 0.03]]
quality_per_genre_var_ads = [[0.01, 0.01, 0.01, 0.02, 0.01, 0.01],
                             [0.02, 0.01, 0.02, 0.02, 0.02, 0.01]]


class SyntheticUser:
    def __init__(self, user_id, genre, age):
        self.user_id = user_id
        self.last_news_clicked = []
        self.last_news_in_allocation = []
        self.viewed_but_not_clicked_news = []
        self.genre = genre
        self.categories = categories
        self.user_quality_measure = []
        self.user_quality_measure_for_ads = []
        # Depending on the age, we split the users in 3 categories
        self.age_slot = "LOW"
        if (age > 20) and (age < 50):
            self.age_slot = "MEDIUM"
        elif age > 50:
            self.age_slot = "HIGH"

        age_index = age_classes.index(self.age_slot)
        genre_index = genre_classes.index(self.genre)

        # THE VALUES ABOVE ARE USED TO CHOOSE THE PROBABLITY OF CLICK OF EACH USER, BASING ON THEIR ATTRIBUTES.
        # SOME NOISE IS ADDED BY MEANS OF GAUSSIAN DISTRIBUTIONS.
        self.user_quality_measure = np.abs(np.random.normal(np.mean([quality_per_age_values[age_index], quality_per_genre_values[genre_index]], axis=0),
                                                            np.mean([quality_per_age_var[age_index], quality_per_genre_var[genre_index]], axis=0)))

        self.user_quality_measure_for_ads = np.abs(np.random.normal(np.mean([quality_per_age_values_ads[age_index], quality_per_genre_values_ads[genre_index]], axis=0),
                                                                    np.mean([quality_per_age_var_ads[age_index], quality_per_genre_var_ads[genre_index]], axis=0)))

        for i in range(len(self.user_quality_measure)):
            # EACH VALUE IS CLAMPED TO BE GREATER OF EQUAL THAN 0.1, AND LESS THEN OR EQUAL THAN 0.9
            value = np.min([0.9, np.max([0.1, self.user_quality_measure[i]])])
            self.user_quality_measure[i] = value

        # This will be used in case of context generation
        self.personal_learner = None

    def get_amount_of_clicks(self, news, get_only_index=False):
        """
        Return the number of times this user clicked the article "news". Looks for the article using a binary search
        :param news: News object
        :param get_only_index: True if only the index of the article is required. Used to access its position in the list
        of all the news in this user's cookie
        :return: The number of clicks or the index
        """
        news_id = news.news_id
        left = 0
        right = len(self.last_news_clicked)

        while left < right:

            mid = int(left + (right - left) / 2)
            if self.last_news_clicked[mid][0] == news_id:
                if get_only_index:
                    return mid
                else:
                    return self.last_news_clicked[mid][1]
            elif self.last_news_clicked[mid][0] < news_id:
                left = mid + 1
            else:
                right = mid

        if get_only_index:
            return -1
        else:
            return 0

    def get_promenance_cumsum(self, news, get_only_index=False):
        """
            Return the sumsum of the slots' prominences in which the article "news" has been allocated for this user.
            Looks for the article using a binary search
            :param news: News object
            :param get_only_index: True if only the index of the article is required. Used to access its position in the list
            of all the news in this user's cookie
            :return: The cumsum or the index
        """
        news_id = news.news_id
        left = 0
        right = len(self.last_news_in_allocation)

        while left < right:

            mid = int(left + (right - left) / 2)
            if self.last_news_in_allocation[mid][0] == news_id:
                if get_only_index:
                    return mid
                else:
                    return self.last_news_in_allocation[mid][1]
            elif self.last_news_in_allocation[mid][0] < news_id:
                left = mid + 1
            else:
                right = mid

        if get_only_index:
            return -1
        else:
            return 0

    def click_news(self, news, interest_decay=False):
        """
        Returns a binary variable that denotes if the user clicked or not the article news according to its parameters,
        then updates the user's state.
        :param news: News object
        :param interest_decay: If true, the user penalizes articles he has already seen or already clicked
        :return: 1 if clicked, 0 otherwise
        """
        category_index = self.categories.index(news.news_category)

        if interest_decay:
            num_of_clicks = self.get_amount_of_clicks(news)
            interest_decay_factor = np.exp(- num_of_clicks)
            interest_decay_factor_2 = np.exp(- 0.5 * self.viewed_but_not_clicked_news.count(news))
            click = np.random.binomial(1, interest_decay_factor * interest_decay_factor_2 *
                                       self.user_quality_measure[category_index])
        else:
            click = np.random.binomial(1, self.user_quality_measure[category_index])

        index = self.get_amount_of_clicks(news, get_only_index=True)

        if (click == 1) and interest_decay and (index == -1):
            length = len(self.last_news_clicked)
            inserted = False
            if length >= 2:
                for k in range(length - 1):
                    if (news.news_id > self.last_news_clicked[k][0]) and (news.news_id < self.last_news_clicked[k+1][0]):
                        self.last_news_clicked.insert(k + 1, [news.news_id, 0, 1])
                        inserted = True
                if not inserted:
                    if news.news_id < self.last_news_clicked[0][0]:
                        self.last_news_clicked.insert(0, [news.news_id, 0, 1])
                        inserted = True
            elif length == 1:
                if self.last_news_clicked[0][0] > news.news_id:
                    self.last_news_clicked.insert(0, [news.news_id, 0, 1])
                    inserted = True

            if not inserted:
                self.last_news_clicked.append([news.news_id, 0, 1])

        elif (click == 0) and interest_decay and (self.viewed_but_not_clicked_news.count(news) <= 2):
            self.viewed_but_not_clicked_news.append(news)

        return click

    def click_ad(self, ad):
        """
        Returns a binary variable denoting if the user clicked or not the ad "ad"
        :param ad: Ad object
        :return: 1 if the user clicked, 0 otherwise
        """
        category_index = self.categories.index(ad.ad_category)
        click = np.random.binomial(1, self.user_quality_measure_for_ads[category_index])
        return click

    def get_reward(self, news):
        """
        Return the parameter of the user associated to the article "news"
        :param news: News object
        :return: the probability of the user clicking "news"
        """
        num_of_clicks = next((x[1] for x in self.last_news_clicked if x[0] == news.news_id), 0)
        interest_decay_factor = np.exp(- num_of_clicks)
        interest_decay_factor_2 = np.exp(- 0.5 * self.viewed_but_not_clicked_news.count(news))
        category_index = self.categories.index(news.news_category)

        return interest_decay_factor * interest_decay_factor_2 * self.user_quality_measure[category_index]







