import numpy as np
from news_learner import *
from pulp import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
from pulp import *
from scipy.interpolate import interp1d

slot_number = 10
categories_number = 6
categories = ["cibo", "gossip", "politic", "scienza", "sport", "tech"]
age_classes = ["LOW", "MEDIUM", "HIGH"]
genre_classes = ["M", "F"]
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
                              [0.1, 0.15, 0.05, 0.1, 0.2, 0.05],
                              [0.03, 0.03, 0.035, 0.015, 0.1, 0.025]]
quality_per_genre_values_ads = [[0.1, 0.1, 0.015, 0.02, 0.1, 0.15],
                                [0.15, 0.02, 0.1, 0.02, 0.015, 0.1]]
quality_per_age_var_ads = [[0.02, 0.01, 0.01, 0.02, 0.02, 0.02],
                           [0.01, 0.02, 0.02, 0.015, 0.01, 0.025],
                           [0.1, 0.1, 0.2, 0.1, 0.2, 0.3]]
quality_per_genre_var_ads = [[0.01, 0.01, 0.01, 0.02, 0.01, 0.01],
                             [0.02, 0.01, 0.02, 0.02, 0.02, 0.01]]


class SyntheticUser:
    def __init__(self, user_id, genre, age, attention_bias_category):
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
        self.attention_function = []
        # Depending on the attention bias category, we define a function that maps slot promenances to real
        # probabilities that the user will observe the slots
        if attention_bias_category == "A":
            self.attention_function = [0.5] * slot_number
        elif attention_bias_category == "B":
            self.attention_function = [0.8] * slot_number
        elif attention_bias_category == "C":
            self.attention_function = [1] * slot_number
        elif attention_bias_category == "D":
            for i in range(slot_number):
                self.attention_function.append(np.max([0, 1 - i * 0.1]))
        elif attention_bias_category == "E":
            for i in range(slot_number):
                self.attention_function.append(np.max([0, 1 - i * 0.2]))
        else:
            for i in range(slot_number):
                self.attention_function.append(np.min(1, np.max([0, np.random.normal(0.5, 2)])))
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

    # Returns the probabilities that a user observe slots given their attention bias category
    def observation_probabilities(self, slot_promenances):
        return slot_promenances * np.array(self.attention_function)

    # To be used in case of context generation....
    def assign_learner_with_data(self, data_matrix):
        pass

    def get_amount_of_clicks(self, news, get_only_index=False):
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

    # Returns the reward obtained by showing the news "news" to the user keeping into account the number of times
    # the user already clicked the news
    def click_news(self, news, interest_decay=False):
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

        category_index = self.categories.index(ad.ad_category)
        click = np.random.binomial(1, self.user_quality_measure_for_ads[category_index])
        return click

    def get_reward(self, news):
        num_of_clicks = next((x[1] for x in self.last_news_clicked if x[0] == news.news_id), 0)
        interest_decay_factor = np.exp(- num_of_clicks)
        interest_decay_factor_2 = np.exp(- 0.5 * self.viewed_but_not_clicked_news.count(news))
        category_index = self.categories.index(news.news_category)

        return interest_decay_factor * interest_decay_factor_2 * self.user_quality_measure[category_index]


# Given a file containing k users, creates a list with k user objects
def read_users_from_file(filename):
    result = []

    file = open(filename, "r")
    lines = file.read().splitlines()

    for user_line in lines:
        elem_list = user_line.split(",")
        result.append(SyntheticUser(user_id=int(elem_list[0]),
                                    genre=elem_list[1],
                                    age=int(elem_list[2]),
                                    attention_bias_category=elem_list[3]))
    return result


# It writes "number of users" random users in a file named "filename"
def write_random_users_in_file(filename, number_of_users):
    genre_list = ["M", "F"]
    age_list = [i for i in range(10, 86)]
    attention_bias_category_list = ["A", "B", "C", "D", "E", "F"]
    file = open(filename, "w")

    for i in range(number_of_users):
        file.write(str(i) +
                   "," +
                   np.random.choice(genre_list) +
                   "," +
                   str(np.random.choice(age_list)) +
                   "," +
                   np.random.choice(attention_bias_category_list) +
                   "\n")
    file.close()


if __name__ == "__main__":

    back = 7
    forward = 7
    f = open("reward_decay_LP_frequent.txt")
    res = f.read().split(",")
    res = list(map(float, res))

    final_res = []

    for i in range(len(res)):
        if i < back:
            final_res.append((sum(res[0:i]) + res[i] + sum(res[i + 1: i + 1 + forward])) /
                             (len(res[0:i]) + len(res[i + 1: i + 1 + forward]) + 1))
        elif i + forward >= len(res):
            final_res.append((sum(res[i - back:i]) + res[i] + sum(res[-len(res) + i - 1: -1])) /
                             (len(res[i - back:i]) + len(res[-len(res) + i - 1: -1]) + 1))
        else:
            final_res.append((sum(res[i - back:i]) + res[i] + sum(res[i + 1: i + 1 + forward])) /
                             (len(res[i - back:i]) + len(res[i + 1: i + 1 + forward]) + 1))

    final_final_res = []

    back = 10
    forward = 10
    for i in range(len(res)):
        if i < back:
            final_final_res.append((sum(final_res[0:i]) + final_res[i] + sum(final_res[i + 1: i + 1 + forward])) /
                             (len(final_res[0:i]) + len(final_res[i + 1: i + 1 + forward]) + 1))
        elif i + forward >= len(res):
            final_final_res.append((sum(final_res[i - back:i]) + final_res[i] + sum(final_res[-len(final_res) + i - 1: -1])) /
                             (len(final_res[i - back:i]) + len(final_res[-len(final_res) + i - 1: -1]) + 1))
        else:
            final_final_res.append((sum(final_res[i - back:i]) + final_res[i] + sum(final_res[i + 1: i + 1 + forward])) /
                             (len(final_res[i - back:i]) + len(final_res[i + 1: i + 1 + forward]) + 1))

    plt.plot(res)
    plt.plot(final_final_res, "r")
    plt.show()







