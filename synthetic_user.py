import numpy as np
from news_learner import *
slot_number = 5
categories_number = 6
categories = ["cibo", "gossip", "politic", "scienza", "sport", "tech"]


class SyntheticUser:
    def __init__(self, user_id, genre, age, attention_bias_category):
        self.user_id = user_id
        self.genre = genre
        self.categories = categories
        self.user_quality_measure = []
        # We assing a random number for each category's quality measure for the user
        for i in range(categories_number):
            self.user_quality_measure.append(np.random.uniform(0, 1))
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

        # This will be used in case of context generation
        self.personal_learner = None

    # Returns the probabilities that a user observe slots given their attention bias category
    def observation_probabilities(self, slot_promenances):
        return slot_promenances * np.array(self.attention_function)

    # To be used in case of context generation....
    def assign_learner_with_data(self, data_matrix):
        pass

    # Returns the reward obtained by showing the news "news" to the user
    def click_news(self, news):
        category_index = self.categories.index(news.news_category)
        return np.random.binomial(1, self.user_quality_measure[category_index]), self.user_quality_measure[category_index]
                                                
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



