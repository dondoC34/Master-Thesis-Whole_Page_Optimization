import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt


class WeightedBetaDistribution:

    def __init__(self, categories, layout_slots, real_slot_promenances, target_dist_auto_increasing=False):
        self.categories = categories
        self.layout_slots = layout_slots
        self.real_slot_promenances = real_slot_promenances
        # TWO MATRICES TO COMPUTE THE WEIGHTED BETA FUNCTION GIVEN A CATEGORY
        self.category_per_slot_reward_count = np.zeros([len(self.categories), self.layout_slots])
        self.category_per_slot_assignment_count = np.zeros([len(self.categories), self.layout_slots])

        # WEIGHTS USED TO REDUCE THE PROPOSAL DISTRIBUTION AND TO AVOID THE REJECTION SAMPLING OPERATION
        # BEGIN SUBOPTIMAL
        self.last_proposal_weights = np.ones(len(self.categories))

    def sample(self, category):
        """
        Given a category, performs the rejection sampling techinique to get a sample from the weighted beta distribution
        corresponding to category. To speed-up the process and to avoid the process itself become unefficient, a
        proposal weight used for the previous iteration is kept in memory.
        :param category: Used to choose the category and then to build the corresponding weighted beta probability
        density function to rejection sampling from.
        :return: A sample of the weighted beta distribution corresponding to the param "category".
        """
        result = -1
        category_index = self.categories.index(category)
        proposal_weight = self.last_proposal_weights[category_index]
        count = 1

        while result == -1:

            if count % 30 == 0:
                proposal_weight /= 10

            if count == 300000:
                print("cannot sample")
                self.plot_distribution(category=category)
                print(self.category_per_slot_assignment_count)
                print("----")
                print(self.category_per_slot_reward_count)

            x_proposal = self.__sample_uniform()
            y_proposal = self.__uniform_pdf(weight=30 * proposal_weight)
            y_sample = np.random.uniform(0, y_proposal)
            y_target = self.__weighted_beta_pdf(category=category, x_value=x_proposal)
            if y_target > y_proposal:
                proposal_weight = y_target / 30 + (y_target - y_proposal) / 30
            elif y_sample < y_target:
                result = x_proposal
                self.last_proposal_weights[category_index] = proposal_weight

            count += 1
        return result

    def __weighted_beta_pdf(self, category, x_value):
        """
        :param category: Used to choose the category and then to build the corresponding weighted beta probability
        density function.
        :param x_value: The x-value of the probability density function we are interested in.
        :return: The value of the weighted beta probability density function corresponding to x-value.
        """
        category_index = self.categories.index(category)
        result = 1
        for i in range(self.layout_slots):
            alpha = self.category_per_slot_reward_count[category_index][i]
            beta = self.category_per_slot_assignment_count[category_index][i] - \
                   self.category_per_slot_reward_count[category_index][i]

            result *= x_value ** alpha * (1 - self.real_slot_promenances[i] * x_value) ** beta

        return result

    def __uniform_pdf(self, weight):
        """
        :param weight: The weight to be multiplicated to the uniform distribution probability density function.
        :return: uniform_pdf * weight
        """
        return 1 * weight

    def __sample_uniform(self):
        """
        :return: A sample from a uniform probability distribution between 0 and 1.
        """
        return np.random.uniform(0, 1)

    def news_allocation(self, news, slot_index):
        """
        Update the weighted beta matrix given a news allocation and the corresponding category.
        :param news: The news being allocated.
        :param slot_index: The slot in which the news is being allocated.
        :return: Nothing.
        """
        category_index = self.categories.index(news.news_category)
        self.category_per_slot_assignment_count[category_index][slot_index] += 1

    def news_click(self, news, slot_index):
        """
        Update the weighted beta matrix given a news click and the corresponding category.
        :param news: The news being clicked.
        :param slot_index: The slot in which the news is being clicked.
        :return: Nothing.
        """
        category_index = self.categories.index(news.news_category)
        self.category_per_slot_reward_count[category_index][slot_index] += 1

    def plot_distribution(self, category):
        """
        Plots the probability density function of the weighted beta distribution corresponding to the category parameter
        :param category: Used to choose the weighted beta pdf to be plotted.
        :return: Nothing.
        """
        category_index = self.categories.index(category)
        result = 1
        result_to_plot = []
        x_value = 0
        while x_value <= 1:
            for i in range(self.layout_slots):
                alpha = self.category_per_slot_reward_count[category_index][i]
                beta = self.category_per_slot_assignment_count[category_index][i] - \
                       self.category_per_slot_reward_count[category_index][i]

                result *= x_value ** alpha * (1 - self.real_slot_promenances[i] * x_value) ** beta
            result_to_plot.append(result)
            result = 1
            x_value += 0.001

        plt.plot(result_to_plot)
        plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                   ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"])
        plt.xlabel("X")
        plt.ylabel("Weighted Beta Pdf")
        plt.title(category + " Estimated Quality")
        plt.show()

