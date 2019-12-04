import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt


class WeightedBetaDistribution:

    def __init__(self, categories, layout_slots, real_slot_promenances, target_dist_auto_increasing=False):
        self.categories = categories
        self.layout_slots = layout_slots
        self.category_per_slot_reward_count = np.zeros([len(self.categories), self.layout_slots])
        self.category_per_slot_assignment_count = np.zeros([len(self.categories), self.layout_slots])
        self.last_proposal_weights = np.ones(len(self.categories))
        self.real_slot_promenances = real_slot_promenances

        self.sample_count = [0] * len(self.categories)
        self.auto_increasing = target_dist_auto_increasing
        self.last_target_multiplier_power = [0] * len(self.categories)

    def sample(self, category):

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
        if self.auto_increasing:
            self.sample_count[category_index] += 1
            if self.sample_count[category_index] >= 50:
                print("beginning")
                self.sample_count[category_index] = 0
                self.shift_up_target_distribution(threshold=10 ** -3)
                print("end")
        return result

    def __weighted_beta_pdf(self, category, x_value):

        category_index = self.categories.index(category)
        result = 1
        for i in range(self.layout_slots):
            alpha = self.category_per_slot_reward_count[category_index][i]
            beta = self.category_per_slot_assignment_count[category_index][i] - \
                   self.category_per_slot_reward_count[category_index][i]

            result *= x_value ** alpha * (1 - self.real_slot_promenances[i] * x_value) ** beta

        result = result * np.power(10, self.last_target_multiplier_power[category_index])
        return result

    def __uniform_pdf(self, weight):
        return 1 * weight

    def __sample_uniform(self):
        return np.random.uniform(0, 1)

    def news_allocation(self, news, slot_index):

        category_index = self.categories.index(news.news_category)
        self.category_per_slot_assignment_count[category_index][slot_index] += 1

    def news_click(self, news, slot_index):

        category_index = self.categories.index(news.news_category)
        self.category_per_slot_reward_count[category_index][slot_index] += 1

    def plot_distribution(self, category):

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
            result = result * np.power(10, self.last_target_multiplier_power[category_index])
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

    def shift_up_target_distribution(self, threshold):

        for category in self.categories:
            category_index = self.categories.index(category)
            result = 1
            target_distribution_function = []
            x_value = 0
            while x_value <= 1:
                for i in range(self.layout_slots):
                    alpha = self.category_per_slot_reward_count[category_index][i]
                    beta = self.category_per_slot_assignment_count[category_index][i] - \
                           self.category_per_slot_reward_count[category_index][i]

                    result *= x_value ** alpha * (1 - self.real_slot_promenances[i] * x_value) ** beta
                target_distribution_function.append(result)
                result = 1
                x_value += 0.001

            target_distribution_function = list(np.array(target_distribution_function) *
                                                np.power(10, self.last_target_multiplier_power[category_index]))
            max_value = np.max(target_distribution_function)
            min_value = np.min(target_distribution_function)
            try:
                value = (max_value - min_value) / max_value
            except ZeroDivisionError:
                continue
            if value >= 0.5:
                count = 0
                while max_value * np.power(10, self.last_target_multiplier_power[category_index]) <= threshold:
                    count += 1
                    if count > 1000:
                        print("inf")
                    # print(max_value * np.power(10, self.last_target_multiplier_power[category_index]))
                    self.last_target_multiplier_power[category_index] += 3
                    self.last_proposal_weights[category_index] = 1

            print("finished with " + str(max_value * np.power(10, self.last_target_multiplier_power[category_index])))
