import numpy as np
from news import *
from synthetic_user import *
from PIL import Image
import scipy.stats as stat
import scipy.special as sp
import matplotlib.pyplot as plt
import seaborn as sns



class NewsLearner:

    # Category and Layout slots for direct user interactions,
    # Real slot promenances for syntethic user interactions

    def __init__(self, categories=[], layout_slots=10, real_slot_promenances=[]):

        self.categories = categories  # params to be learnt, the categories of news and ads
        self.last_proposal_weights = np.ones(len(self.categories))  # used to speed up the process of rejecton sampl.
        self.multiple_arms_avg_reward = []  # here we store the values of each pulled arm, for the regret plot
        self.news_pool = []  # all available news are kept here
        self.click_per_page = []
        self.layout_slots = layout_slots  # the number of slots of a single page
        self.real_slot_promenances = real_slot_promenances  # The real values of slot promenance
        self.last_news_observed = []  # we consider here only the last 50 news observed
        # The number of times we assigned category k to slot i
        self.category_per_slot_assignment_count = np.zeros([len(self.categories), self.layout_slots])
        # The number of times we observed a positive reward for category k allocated in slot i
        self.category_per_slot_reward_count = np.zeros([len(self.categories), self.layout_slots])
        self.quality_parameters = np.ones([len(self.categories), 2])  # TS parameters for quality estimate
        self.promenance_parameters = np.ones([self.layout_slots, 2])  # TS parameters for slot prom. estimate

    # Returns a TS-sample for category k, either with standard TS approach or PBM approach
    def sample_quality(self, category, approach="standard"):

        if approach == "standard":

            index = self.categories.index(category)
            return np.random.beta(a=self.quality_parameters[index][0], b=self.quality_parameters[index][1])
        elif approach == "position_based_model":

            result = -1
            category_index = self.categories.index(category)
            proposal_weight = self.last_proposal_weights[category_index]
            count = 1

            while result == -1:

                if count % 30 == 0:
                    proposal_weight /= 10

                x_proposal = self.sample_from_alternative_proposal_distribution()
                y_proposal = self.alternative_proposal_distribution_pdf(weight=30 * proposal_weight)
                y_sample = np.random.uniform(0, y_proposal)
                y_target = self.target_distribution_pdf(category=category, x_value=x_proposal)
                if y_target > y_proposal:
                    proposal_weight = y_target / 30 + (y_target - y_proposal) / 30
                elif y_sample < y_target:
                    result = x_proposal
                    self.last_proposal_weights[category_index] = proposal_weight

                count += 1

            return result

    # Returns a TS- sample or a real value for the slot promenance of the slot k (k=-1 for all the slots)
    def sample_promenance(self, slot=-1, use_real_value=True):

        if use_real_value:
            return self.real_slot_promenances

        if slot < 0:
            result = []
            for param in self.promenance_parameters:
                result.append(np.random.beta(a=param[0], b=param[1]))
            return result
        else:
            return np.random.beta(a=self.promenance_parameters[slot][1], b=self.promenance_parameters[slot][2])

    # Collect a positive / negative reward for slot the observation of slot k
    def slot_observation(self, slots_nr, observed=True):

        if observed:
            for slot_nr in slots_nr:
                self.promenance_parameters[slot_nr][0] += 1
        else:
            for slot_nr in slots_nr:
                self.promenance_parameters[slot_nr][1] += 1

    # Collect a positive / negative reward for the news(s) k click allocated in slot i(s)
    def news_click(self, news, clicked=True, slot_nr=[]):

        if clicked:
            i = 0
            for content in news:
                category_index = self.categories.index(content.news_category)
                self.quality_parameters[category_index][0] += 1
                if len(slot_nr) > 0:
                    self.category_per_slot_reward_count[category_index][slot_nr[i]] += 1
                    i += 1
        else:
            for content in news:
                index = self.categories.index(content.news_category)
                self.quality_parameters[index][1] += 1

    # Given the knowledge of quality and promencances parameters up to now, tries to find the best allocation
    # by start allocating more promising news in more promenant slots
    def find_best_allocation(self, verbose=False):

        result_news_allocation = [0] * self.layout_slots

        for news in self.news_pool:
            # decay_factor = np.exp(- self.last_news_observed.count(news.news_id))
            news.set_sampled_quality(value=self.sample_quality(news.news_category, approach="position_based_model"))

        self.news_pool.sort(key=lambda x: x.sampled_quality, reverse=True)
        tmp_news_pool = self.news_pool.copy()

        slot_promenances = self.sample_promenance().copy()

        if verbose:
            print(slot_promenances)

        for i in range(len(slot_promenances)):
            target_slot_index = np.argmax(slot_promenances)
            assigning_news = tmp_news_pool.pop(0)
            result_news_allocation[int(target_slot_index)] = assigning_news
            category_index = self.categories.index(assigning_news.news_category)
            self.category_per_slot_assignment_count[category_index][target_slot_index] += 1
            slot_promenances[target_slot_index] = -1

        if verbose:
            for elem in result_news_allocation:
                print(elem.news_name)
        return result_news_allocation

    # Adds news to the current pool
    def fill_news_pool(self, news_list, append=True):

        if append:
            for news in news_list:
                self.news_pool.append(news)
        else:
            self.news_pool = news_list.copy()

    # Keeps track on the last news observed by the user
    def observed_news(self, news_observed):

        for news in news_observed:
            self.last_news_observed.append(news.news_id)

    # Gets a sample from the proposal distribution (not ideal distribution for our purpose)
    def sample_from_proposal_distribution(self, category):

        category_index = self.categories.index(category)
        target_slot = np.argmax(self.category_per_slot_assignment_count[category_index])
        return np.random.beta(a=1 + self.category_per_slot_reward_count[category_index][target_slot],
                              b=1 + self.category_per_slot_assignment_count[category_index][target_slot] -
                                    self.category_per_slot_reward_count[category_index][target_slot])

    # Return a value from the probability density function of the proposal distribution corresponding to a category
    # and an x_value
    def proposal_distribution_pdf(self, category, x_value):

        category_index = self.categories.index(category)
        target_slot = np.argmax(self.category_per_slot_assignment_count[category_index])
        return 1 / self.real_slot_promenances[int(target_slot)] * \
               stat.beta(a=1 + self.category_per_slot_reward_count[category_index][target_slot],
                         b=1 + self.category_per_slot_assignment_count[category_index][target_slot] -
                           self.category_per_slot_reward_count[category_index][target_slot]).pdf(x_value)

    # Return a value from the probability density function of target distribution corresponding to a category and an
    # x_value
    def target_distribution_pdf(self, category, x_value):

        category_index = self.categories.index(category)
        result = 1
        for i in range(self.layout_slots):
            alpha = self.category_per_slot_reward_count[category_index][i]
            beta = self.category_per_slot_assignment_count[category_index][i] - \
                   self.category_per_slot_reward_count[category_index][i]

            result *= x_value**alpha * (1 - self.real_slot_promenances[i] * x_value)**beta

        return result

    # Sample a value from an alternative proposal defined by me
    def sample_from_alternative_proposal_distribution(self):
        return np.random.uniform(0, 1)

    # Return a value from the probability density funtion of an alternative proposal dist, scaled by a weight
    def alternative_proposal_distribution_pdf(self, weight=1):
        return 1 * weight

    # When a user arrives, finds an allocation for that user by exploiting the knowledge accumulated so far, then
    # simulates the users interactions with the proposed page and collects the information
    def user_arrival(self, user):
        allocation = self.find_best_allocation()
        user_observation_probabilities = user.observation_probabilities(self.real_slot_promenances)
        arm_rewards = []
        page_clicks = 0

        for i in range(len(user_observation_probabilities)):
            outcome = np.random.binomial(1, user_observation_probabilities[i])
            clicked, avg_reward = user.click_news(allocation[i])
            arm_rewards.append(avg_reward)
            if (outcome == 1) and (clicked == 1):
                self.news_click([allocation[i]], slot_nr=[i])
                page_clicks += 1

        self.multiple_arms_avg_reward.append(np.mean(arm_rewards))
        self.click_per_page.append(page_clicks)

    # Plots the proposal distribution of a given category
    def print_proposal_distributin(self, category):
        category_index = self.categories.index(category)
        target_slot = np.argmax(self.category_per_slot_assignment_count[category_index])
        result = []
        i = 0
        while i <= 1:
            result.append(1 / self.real_slot_promenances[int(target_slot)] * \
               stat.beta(a=1 + self.category_per_slot_reward_count[category_index][target_slot],
                         b=1 + self.category_per_slot_assignment_count[category_index][target_slot] -
                           self.category_per_slot_reward_count[category_index][target_slot]).pdf(i))
            i += 0.001
        plt.plot(result)
        plt.show()

    # Plots the target distribution of a given category
    def print_target_distribution(self, category):

        category_index = self.categories.index(category)
        result = 1
        result_to_plot = []
        x_value = 0

        while x_value <= 1.001:
            for i in range(self.layout_slots):
                alpha = self.category_per_slot_reward_count[category_index][i]
                beta = self.category_per_slot_assignment_count[category_index][i] - \
                       self.category_per_slot_reward_count[category_index][i]

                result *= x_value ** alpha * (1 - self.real_slot_promenances[i] * x_value) ** beta
            result_to_plot.append(result)
            result = 1
            x_value += 0.001

        plt.title(category)
        plt.plot(result_to_plot)
        plt.show()


if __name__ == "__main__":

    # We fill the news pool with a bounch of news
    news_pool = []
    k = 0
    for category in ["sport", "cibo", "tech", "politic", "gossip", "scienza"]:
        for id in range(1, 21):
            news_pool.append(News(news_id=k,
                                  news_name=category + "-" + str(id)))
            k += 1

            if category + "-" + str(id) in ["cibo-1", "cibo-6", "cibo-13", "cibo-17",
                                            "gossip-14",
                                            "politic-5", "politic-19",
                                            "scienza-6", "scienza-11",
                                            "sport-1", "sport-6", "sport-8", "sport-19", "sport-20",
                                            "tech-4", "tech-10", "tech-14", "tech-20"]:
                news_pool.__delitem__(-1)

    # We create a user and set their quality metrics that we want to estimate
    u = SyntheticUser(23, "M", 27, "C")  # A male 27 years old user, that is transparent to slot promenances
    u.user_quality_measure = [0.2, 0.3, 0.7, 0.2, 0.2, 0.4]

    exp = 0
    result = []
    click_result = []

    # Then we perform 100 experiments and use the collected data to plot the regrets and distributions
    while exp < 300:
        print("exp " + str(exp))
        a = NewsLearner(categories=["cibo", "gossip", "politic", "scienza", "sport", "tech"], layout_slots=5,
                        real_slot_promenances=[0.7, 0.8, 0.3, 0.5, 0.3])
        a.fill_news_pool(news_list=news_pool, append=False)

        for i in range(200):
            a.user_arrival(u)  # we simulate 200 interactions per user
        result.append(a.multiple_arms_avg_reward)
        click_result.append(a.click_per_page)
        exp += 1
        if exp == 299:
            a.print_target_distribution("politic")
            a.print_target_distribution("tech")

    plt.plot(np.mean(result, axis=0))
    plt.title("Reward - " + str(u.user_quality_measure))
    plt.show()
    plt.title("Regret - " + str(u.user_quality_measure))
    plt.plot(np.cumsum(np.max(u.user_quality_measure) - np.array(np.mean(result, axis=0))))
    plt.show()
    plt.title("Page Clicks - " + str(u.user_quality_measure))
    plt.plot(np.mean(click_result, axis=0))
    plt.show()

