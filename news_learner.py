from synthetic_user import *
from weighted_beta_distribution import *
import scipy.optimize as opt
import time as time
from ads_news import *


class NewsLearner:

    # Category and Layout slots for direct user interactions,
    # Real slot promenances for syntethic user interactions

    def __init__(self, categories=[], layout_slots=10, real_slot_promenances=[], news_column_pivot=[0.01, 1],
                 news_row_pivot=[1], allocation_approach="standard",
                 allocation_diversity_bounds=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1), lp_rand_technique="rand_1"):

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
        self.weighted_betas_matrix = []
        self.news_row_pivots = news_row_pivot
        self.news_column_pivots = news_column_pivot
        self.allocation_approach = allocation_approach
        self.A = []
        self.B = list(-1 * np.array(allocation_diversity_bounds)) + [1] * self.layout_slots
        self.bounds = []
        self.lambdas = []
        self.C = []
        self.lp_rand_tech = lp_rand_technique

        for _ in range(len(news_row_pivot) + 1):
            row = []
            for _ in range(len(news_column_pivot) + 1):
                row.append(WeightedBetaDistribution(self.categories,
                                                    self.layout_slots,
                                                    self.real_slot_promenances))
            self.weighted_betas_matrix.append(row.copy())

    # Returns a TS-sample for category k, either with standard TS approach or PBM approach
    def sample_quality(self, news, approach="standard", interest_decay=False):

        category = news[0].news_category
        if approach == "standard":

            index = self.categories.index(category)
            return np.random.beta(a=self.quality_parameters[index][0], b=self.quality_parameters[index][1])
        elif approach == "position_based_model":

            if interest_decay:
                news[0].set_sampled_quality(value=self.weighted_betas_matrix[news[1]][news[2]].sample(category=category))
            else:
                news[0].set_sampled_quality(value=self.weighted_betas_matrix[0][0].sample(category=category))

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
    def news_click(self, news, clicked=True, slot_nr=[], interest_decay=False):

        if clicked:
            i = 0
            for content in news:
                category_index = self.categories.index(content[0].news_category)
                self.quality_parameters[category_index][0] += 1
                if len(slot_nr) > 0:
                    if interest_decay:
                        self.weighted_betas_matrix[content[1]][content[2]].news_click(content[0], slot_nr[i])
                        content[0].click_sum += 1
                        if content[0].click_sum <= self.news_row_pivots[-1]:
                            i = 0
                            while content[0].click_sum > self.news_row_pivots[i]:
                                i += 1
                            content[1] = i + 1
                        else:
                            content[1] = len(self.news_row_pivots)
                        content[2] = content[3]
                    else:
                        self.weighted_betas_matrix[0][0].news_click(content[0], slot_nr[i])
                    i += 1
        else:
            for content in news:
                index = self.categories.index(content[0].news_category)
                self.quality_parameters[index][1] += 1

    # Given the knowledge of quality and promencances parameters up to now, tries to find the best allocation
    # by start allocating more promising news in more promenant slots
    def find_best_allocation(self, interest_decay=False):

        result_news_allocation = [0] * self.layout_slots

        for news in self.news_pool:
            self.sample_quality(news=news, approach="position_based_model", interest_decay=interest_decay)

        if self.allocation_approach == "standard":
            self.news_pool.sort(key=lambda x: x[0].sampled_quality, reverse=True)
            tmp_news_pool = self.news_pool.copy()
            slot_promenances = self.sample_promenance().copy()

            for i in range(len(slot_promenances)):
                target_slot_index = np.argmax(slot_promenances)
                assigning_news = tmp_news_pool.pop(0)
                result_news_allocation[int(target_slot_index)] = assigning_news
                slot_promenances[int(target_slot_index)] = -1

        elif self.allocation_approach == "LP":

            thetas = []
            for news in self.news_pool:
                thetas += [news[0].sampled_quality] * self.layout_slots
            self.C = list(np.array(thetas) * np.array(self.lambdas))
            linear_problem = opt.linprog(A_ub=self.A, b_ub=self.B, c=self.C)
            slots_assegnation_probabilities = []
            slot_counter = 0
            tmp_slot_probabilities = []
            while slot_counter < self.layout_slots:
                i = slot_counter
                while i < len(linear_problem.x):
                    tmp_slot_probabilities.append(linear_problem.x[i])
                    i += self.layout_slots
                slots_assegnation_probabilities.append(tmp_slot_probabilities.copy())
                tmp_slot_probabilities.clear()
                slot_counter += 1

            if self.lp_rand_tech == "rand_1":
                tmp_slot_promenances = self.real_slot_promenances.copy()
                feasible_news = [i for i in range(len(self.news_pool))]
                slot_counter = 0
                while slot_counter < self.layout_slots:
                    target_slot = np.argmax(tmp_slot_promenances)
                    target_slot_assegnation_probabilities = slots_assegnation_probabilities[int(target_slot)]
                    target_slot_assegnation_probabilities = list(np.array(target_slot_assegnation_probabilities) * 1 /
                                                                 sum(target_slot_assegnation_probabilities))
                    selected_news = np.random.choice(feasible_news, p=target_slot_assegnation_probabilities)
                    result_news_allocation[int(target_slot)] = self.news_pool[selected_news]
                    deletion_index = feasible_news.index(selected_news)
                    feasible_news.__delitem__(deletion_index)
                    for probs in slots_assegnation_probabilities:
                        probs.__delitem__(deletion_index)
                    tmp_slot_promenances[int(target_slot)] = -1
                    slot_counter += 1

            elif self.lp_rand_tech == "rand_2":
                pass

        for i in range(len(result_news_allocation)):
            if interest_decay:
                self.weighted_betas_matrix[result_news_allocation[i][1]][result_news_allocation[i][2]].news_allocation(result_news_allocation[i][0], i)
                assigned_slot_promenance = self.real_slot_promenances[i]
                result_news_allocation[i][0].slot_promenance_cumsum += assigned_slot_promenance
                if result_news_allocation[i][0].slot_promenance_cumsum <= self.news_column_pivots[-1]:
                    k = 0
                    while result_news_allocation[i][0].slot_promenance_cumsum > self.news_column_pivots[k]:
                        k += 1
                    result_news_allocation[i][3] = k
                else:
                    result_news_allocation[i][3] = len(self.news_column_pivots)
            else:
                self.weighted_betas_matrix[0][0].news_allocation(result_news_allocation[i][0], i)

        return result_news_allocation

    # Adds news to the current pool
    def fill_news_pool(self, news_list, append=True):

        news_per_category_count = [0] * len(self.categories)
        if append:
            for news in news_list:
                self.news_pool.append([news, 0, 0, 0])
                index = self.categories.index(news.news_category)
                news_per_category_count[index] += 1

            if self.allocation_approach == "LP":
                for _ in range(len(self.news_pool)):
                    self.bounds.append((0, 1))
                    self.lambdas += list(np.array(self.real_slot_promenances) * -1)
                category_count = 0

                # Category connstraints creation and insertion into matrix A
                for i in range(len(self.categories)):
                    row = [0] * (len(self.news_pool) * self.layout_slots - 1)
                    row_slot_promenances = []
                    for _ in range(news_per_category_count[i]):
                        row_slot_promenances += self.real_slot_promenances

                    row_slot_promenances = np.array(row_slot_promenances)
                    tmp_row = [-1] * (news_per_category_count[i] * self.layout_slots)
                    tmp_row = list(row_slot_promenances * tmp_row)
                    row[category_count: category_count + news_per_category_count[i] * self.layout_slots - 1] = tmp_row
                    self.A.append(row.copy())
                    category_count += news_per_category_count[i] * self.layout_slots

                # Slots constraints creation and insertion into matrix A
                for i in range(self.layout_slots):
                    row = [0] * (len(self.news_pool) * self.layout_slots)
                    target_index = i
                    while target_index < len(row):
                        row[target_index] = 1
                        target_index += self.layout_slots

                    self.A.append(row.copy())


        else:
            self.news_pool = news_list.copy()

    # Keeps track on the last news observed by the user
    def observed_news(self, news_observed):

        for news in news_observed:
            self.last_news_observed.append(news.news_id)

    # When a user arrives, finds an allocation for that user by exploiting the knowledge accumulated so far, then
    # simulates the users interactions with the proposed page and collects the information
    def user_arrival(self, user, interest_decay=False):
        allocation = self.find_best_allocation(interest_decay=interest_decay)
        user_observation_probabilities = user.observation_probabilities(self.real_slot_promenances)
        arm_rewards = []
        page_clicks = 0

        for i in range(len(user_observation_probabilities)):
            outcome = np.random.binomial(1, user_observation_probabilities[i])
            clicked, avg_reward = user.click_news(allocation[i][0], interest_decay=interest_decay)
            arm_rewards.append(avg_reward)
            if (outcome == 1) and (clicked == 1):
                self.news_click([allocation[i]], slot_nr=[i], interest_decay=interest_decay)
                page_clicks += 1
            else:
                allocation[i][2] = allocation[i][3]

        self.multiple_arms_avg_reward.append(np.mean(arm_rewards))
        self.click_per_page.append(page_clicks)


if __name__ == "__main__":

    # We fill the news pool with a bounch of news
    news_pool = []
    k = 0
    for category in ["sport", "cibo", "tech", "politic", "gossip", "scienza"]:
        for id in range(1, 100):
            news_pool.append(News(news_id=k,
                                  news_name=category + "-" + str(id)))
            k += 1

    exp = 0
    result = []
    click_result = []

    # Then we perform 100 experiments and use the collected data to plot the regrets and distributions
    while exp < 1:
        print("exp " + str(exp))
        # We create a user and set their quality metrics that we want to estimate
        u = SyntheticUser(23, "M", 27, "C")  # A male 27 years old user, that is transparent to slot promenances
        u.user_quality_measure = [0.2, 0.3, 0.7, 0.2, 0.2, 0.4]
        a = NewsLearner(categories=["cibo", "gossip", "politic", "scienza", "sport", "tech"], layout_slots=5,
                        real_slot_promenances=[0.7, 0.8, 0.3, 0.5, 0.3], allocation_approach="LP")
        a.fill_news_pool(news_list=news_pool, append=True)

        for i in range(300):
            if i % 100 == 0:
                print(i)
            a.user_arrival(u, interest_decay=True)  # we simulate 200 interactions per user
        result.append(a.multiple_arms_avg_reward)
        click_result.append(a.click_per_page)
        if exp == 0:
            a.weighted_betas_matrix[0][0].plot_distribution("politic")
            a.weighted_betas_matrix[0][1].plot_distribution("politic")
            a.weighted_betas_matrix[1][2].plot_distribution("politic")
            print(a.weighted_betas_matrix[0][0].category_per_slot_reward_count)
            print(a.weighted_betas_matrix[0][0].category_per_slot_assignment_count)
            print("--------------------------")
            print(a.weighted_betas_matrix[0][1].category_per_slot_reward_count)
            print(a.weighted_betas_matrix[0][1].category_per_slot_assignment_count)
            print("--------------------------")
            print(a.weighted_betas_matrix[1][2].category_per_slot_reward_count)
            print(a.weighted_betas_matrix[1][2].category_per_slot_assignment_count)
        exp += 1

    plt.plot(np.mean(result, axis=0))
    plt.title("Reward - " + str(u.user_quality_measure))
    plt.show()
    plt.title("Regret - " + str(u.user_quality_measure))
    plt.plot(np.cumsum(np.max(u.user_quality_measure) - np.array(np.mean(result, axis=0))))
    plt.show()
    plt.title("Page Clicks - " + str(u.user_quality_measure))
    plt.plot(np.mean(click_result, axis=0))
    plt.show()


