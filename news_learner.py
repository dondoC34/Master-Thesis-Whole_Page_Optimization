from synthetic_user import *
from weighted_beta_distribution import *
import scipy.optimize as opt
import time as time
from ads_news import *
from pulp import *
from tqdm import tqdm


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
        for _ in range(len(news_row_pivot) + 1):
            row = []
            for _ in range(len(news_column_pivot) + 1):
                row.append(WeightedBetaDistribution(self.categories,
                                                    self.layout_slots,
                                                    self.real_slot_promenances,
                                                    target_dist_auto_increasing=False))
            self.weighted_betas_matrix.append(row.copy())

        # Linear programming attributes
        self.allocation_approach = allocation_approach
        self.A = []
        self.B = list(-1 * np.array(allocation_diversity_bounds)) + [1] * (self.layout_slots + len(self.categories) * self.layout_slots)
        self.bounds = [(0, 1)] * len(self.categories) * self.layout_slots * self.layout_slots
        self.lambdas = []
        self.C = []
        self.lp_rand_tech = lp_rand_technique
        for _ in range(len(self.categories) * self.layout_slots):
            self.lambdas += list(np.array(self.real_slot_promenances) * -1)
        # Category connstraints creation and insertion into matrix A
        category_count = 0
        for i in range(len(self.categories)):
            row = [0] * (len(self.categories) * self.layout_slots * self.layout_slots - 1)
            row_slot_promenances = []
            for _ in range(self.layout_slots):
                row_slot_promenances += self.real_slot_promenances
            row_slot_promenances = np.array(row_slot_promenances)
            tmp_row = [-1] * self.layout_slots * self.layout_slots
            tmp_row = list(row_slot_promenances * tmp_row)
            row[category_count: category_count + self.layout_slots * self.layout_slots - 1] = tmp_row
            self.A.append(row.copy())
            category_count += self.layout_slots * self.layout_slots
        # Slots constraints creation and insertion into matrix A
        for i in range(self.layout_slots):
            row = [0] * (len(self.categories) * self.layout_slots * self.layout_slots)
            target_index = i
            while target_index < len(row):
                row[target_index] = 1
                target_index += self.layout_slots

            self.A.append(row.copy())

        # Variables' capacity constraints creation and insertion into matrix A
        initial_index = 0
        for _ in range(len(self.categories) * self.layout_slots):
            row = [0] * (len(self.categories) * self.layout_slots * self.layout_slots)
            row[initial_index:initial_index + self.layout_slots] = [1] * self.layout_slots
            self.A.append(row.copy())
            initial_index += self.layout_slots

        self.rand_1_errors = []
        self.rand_2_errors = []
        self.rand_3_errors = []

    # Returns a TS-sample for category k, either with standard TS approach or PBM approach
    def sample_quality(self, news, user, approach="standard", interest_decay=False):

        category = news.news_category
        if approach == "standard":

            index = self.categories.index(category)
            return np.random.beta(a=self.quality_parameters[index][0], b=self.quality_parameters[index][1])
        elif approach == "position_based_model":

            if interest_decay:
                slot_promenance_cumsum = next((x[1] for x in user.last_news_in_allocation if x[0] == news), 0)
                total_num_of_clicks = next((x[1] for x in user.last_news_clicked if x[0] == news), 0)

                if slot_promenance_cumsum < self.news_column_pivots[-1]:
                    k = 0
                    while slot_promenance_cumsum >= self.news_column_pivots[k]:
                        k += 1
                    weighted_beta_matrix_posy = k
                else:
                    weighted_beta_matrix_posy = len(self.news_column_pivots)

                if total_num_of_clicks < self.news_row_pivots[-1]:
                    k = 0
                    while total_num_of_clicks >= self.news_row_pivots[k]:
                        k += 1
                    weighted_beta_matrix_posx = k
                else:
                    weighted_beta_matrix_posx = len(self.news_row_pivots)
                news.set_sampled_quality(value=self.weighted_betas_matrix[weighted_beta_matrix_posx][weighted_beta_matrix_posy].sample(category=category))
            else:
                news.set_sampled_quality(value=self.weighted_betas_matrix[0][0].sample(category=category))

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
    def news_click(self, news, user, clicked=True, slot_nr=[], interest_decay=False):

        if clicked:
            i = 0
            for content in news:
                category_index = self.categories.index(content.news_category)
                self.quality_parameters[category_index][0] += 1
                if len(slot_nr) > 0:
                    if interest_decay:
                        slot_promenance_cumsum = next((x[1] for x in user.last_news_in_allocation if x[0] == content), 0)
                        total_num_of_clicks = next((x[1] for x in user.last_news_clicked if x[0] == content), 0)

                        if slot_promenance_cumsum < self.news_column_pivots[-1]:
                            k = 0
                            while slot_promenance_cumsum >= self.news_column_pivots[k]:
                                k += 1
                            weighted_beta_matrix_posy = k
                        else:
                            weighted_beta_matrix_posy = len(self.news_column_pivots)

                        if total_num_of_clicks < self.news_row_pivots[-1]:
                            k = 0
                            while total_num_of_clicks >= self.news_row_pivots[k]:
                                k += 1
                            weighted_beta_matrix_posx = k
                        else:
                            weighted_beta_matrix_posx = len(self.news_row_pivots)

                        self.weighted_betas_matrix[weighted_beta_matrix_posx][weighted_beta_matrix_posy].news_click(content, slot_nr[i])
                        alloc_index = next((x[3] for x in user.last_news_in_allocation if x[0] == content), -1)
                        click_index = next((x[3] for x in user.last_news_clicked if x[0] == content), -1)
                        user.last_news_in_allocation[alloc_index][1] = user.last_news_in_allocation[alloc_index][2]
                        user.last_news_clicked[click_index][1] = user.last_news_clicked[click_index][2]
                    else:
                        self.weighted_betas_matrix[0][0].news_click(content, slot_nr[i])
                    i += 1
        else:
            for content in news:
                index = self.categories.index(content.news_category)
                self.quality_parameters[index][1] += 1

    # Given the knowledge of quality and promencances parameters up to now, tries to find the best allocation
    # by start allocating more promising news in more promenant slots
    def find_best_allocation(self, user, interest_decay=False, update_assignment_matrices=True):
        result_news_allocation = [0] * self.layout_slots

        for news in self.news_pool:
            self.sample_quality(news=news, user=user, approach="position_based_model", interest_decay=interest_decay)

        if self.allocation_approach == "standard":
            self.news_pool.sort(key=lambda x: x.sampled_quality, reverse=True)
            tmp_news_pool = self.news_pool.copy()
            slot_promenances = self.sample_promenance().copy()

            for i in range(len(slot_promenances)):
                target_slot_index = np.argmax(slot_promenances)
                assigning_news = tmp_news_pool.pop(0)
                result_news_allocation[int(target_slot_index)] = assigning_news
                slot_promenances[int(target_slot_index)] = -1

        elif self.allocation_approach == "LP":
            result_news_allocation = self.solve_linear_problem(continuity_relaxation=True)

        else:
            raise RuntimeError("Allocation approach not recognized.")

        if update_assignment_matrices:
            for i in range(len(result_news_allocation)):
                if interest_decay:
                    slot_promenance_cumsum = next((x[1] for x in user.last_news_in_allocation if x[0] == result_news_allocation[i]), 0)
                    total_num_of_clicks = next((x[1] for x in user.last_news_clicked if x[0] == result_news_allocation[i]), 0)

                    if slot_promenance_cumsum < self.news_column_pivots[-1]:
                        k = 0
                        while slot_promenance_cumsum >= self.news_column_pivots[k]:
                            k += 1
                        weighted_beta_matrix_posy = k
                    else:
                        weighted_beta_matrix_posy = len(self.news_column_pivots)

                    if total_num_of_clicks < self.news_row_pivots[-1]:
                        k = 0
                        while total_num_of_clicks >= self.news_row_pivots[k]:
                            k += 1
                        weighted_beta_matrix_posx = k
                    else:
                        weighted_beta_matrix_posx = len(self.news_row_pivots)

                    self.weighted_betas_matrix[weighted_beta_matrix_posx][weighted_beta_matrix_posy].news_allocation(result_news_allocation[i], i)
                    assigned_slot_promenance = self.real_slot_promenances[i]
                    index = next((x[3] for x in user.last_news_in_allocation if x[0] == result_news_allocation[i]), -1)
                    if index == -1:
                        user.last_news_in_allocation.append([result_news_allocation[i], 0, assigned_slot_promenance,
                                                             len(user.last_news_in_allocation)])
                    else:
                        user.last_news_in_allocation[index][2] += assigned_slot_promenance
                else:
                    self.weighted_betas_matrix[0][0].news_allocation(result_news_allocation[i], i)

        return result_news_allocation

    # Adds news to the current pool
    def fill_news_pool(self, news_list, append=True):

        news_per_category_count = [0] * len(self.categories)
        if append:
            for news in news_list:
                self.news_pool.append(news)
                index = self.categories.index(news.news_category)
                news_per_category_count[index] += 1

        else:
            self.news_pool = news_list.copy()

    # Keeps track on the last news observed by the user
    def observed_news(self, news_observed):

        for news in news_observed:
            self.last_news_observed.append(news.news_id)

    # When a user arrives, finds an allocation for that user by exploiting the knowledge accumulated so far, then
    # simulates the users interactions with the proposed page and collects the information
    def user_arrival(self, user, interest_decay=False):
        allocation = self.find_best_allocation(user=user, interest_decay=interest_decay)
        user_observation_probabilities = user.observation_probabilities(self.real_slot_promenances)
        arm_rewards = []
        page_clicks = 0

        for i in range(len(user_observation_probabilities)):
            outcome = np.random.binomial(1, user_observation_probabilities[i])
            if outcome == 1:
                clicked, avg_reward = user.click_news(allocation[i], interest_decay=interest_decay)
                arm_rewards.append(avg_reward)
                if clicked == 1:
                    self.news_click(news=[allocation[i]], slot_nr=[i], interest_decay=interest_decay, user=user)
                    page_clicks += 1
            elif interest_decay:
                index = next((x[3] for x in user.last_news_in_allocation if x[0] == allocation[i]), -1)
                user.last_news_in_allocation[index][1] = user.last_news_in_allocation[index][2]

        # self.multiple_arms_avg_reward.append(np.mean(arm_rewards))
        # self.click_per_page.append(page_clicks)

    def save_weighted_beta_matrices(self, desinence):
        for i in range(len(self.weighted_betas_matrix)):
            for j in range(len(self.weighted_betas_matrix[i])):
                file = open("Weighted_Beta_" + str(i) + "_" + str(j) + "_reward_" + desinence + ".txt", "w")
                for reward_row in self.weighted_betas_matrix[i][j].category_per_slot_reward_count:
                    file.write(str(reward_row[0]))
                    for k in range(1, len(reward_row)):
                        file.write("," + str(reward_row[k]))
                    file.write("\n")
                file.close()
                file = open("Weighted_Beta_" + str(i) + "_" + str(j) + "_assignment_" + desinence + ".txt", "w")
                for assignment_row in self.weighted_betas_matrix[i][j].category_per_slot_assignment_count:
                    file.write(str(assignment_row[0]))
                    for k in range(1, len(assignment_row)):
                        file.write("," + str(assignment_row[k]))
                    file.write("\n")
                file.close()

    def insert_into_news_pool(self, news):

        category_index = self.categories.index(news.news_category)
        for i in range(len(self.news_pool)):
            if self.news_pool[i].news_category == news.news_category:
                self.news_pool.insert(i + 1, news)

        if self.allocation_approach == "LP":
            # Update of the category constraints in matrix A due to the new news
            for i in range(len(self.categories)):
                if (i != len(self.categories) - 1) and (i != category_index):
                    self.A[i] += [0] * len(self.real_slot_promenances)
                elif (i == len(self.categories) - 1) and (i != category_index):
                    self.A[i] = [0] * len(self.real_slot_promenances) + self.A[i]
                else:
                    block_to_be_inserted = list(np.array(self.real_slot_promenances) * -1)
                    for j in range(len(self.A[i])):
                        if self.A[i][j] != 0:
                            break
                    self.A[i] = self.A[i][0:j] + block_to_be_inserted + self.A[i][j:len(self.A[i])]

            # Update of the slot's capacity constraints in matrix A due to the new news
            for i in range(len(self.categories), len(self.A)):
                target_index = i - len(self.categories)
                block_to_be_inserted = [0] * self.layout_slots
                block_to_be_inserted[target_index] = 1
                self.A[i] += block_to_be_inserted

    def read_weighted_beta_matrix_from_file(self, indexes, desinences, folder="Trained_betas_matrices"):

        for i in range(len(indexes)):
            matrix = []
            file = open(folder + "Weighted_Beta_" + str(indexes[i][0]) + "_" + str(indexes[i][1]) + "_assignment_" +
                        str(desinences[i]) + ".txt", 'r')
            lines = file.read().splitlines()
            for line in lines:
                line_splitted = line.split(",")
                matrix.append(list(map(float, line_splitted)))
            self.weighted_betas_matrix[indexes[i][0]][indexes[i][1]].category_per_slot_assignment_count = matrix.copy()

            matrix.clear()
            file = open(folder + "Weighted_Beta_" + str(indexes[i][0]) + "_" + str(indexes[i][1]) + "_reward_" + str(
                desinences[i]) + ".txt", 'r')
            lines = file.read().splitlines()
            for line in lines:
                line_splitted = line.split(",")
                matrix.append(list(map(float, line_splitted)))
            self.weighted_betas_matrix[indexes[i][0]][indexes[i][1]].category_per_slot_reward_count = matrix.copy()

    def remove_news_from_pool(self, news):
        # TODO
        category_index = self.categories.index(news.news_category)

    def measure_allocation_diversity_bounds_errors(self, slots_assegnation_probabilities, LP_news_pool):
        for tech in ["rand_1", "rand_2", "rand_3"]:
            max_errors_per_iter = []
            for k in range(5000):
                tmp_slots_assegnation_probabilities = []
                for elem in slots_assegnation_probabilities:
                    tmp_slots_assegnation_probabilities.append(elem.copy())
                constraints_error = [0] * len(self.categories)
                promenance_per_category = [0] * len(self.categories)
                result = self.de_randomize_LP(LP_news_pool, tmp_slots_assegnation_probabilities, tech)
                for i in range(len(result)):
                    category_index = self.categories.index(result[i].news_category)
                    promenance_per_category[category_index] += self.real_slot_promenances[i]

                for i in range(len(promenance_per_category)):
                    if promenance_per_category[i] < self.B[i] * -1:
                        constraints_error[i] += (self.B[i] * -1 - promenance_per_category[i]) / (self.B[i] * -1)

                max_errors_per_iter.append(np.max(constraints_error))
            if tech == "rand_1":
                self.rand_1_errors += max_errors_per_iter
            elif tech == "rand_2":
                self.rand_2_errors += max_errors_per_iter
            else:
                self.rand_3_errors += max_errors_per_iter

    def solve_linear_problem(self, continuity_relaxation=True):

        result = [0] * self.layout_slots
        self.news_pool.sort(key=lambda x: (x.news_category, x.sampled_quality), reverse=True)
        LP_news_pool = []
        done_for_category = False
        category_count = 0
        prev_category = self.news_pool[0].news_category
        for news in self.news_pool:
            if prev_category != news.news_category:
                category_count = 0
                done_for_category = False
                prev_category = news.news_category
            if not done_for_category:
                LP_news_pool.append(news)
                category_count += 1
            if category_count == self.layout_slots:
                done_for_category = True

        LP_news_pool.sort(key=lambda x: x.news_category, reverse=False)
        thetas = []
        for news in LP_news_pool:
            thetas += [news.sampled_quality] * self.layout_slots
        self.C = list(np.array(thetas) * np.array(self.lambdas))

        if continuity_relaxation:
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

            result = self.de_randomize_LP(LP_news_pool, slots_assegnation_probabilities, self.lp_rand_tech)

        else:
            LP = LpProblem("News_ILP", LpMaximize)
            LP_variables = []

            for cat in range(len(self.categories)):
                for j in range(self.layout_slots):
                    for s in range(self.layout_slots):
                        LP_variables.append(LpVariable(name=str(cat) + "-" + str(j) + "-" + str(s), lowBound=0, upBound=1, cat="Binary"))

            # Objective function addition to the problem
            C = list(np.array(self.C) * -1)
            LP += lpSum([C[i] * LP_variables[i] for i in range(len(self.C))])

            # Category constraints addition to the problem
            for i in range(len(self.categories)):
                LP += lpSum([self.A[i][j] * LP_variables[j] for j in range(len(self.C))]) <= self.B[i]

            # Slots capacity constraints addition to the problem
            for i in range(len(self.categories), len(self.categories) + self.layout_slots):
                LP += lpSum([self.A[i][j] * LP_variables[j] for j in range(len(self.C))]) <= self.B[i]

            # News capacity constraints addition to the problem
            for i in range(len(self.categories) + self.layout_slots, len(self.categories) + self.layout_slots + len(self.categories) * self.layout_slots):
                LP += lpSum([self.A[i][j] * LP_variables[j] for j in range(len(self.C))]) <= self.B[i]

            LP.solve()

            slots_assegnation_probabilities = []
            slot_counter = 0
            tmp_slot_probabilities = []
            while slot_counter < self.layout_slots:
                i = slot_counter
                while i < len(LP.variables()):
                    tmp_slot_probabilities.append(LP.variables().__getitem__(i))
                    i += self.layout_slots
                slots_assegnation_probabilities.append(tmp_slot_probabilities.copy())
                tmp_slot_probabilities.clear()
                slot_counter += 1

            for elem in slots_assegnation_probabilities:
                for k in elem:
                    if k.varValue > 0:
                        print(k.name, "=", k.varValue)
        return result

    def de_randomize_LP(self, LP_news_pool, tmp_slots_assignation_probabilities, de_rand_technique):
        result = [0] * self.layout_slots
        tmp_slot_promenances = self.real_slot_promenances.copy()
        feasible_news = [i for i in range(len(LP_news_pool))]
        slot_counter = 0
        allocated_slots = []
        while slot_counter < self.layout_slots:
            if (de_rand_technique == "rand_1") or (de_rand_technique == "rand_3"):
                target_slot = np.argmax(tmp_slot_promenances)
            else:
                tmp_slot_promenance_norm = list(np.array(tmp_slot_promenances) / sum(tmp_slot_promenances))
                target_slot_promenance = np.random.choice(tmp_slot_promenances, p=tmp_slot_promenance_norm)
                target_slot = tmp_slot_promenances.index(target_slot_promenance)

            target_slot_assegnation_probabilities = tmp_slots_assignation_probabilities[int(target_slot)]
            if de_rand_technique == "rand_3":
                for p in range(len(tmp_slots_assignation_probabilities)):
                    if (p not in allocated_slots) and (p != target_slot):
                        target_slot_assegnation_probabilities = \
                            list(np.array(target_slot_assegnation_probabilities) *
                                 (1 - np.array(tmp_slots_assignation_probabilities[p])))
                allocated_slots.append(target_slot)

            target_slot_assegnation_probabilities_norm = list(np.array(target_slot_assegnation_probabilities) /
                                                              sum(target_slot_assegnation_probabilities))
            selected_news = np.random.choice(feasible_news, p=target_slot_assegnation_probabilities_norm)
            result[int(target_slot)] = LP_news_pool[selected_news]
            deletion_index = feasible_news.index(selected_news)
            feasible_news.__delitem__(deletion_index)
            for probs in tmp_slots_assignation_probabilities:
                probs.__delitem__(deletion_index)
            tmp_slot_promenances[int(target_slot)] = 0
            slot_counter += 1

        return result


if __name__ == "__main__":

    # We fill the news pool with a bounch of news
    news_pool = []
    k = 0
    for category in ["cibo", "gossip", "politic", "scienza", "sport", "tech"]:
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
        a = NewsLearner(categories=["cibo", "gossip", "politic", "scienza", "sport", "tech"], layout_slots=6,
                        real_slot_promenances=[0.7, 0.8, 0.4, 0.5, 0.3, 0.1], allocation_approach="LP")

        a.fill_news_pool(news_list=news_pool, append=True)
        a.find_best_allocation(u)
        for i in range(300):
            print(i)
            a.user_arrival(u, interest_decay=False)  # we simulate 200 interactions per user
        result.append(a.multiple_arms_avg_reward)
        click_result.append(a.click_per_page)
        if exp == 0:
            a.weighted_betas_matrix[0][0].plot_distribution("politic")
            a.weighted_betas_matrix[0][1].plot_distribution("politic")
            a.weighted_betas_matrix[0][2].plot_distribution("politic")
            print(a.weighted_betas_matrix[0][0].category_per_slot_reward_count)
            print(a.weighted_betas_matrix[0][0].category_per_slot_assignment_count)
            print("--------------------------")
            print(a.weighted_betas_matrix[0][1].category_per_slot_reward_count)
            print(a.weighted_betas_matrix[0][1].category_per_slot_assignment_count)
            print("--------------------------")
            print(a.weighted_betas_matrix[1][2].category_per_slot_reward_count)
            print(a.weighted_betas_matrix[1][2].category_per_slot_assignment_count)
            print(u.last_news_clicked)
        exp += 1
        a.save_weighted_beta_matrices(desinence="ciaoo")

    # plt.plot(np.mean(result, axis=0))
    # plt.title("Reward - " + str(u.user_quality_measure))
    # plt.show()
    # plt.title("Regret - " + str(u.user_quality_measure))
    # plt.plot(np.cumsum(np.max(u.user_quality_measure) - np.array(np.mean(result, axis=0))))
    # plt.show()
    # plt.title("Page Clicks - " + str(u.user_quality_measure))
    # plt.plot(np.mean(click_result, axis=0))
    # plt.show()


