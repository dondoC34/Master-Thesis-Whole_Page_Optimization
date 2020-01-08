from synthetic_user import *
from weighted_beta_distribution import *
import scipy.optimize as opt
import time as time
from ads_news import *
from pulp import *
from tqdm import tqdm
import random


class NewsLearner:

    def __init__(self, categories=[], layout_slots=10, real_slot_promenances=[], news_column_pivot=[0.01, 1],
                 news_row_pivot=[1], allocation_approach="standard",
                 allocation_diversity_bounds=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1), lp_rand_technique="rand_1",
                 ads_slots=2, ads_real_slot_promenances=[0.7, 0.4], ads_allocation=True, maximize_for_bids=False):

        self.categories = categories  # params to be learnt, the categories of news and ads
        self.last_proposal_weights = np.ones(len(self.categories))  # used to speed up the process of rejecton sampl.
        self.multiple_arms_avg_reward = []  # here we store the values of each pulled arm, for the regret plot
        self.news_pool = []  # all available news are kept here
        self.ads_pool = []  # all available ads are kept here
        self.click_per_page = []
        self.times = []
        self.layout_slots = layout_slots  # the number of slots of a single page
        self.ads_slots = ads_slots
        self.ads_real_slot_promenances = ads_real_slot_promenances
        self.real_slot_promenances = real_slot_promenances  # The real values of slot promenance
        self.last_news_observed = []  # we consider here only the last 50 news observed
        # The number of times we assigned category k to slot i
        self.category_per_slot_assignment_count = np.zeros([len(self.categories), self.layout_slots])
        # The number of times we observed a positive reward for category k allocated in slot i
        self.category_per_slot_reward_count = np.zeros([len(self.categories), self.layout_slots])
        self.quality_parameters = np.ones([len(self.categories), 2])  # TS parameters for quality estimate
        self.promenance_parameters = np.ones([self.layout_slots, 2])  # TS parameters for slot prom. estimate
        self.weighted_betas_matrix = []
        self.ads_weighted_beta = WeightedBetaDistribution(categories=self.categories,
                                                          layout_slots=self.ads_slots,
                                                          real_slot_promenances=self.ads_real_slot_promenances)
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

        # Linear programming attributes for NEWS ALLOCATION
        self.allocation_approach = allocation_approach
        self.A = []
        self.B = list(-1 * np.array(allocation_diversity_bounds)) + [1] * (self.layout_slots + len(self.categories) * self.layout_slots)
        self.bounds = [(0, 1)] * len(self.categories) * self.layout_slots * self.layout_slots
        self.lambdas = []
        self.C = []
        self.rand_1_errors = []
        self.rand_2_errors = []
        self.rand_3_errors = []
        self.lp_rand_tech = lp_rand_technique
        """
        In the following, the proper initializations for the matrices and the vectors for the LP approach (news) are done.
        Only a small subset of news will be considered to be allocated with the linear problem. In particular, the 
        number of considered news is (num_of_category) * (num_of_slots_of_a_page)
        """
        for _ in range(len(self.categories) * self.layout_slots):
            self.lambdas += list(np.array(self.real_slot_promenances) * -1)

        # Category constraints creation and insertion into matrix A
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

        # Slots' capacity constraints creation and insertion into matrix A
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

        # Linear programming attributes for ADS ALLOCATION
        self.ads_allocation = ads_allocation
        self.ads_lambdas = []
        self.ads_A = []
        self.ads_B = ([1] * self.ads_slots) + ([1] * len(self.categories) * self.ads_slots) + \
                     ([0] * len(self.categories) * self.ads_slots)
        self.ads_C = []
        self.M = 1000  # Big constant value used in the ILP resolution
        self.maximize_for_bids = maximize_for_bids

        """
        In the following, the proper initializations for the matrices and the vectors for the LP approach (ads) are done.
        Only a small subset of ads will be considered to be allocated with the integer linear problem. In particular, the 
        number of considered ads is (num_of_category) * (num_of_ads_slots_of_a_page)
        """
        for _ in range(len(self.categories) * self.ads_slots):
            self.ads_lambdas += self.ads_real_slot_promenances

        # Slots' capacity constraints creation and insertion into matrix ads_A
        for i in range(self.ads_slots):
            row = [0] * (len(self.categories) * self.ads_slots * self.ads_slots)
            target_index = i
            while target_index < len(row):
                row[target_index] = 1
                target_index += self.ads_slots

            self.ads_A.append(row.copy())

        # Variables' capacity constraints creation and insertion into matrix ads_A
        initial_index = 0
        for _ in range(len(self.categories) * self.ads_slots):
            row = [0] * (len(self.categories) * self.ads_slots * self.ads_slots)
            row[initial_index:initial_index + self.ads_slots] = [1] * self.ads_slots
            self.ads_A.append(row.copy())
            initial_index += self.ads_slots

        # Competitors exclusion constraints creation and insertion into matrix ads_A.
        initial_index = 0
        var_initial_index = 0
        for _ in range(len(self.categories)):
            for _ in range(self.ads_slots):
                row = [0] * len(self.categories) * self.ads_slots * self.ads_slots
                row[initial_index:initial_index + self.ads_slots * self.ads_slots] = [1] * self.ads_slots * self.ads_slots
                row[var_initial_index:var_initial_index + self.ads_slots] = [self.M] * self.ads_slots
                var_initial_index += self.ads_slots
                self.ads_A.append(row.copy())
            initial_index += self.ads_slots * self.ads_slots

    def sample_quality(self, content, user, approach="standard", interest_decay=False):
        """
        Returns a sample for the proper weighted beta distribution
        :param content: The news for which we want a sample describing the probability the user clicks it
        :param user: The user itself
        :param approach: Use position_based_model", ignore the rest for now
        :param interest_decay: Whether to consider if the user already clicked the news or whether it has already seen
        it etc. If so, returns a sample from the corresponding beta, otherwise froma fixed beta.
        :return: A sample from a proper beta distribution considering the category of the news passed as parameter
        """
        if isinstance(content, News):
            category = content.news_category
            if approach == "standard":

                index = self.categories.index(category)
                return np.random.beta(a=self.quality_parameters[index][0], b=self.quality_parameters[index][1])
            elif approach == "position_based_model":

                if interest_decay:
                    # Determines which beta to pull from:
                    weighted_beta_matrix_posx, weighted_beta_matrix_posy = self.compute_position_in_learning_matrix(user,
                                                                                                                    content)
                    content.set_sampled_quality(value=self.weighted_betas_matrix[weighted_beta_matrix_posx][weighted_beta_matrix_posy].sample(category=category))
                else:
                    # Pulls from a fixed beta otherwise
                    content.set_sampled_quality(value=self.weighted_betas_matrix[0][0].sample(category=category))

        elif isinstance(content, Ad):
            category = content.ad_category
            return self.ads_weighted_beta.sample(category=category)

        else:
            raise RuntimeError("Type of content not recognized.")

    def compute_position_in_learning_matrix(self, user, news):
        """
        Observing the number of time the news has been allocated for user and the number of times the user already
        clicked the news, computes the position in the matrix of the corresponding weighted beta distribution
        :param user: The user itself
        :param news: The news itself
        :return: The coordinates of the corresponding beta in the weighted beta matrix
        """
        slot_promenance_cumsum = user.get_promenance_cumsum(news)
        total_num_of_clicks = user.get_amount_of_clicks(news)

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

        return weighted_beta_matrix_posx, weighted_beta_matrix_posy

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

    def ad_click(self, ad, slot_nr):

        self.ads_weighted_beta.click(ad, slot_nr)

        # Collect a positive / negative reward for the news(s) k click allocated in slot i(s)
    def news_click(self, content, user, clicked=True, slot_nr=[], interest_decay=False):
        """
        Communicates (update the parameters) to the corresponding weighted beta distribution that a news has been
        clicked.
        :param content: The clicked news
        :param user: The user that clicked
        :param clicked: If the news has been clicked (always yes is this application)
        :param slot_nr: The slot in which the news has been clicked
        :param interest_decay: Determines whether to communicate to the corresponding beta or to a fixed beta
        :return: Nothing
        """
        if clicked:
            category_index = self.categories.index(content.news_category)
            self.quality_parameters[category_index][0] += 1
            if len(slot_nr) > 0:
                if interest_decay:
                    # Computes the coordinates of the corresponding weighted beta dist. in the weighted beta matrix
                    weighted_beta_matrix_posx, weighted_beta_matrix_posy = self.compute_position_in_learning_matrix(user,
                                                                                                                        content)
                    self.weighted_betas_matrix[weighted_beta_matrix_posx][weighted_beta_matrix_posy].click(content, slot_nr[0])
                    alloc_index = user.get_promenance_cumsum(content, get_only_index=True)
                    click_index = user.get_amount_of_clicks(content, get_only_index=True)
                    # Update with the values of the temporary variables
                    user.last_news_in_allocation[alloc_index][1] = user.last_news_in_allocation[alloc_index][2]
                    user.last_news_clicked[click_index][1] = user.last_news_clicked[click_index][2]
                else:
                    # Otherwise update a fixed weighted beta matrix
                    self.weighted_betas_matrix[0][0].click(content, slot_nr[0])
        else:
            index = self.categories.index(content.news_category)
            self.quality_parameters[index][1] += 1

    def find_best_allocation(self, user, interest_decay=False, continuity_relaxation=True,
                             update_assignment_matrices=True):
        """
        For each news in the news pool set a news sample pulled from the corresponding beta distributions.
        Allocates the best news by adopting either the classic standard allocation approach (allocate best news starting
        from best slots) or the LP allocation approach, that makes use of a linear problem to solve the task. The
        linear problem approach takes into account also the variety of a page, making sure to give to each category
        a percentage of the total slot promenance of the page.
        :param user: The user to which we are presenting the page.
        :param interest_decay: Whether to pull from the corresponding weighted beta distribution or to pull from a fixed
        weighted beta distribution.
        :param continuity_relaxation: In case of linear problem approach, this variable discriminates between the
        continuity relaxation of the problem's variables, of to use binary variable (btw this option increases the
        complexity of the resolution, since it can be seen as an NP complete problem)
        :param update_assignment_matrices: Whether to update the corresponding weighted beta distribution with the
        performed allocations. Useful to be False only in case of testing the performances of a trained model.
        :return: A list of news corresponding to the allocation in the page. The order of the news in the list
        correspond to the order of the slots in which the news are allocated.
        """
        result_news_allocation = [0] * self.layout_slots
        for news in self.news_pool:
            self.sample_quality(content=news, user=user, approach="position_based_model", interest_decay=interest_decay)

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
            result_news_allocation = self.solve_linear_problem(continuity_relaxation=continuity_relaxation)

        else:
            raise RuntimeError("Allocation approach not recognized.")

        if update_assignment_matrices:
            # Update weighted betas parameters with the allocation results:
            for i in range(len(result_news_allocation)):
                if interest_decay:
                    weighted_beta_matrix_posx, weighted_beta_matrix_posy = self.compute_position_in_learning_matrix(user,
                                                                                                                    result_news_allocation[i])
                    self.weighted_betas_matrix[weighted_beta_matrix_posx][weighted_beta_matrix_posy].allocation(result_news_allocation[i], i)
                    assigned_slot_promenance = self.real_slot_promenances[i]
                    # In the following, save into the user "cookie" that the current news has been allocated for him.
                    # If present, update the counter properly, if not a new entry is added.
                    index = user.get_promenance_cumsum(result_news_allocation[i], get_only_index=True)
                    if index == -1:
                        # Entry not found
                        length = len(user.last_news_in_allocation)
                        inserted = False
                        if length >= 2:
                            for k in range(length - 1):
                                if (result_news_allocation[i].news_id > user.last_news_in_allocation[k][0]) and (
                                        result_news_allocation[i].news_id < user.last_news_in_allocation[k + 1][0]):
                                    user.last_news_in_allocation.insert(k + 1, [result_news_allocation[i].news_id, 0, assigned_slot_promenance])
                                    inserted = True
                            if not inserted:
                                if result_news_allocation[i].news_id < user.last_news_in_allocation[0][0]:
                                    user.last_news_in_allocation.insert(0, [result_news_allocation[i].news_id, 0, assigned_slot_promenance])
                                    inserted = True
                        elif length == 1:
                            if user.last_news_in_allocation[0][0] > result_news_allocation[i].news_id:
                                user.last_news_in_allocation.insert(0, [result_news_allocation[i].news_id, 0, assigned_slot_promenance])
                                inserted = True

                        if not inserted:
                            user.last_news_in_allocation.append([result_news_allocation[i].news_id, 0, assigned_slot_promenance])
                    else:
                        # Entry found
                        user.last_news_in_allocation[index][2] += assigned_slot_promenance
                else:
                    self.weighted_betas_matrix[0][0].allocation(result_news_allocation[i], i)

        return result_news_allocation

    # Adds news to the current pool
    def fill_news_pool(self, news_list, append=True):
        """
        Fills the news pool with a list of news. Always to be done before starting any process with this learner.
        :param news_list: The list of news itself-
        :param append: If true append each element of the list, otherwise copies the entire list
        :return: Nothing.
        """
        if append:
            for news in news_list:
                self.news_pool.append(news)
        else:
            self.news_pool = news_list

    def fill_ads_pool(self, ads_list, append=True):

        if append:
            for ad in ads_list:
                self.ads_pool.append(ad)
        else:
            self.ads_pool = ads_list

    # Keeps track on the last news observed by the user
    def observed_news(self, news_observed):

        for news in news_observed:
            self.last_news_observed.append(news.news_id)

    def find_ads_best_allocation(self, news_allocation):

        ads_allocation = self.solve_ads_integer_linear_problem(news_allocation=news_allocation)

        final_ads_allocation = []
        for ad in ads_allocation:
            if not ad.is_buyer():
                outcome = np.random.binomial(1, ad.sampled_quality)
                if outcome == 1:
                    final_ads_allocation.append(ad)
                    self.remove_ad_from_pool([ad])
                else:
                    ad.set_as_buyer()
            else:
                final_ads_allocation.append(ad)
                self.remove_ad_from_pool([ad])

        for i in range(len(final_ads_allocation)):
            self.ads_weighted_beta.allocation(final_ads_allocation[i], slot_index=i)

        return final_ads_allocation

    def user_arrival(self, user, interest_decay=False):
        """
        This method defines the procedure to be adopted when a user interacts with the site (and then with the learner).
        First finds the best page allocation for that user, by using a fixed or corresponding beta distributions.
        Collect then the user interactions with the page and update the corresponding beta distributions.
        Collect also the average quality of the page depending on the user tastes and the number of received clicks.
        :param user: The user itself
        :param interest_decay: Whether to pull from the corresponding weighted beta distribution or to pull from a fixed
        weighted beta distribution.
        :return: Nothing.
        """
        t1 = time.time()
        allocation = self.find_best_allocation(user=user, interest_decay=interest_decay)
        t2 = time.time()
        t3 = 0
        t4 = 0
        user_observation_probabilities = user.observation_probabilities(self.real_slot_promenances)
        arm_rewards = []
        page_clicks = 0

        for i in range(len(allocation)):
            outcome = np.random.binomial(1, user_observation_probabilities[i])
            arm_rewards.append(user.get_reward(allocation[i]))
            if outcome == 1:
                clicked = user.click_news(allocation[i], interest_decay=interest_decay)
                if clicked == 1:
                    self.news_click(content=allocation[i], slot_nr=[i], interest_decay=interest_decay, user=user)
                    page_clicks += 1
                else:
                    index = user.get_promenance_cumsum(allocation[i], get_only_index=True)
                    user.last_news_in_allocation[index][1] = user.last_news_in_allocation[index][2]
            elif interest_decay:
                index = user.get_promenance_cumsum(allocation[i], get_only_index=True)
                user.last_news_in_allocation[index][1] = user.last_news_in_allocation[index][2]

        if self.ads_allocation:
            t3 = time.time()
            ads_allocation = self.find_ads_best_allocation(news_allocation=allocation)
            t4 = time.time()
            for i in range(len(ads_allocation)):
                outcome = np.random.binomial(1, self.ads_real_slot_promenances[i])
                if outcome == 1:
                    clicked = user.click_ad(ads_allocation[i])
                    if clicked == 1:
                        self.ad_click(ad=ads_allocation[i],
                                      slot_nr=[i])

        self.times.append(t2 - t1 + t4 - t3)
        self.multiple_arms_avg_reward.append(np.mean(arm_rewards))
        self.click_per_page.append(page_clicks)

    def save_weighted_beta_matrices(self, desinence):
        """
        Saves in .txt files the content of all the weighted beta distribution present in the weighted beta matrix.
        Add a specific desinence to the file name in order to distinguish different learning matrices from different
        learners
        :param desinence: The desinence itself
        :return: Nothing
        """
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
        """
        Add a news into the news pool.
        :param news: The news itself
        :return: Nothing
        """
        self.news_pool.append(news)

    def read_weighted_beta_matrix_from_file(self, indexes, desinences, folder="Trained_betas_matrices/"):
        """
        Read the parameters of some weighted beta distribution from file. In particular, the wheighted betas that are
        going to be read are the ones in the matrix specified by the indexes touples present in the parameter "indexes".
        :param indexes: List of touples containing the indexes of the weighted beta distribution to be read from file
        :param desinences: A specific desinence of the file we are reading from
        :param folder: An eventual folder in which the files are contained. Specify "/" at the end of the folder name.
        :return: Nothing.
        """
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

    def remove_news_from_pool(self, news_list):
        """
        Remove all the news present in the news_list from the news pool.
        :param news_list: The news list itself.
        :return: Nothing.
        """
        for i in range(-len(self.news_pool), 0):
            if self.news_pool[i] in news_list:
                self.news_pool.__delitem__(i)

    def remove_ad_from_pool(self, ads_list):

        for i in range(-len(self.ads_pool), 0):
            if self.ads_pool[i] in ads_list:
                self.ads_pool.__delitem__(i)

    def measure_allocation_diversity_bounds_errors(self, slots_assegnation_probabilities, LP_news_pool, iter=5000):
        """
        This method only checks and collect data about how good are the three possible de-randomization techniques in
        respecting the diversity bounds formulated in the LP (with continuity relaxation). The data are collected by
        running "iter" number of derandomizations and are saved in the class attributes: "rand_1_errors",
        "rand_2_errors" and "rand_3_errors". The single error per derandomization is quantified as the mximum
        percentage of displacement bewteen the required and presented promenance per category.
        :param slots_assegnation_probabilities: The randomized solution of a LP.
        :param LP_news_pool: The restricted news pool used by the LP.
        :param iter: Number of the derandomization performed for each technique
        :return: Nothing.
        """
        for tech in ["rand_1", "rand_2", "rand_3"]:
            max_errors_per_iter = []
            for k in range(iter):
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

                max_errors_per_iter.append(np.mean(constraints_error))
            if tech == "rand_1":
                self.rand_1_errors += max_errors_per_iter
            elif tech == "rand_2":
                self.rand_2_errors += max_errors_per_iter
            else:
                self.rand_3_errors += max_errors_per_iter

    def solve_linear_problem(self, continuity_relaxation=True):
        """
        Solve a linear problem to find the best allocation for the current page.
        First selects a subset of "num_of_slots" news for each category.
        If there are not at least "num_of_slots" news for each category random news from the news pool will be chosen.
        this will lead the solution to be significantly worse. In real scenarios this case will never happen.
        Using the selected news solves the linear problem either with continuity relaxation of the variable or without
        it.
        :param continuity_relaxation: Whether to use an LP approach or an ILP approach.
        :return: A list of news corresponding to the allocation in the page. The order of the news in the list
        correspond to the order of the slots in which the news are allocated.
        """
        result = [0] * self.layout_slots
        self.news_pool.sort(key=lambda x: (x.news_category, x.sampled_quality), reverse=True)
        LP_news_pool = []
        done_for_category = False
        category_count = 0
        prev_category = self.news_pool[0].news_category
        # First build a subset of news to easily handle the LP resolution
        for news in self.news_pool:
            if prev_category != news.news_category:
                if category_count < self.layout_slots:
                    raise RuntimeWarning("Not enough news per category found. There should be at least " +
                                         str(self.layout_slots) + " news with category = " + prev_category + ", but "
                                         "only " + str(category_count) + "are present. The allocation maybe "
                                                                         "sub-optimal.")
                category_count = 0
                done_for_category = False
                prev_category = news.news_category
            if not done_for_category:
                LP_news_pool.append(news)
                category_count += 1
            if category_count == self.layout_slots:
                done_for_category = True

        # If not all the required news are present, add some other news at random.
        while len(LP_news_pool) < len(self.categories) * self.layout_slots:
            random_news = np.random.choice(self.news_pool)
            if random_news not in LP_news_pool:
                LP_news_pool.append(random_news)

        LP_news_pool.sort(key=lambda x: x.news_category, reverse=False)
        thetas = []
        # Compute the vector of coefficients for the LP objective function
        for news in LP_news_pool:
            thetas += [news.sampled_quality] * self.layout_slots
        self.C = list(np.array(thetas) * np.array(self.lambdas))

        # Then solve an LP or an ILP
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

            self.measure_allocation_diversity_bounds_errors(slots_assegnation_probabilities, LP_news_pool, iter=10)

            result = self.de_randomize_LP(LP_news_pool, slots_assegnation_probabilities, self.lp_rand_tech)

        else:
            # INITIALIZES AN INTEGER LINEAR PROBLEM
            ILP = LpProblem("News_ILP", LpMaximize)
            ILP_variables = []

            for cat in range(len(self.categories)):
                for j in range(self.layout_slots):
                    for s in range(self.layout_slots):
                        ILP_variables.append(LpVariable(name=str(cat) + "_" + str(j) + "_" + str(s), lowBound=0, upBound=1, cat="Binary"))

            # Objective function addition to the problem
            C = list(np.array(self.C) * -1)
            ILP += lpSum([C[i] * ILP_variables[i] for i in range(len(self.C))])

            # Category constraints addition to the problem
            for i in range(len(self.categories)):
                ILP += lpSum([self.A[i][j] * ILP_variables[j] for j in range(len(self.C))]) <= self.B[i]

            # Slots capacity constraints addition to the problem
            for i in range(len(self.categories), len(self.categories) + self.layout_slots):
                ILP += lpSum([self.A[i][j] * ILP_variables[j] for j in range(len(self.C))]) <= self.B[i]

            # News capacity constraints addition to the problem
            for i in range(len(self.categories) + self.layout_slots, len(self.categories) + self.layout_slots + len(self.categories) * self.layout_slots):
                ILP += lpSum([self.A[i][j] * ILP_variables[j] for j in range(len(self.C))]) <= self.B[i]

            ILP.solve()

            # FOR EACH SLOT, ISOLATES THE CORRESPONDING VARIABLES
            slots_assegnation_probabilities = []
            slot_counter = 0
            tmp_slot_probabilities = []
            while slot_counter < self.layout_slots:
                i = slot_counter
                while i < len(ILP.variables()):
                    tmp_slot_probabilities.append(ILP.variables().__getitem__(i))
                    i += self.layout_slots
                slots_assegnation_probabilities.append(tmp_slot_probabilities.copy())
                tmp_slot_probabilities.clear()
                slot_counter += 1

            # TAKES THE VARIABLES WHICH VALUE IS 1, THEN ALLOCATES THE CORRESPONDING NEWS IN THE RESULT PAGE
            for i in range(len(result)):
                for probabilities in slots_assegnation_probabilities[i]:
                    if probabilities.varValue > 0:
                        var_name = probabilities.name
                        break
                indexes = var_name.split("_")
                category_index = int(indexes[0])
                news_number = int(indexes[1])
                news_index = category_index * self.layout_slots + news_number
                result[i] = LP_news_pool[news_index]

        return result

    def solve_ads_integer_linear_problem(self, news_allocation):

        result = [0] * self.ads_slots
        category_percentage_in_allocation = [0] * len(self.categories)
        for i in range(len(news_allocation)):
            category_index = self.categories.index(news_allocation[i].news_category)
            category_percentage_in_allocation[category_index] += self.real_slot_promenances[i]
        category_percentage_in_allocation = list(np.array(category_percentage_in_allocation) / sum(self.real_slot_promenances))

        for ad in self.ads_pool:
            ad.set_sampled_quality(value=self.sample_quality(content=ad, user=None))
        if self.maximize_for_bids:
            self.ads_pool.sort(key=lambda x: (x.ad_category, x.sampled_quality * x.bid), reverse=True)
        else:
            self.ads_pool.sort(key=lambda x: (x.ad_category, x.sampled_quality), reverse=True)
        ads_ILP_news_pool = []
        done_for_category = False
        category_count = 0
        prev_category = self.ads_pool[0].ad_category
        # First build a subset of ads to easily handle the ILP resolution
        for ad in self.ads_pool:
            if prev_category != ad.ad_category:
                if category_count < self.ads_slots:
                    raise RuntimeWarning("Not enough news per category found. There should be at least " +
                                         str(self.ads_slots) + " news with category = " + prev_category + ", but "
                                         "only " + str(category_count) + "are present. The allocation maybe "
                                         "sub-optimal.")
                category_count = 0
                done_for_category = False
                prev_category = ad.ad_category
            if not done_for_category:
                ads_ILP_news_pool.append(ad)
                category_count += 1
            if category_count == self.ads_slots:
                done_for_category = True

        ads_ILP_news_pool.sort(key=lambda x: x.ad_category, reverse=False)
        thetas = []
        percentages = []
        # Compute the vector of coefficients for the LP objective function
        competitors_constraints_starting_index = self.ads_slots + len(self.categories) * self.ads_slots
        for i in range(len(ads_ILP_news_pool)):
            ad_category_index = self.categories.index(ads_ILP_news_pool[i].ad_category)
            ad_category_percentage = category_percentage_in_allocation[ad_category_index]
            if self.maximize_for_bids:
                thetas += [ads_ILP_news_pool[i].sampled_quality * ads_ILP_news_pool[i].bid] * self.ads_slots
            else:
                thetas += [ads_ILP_news_pool[i].sampled_quality] * self.ads_slots
            percentages += [ad_category_percentage] * self.ads_slots
            self.ads_B[competitors_constraints_starting_index + i] = self.M * (2 - ads_ILP_news_pool[i].exclude_competitors)

        self.ads_C = list(np.array(thetas) * np.array(self.ads_lambdas) * np.array(percentages))

        # INITIALIZES AN INTEGER LINEAR PROBLEM
        ILP = LpProblem("Ads_ILP", LpMaximize)
        ILP_variables = []

        for cat in range(len(self.categories)):
            for j in range(self.ads_slots):
                for s in range(self.ads_slots):
                    ILP_variables.append(
                        LpVariable(name=str(cat) + "_" + str(j) + "_" + str(s), lowBound=0, upBound=1, cat="Binary"))

        # Objective function addition to the problem
        ILP += lpSum([self.ads_C[i] * ILP_variables[i] for i in range(len(self.ads_C))])

        # Slots capacity constraints addition to the problem
        for i in range(self.ads_slots):
            ILP += lpSum([self.ads_A[i][j] * ILP_variables[j] for j in range(len(self.ads_C))]) <= self.ads_B[i]

        # Ads capacity constraints addition to the problem
        for i in range(self.ads_slots, self.ads_slots + len(self.categories) * self.ads_slots):
            ILP += lpSum([self.ads_A[i][j] * ILP_variables[j] for j in range(len(self.ads_C))]) <= self.ads_B[i]

        # Competitors exclusion constraints addition to the problem
        for i in range(self.ads_slots + len(self.categories) * self.ads_slots,
                       self.ads_slots + len(self.categories) * self.ads_slots +
                       len(self.categories) * self.ads_slots):
            ILP += lpSum([self.ads_A[i][j] * ILP_variables[j] for j in range(len(self.ads_C))]) <= self.ads_B[i]

        ILP.solve()

        # FOR EACH SLOT, ISOLATES THE CORRESPONDING VARIABLES
        slots_assegnation_probabilities = []
        slot_counter = 0
        tmp_slot_probabilities = []
        while slot_counter < self.ads_slots:
            i = slot_counter
            while i < len(ILP.variables()):
                tmp_slot_probabilities.append(ILP.variables().__getitem__(i))
                i += self.ads_slots
            slots_assegnation_probabilities.append(tmp_slot_probabilities.copy())
            tmp_slot_probabilities.clear()
            slot_counter += 1

        # TAKES THE VARIABLES WHICH VALUE IS 1, THEN ALLOCATES THE CORRESPONDING NEWS IN THE RESULT PAGE
        for i in range(len(result)):
            for probabilities in slots_assegnation_probabilities[i]:
                if probabilities.varValue > 0:
                    var_name = probabilities.name
                    break
            indexes = var_name.split("_")
            category_index = int(indexes[0])
            ad_number = int(indexes[1])
            ad_index = category_index * self.ads_slots + ad_number
            result[i] = ads_ILP_news_pool[ad_index]

        return result

    def de_randomize_LP(self, LP_news_pool, tmp_slots_assignation_probabilities, de_rand_technique):
        """
        Given a randomized solution provided by a LP or an ILP, provide a derandomization, finding then the actual
        allocation of the page. The de-randomization techniques that can be used are "rand_1", "rand_2" and "rand_3".
        :param LP_news_pool: The subset of news used by the linear problem.
        :param tmp_slots_assignation_probabilities: The randomized solution provided by the LP or ILP.
        :param de_rand_technique: the derandomization technique itself.
        :return: A list of news corresponding to the allocation in the page. The order of the news in the list
        correspond to the order of the slots in which the news are allocated.
        """
        result = [0] * self.layout_slots
        tmp_slot_promenances = self.real_slot_promenances.copy()
        feasible_news = [i for i in range(len(LP_news_pool))]
        slot_counter = 0
        allocated_slots = []
        while slot_counter < self.layout_slots:
            if (de_rand_technique == "rand_1") or (de_rand_technique == "rand_3"):
                # Start from the best slot
                target_slot = np.argmax(tmp_slot_promenances)
            else:
                # Start from slot j with probability proportional to j's slot promenance
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

            # Normalize the vector of the variable assigning to the target slot
            target_slot_assegnation_probabilities_norm = list(np.array(target_slot_assegnation_probabilities) /
                                                              sum(target_slot_assegnation_probabilities))
            # Choose the allocating news with probability proportional to the values of the variables
            selected_news = np.random.choice(feasible_news, p=target_slot_assegnation_probabilities_norm)
            # Insert the winner news in the allocation and repeat after removing the variables.
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
    ads_pool = []
    times = []
    exclude = [True, False]
    k = 0
    for category in ["cibo", "gossip", "politic", "scienza", "sport", "tech"]:
        for id in range(1, 101):
            news_pool.append(News(news_id=k,
                                  news_name=category + "-" + str(id)))
            k += 1
    for category in ["cibo", "gossip", "politic", "scienza", "sport", "tech"]:
        for id in range(1, 501):
            ads_pool.append(Ad(id, category + "-" + str(id), np.random.choice(exclude)))
            k += 1

    exp = 0
    dynamic_refill = False
    result = []
    click_result = []
    diversity_percentage_for_category = 5
    real_slot_promenances = [0.7, 0.8, 0.4, 0.5, 0.3, 0.1]
    promenance_percentage_value = diversity_percentage_for_category / 100 * sum(real_slot_promenances)
    allocation_diversity_bounds = (promenance_percentage_value, promenance_percentage_value) * 3

    # Then we perform 100 experiments and use the collected data to plot the regrets and distributions
    for k in tqdm(range(1)):
        # We create a user and set their quality metrics that we want to estimate
        u = SyntheticUser(23, "M", 27, "C")  # A male 27 years old user, that is transparent to slot promenances
        a = NewsLearner(categories=["cibo", "gossip", "politic", "scienza", "sport", "tech"], layout_slots=6,
                        real_slot_promenances=real_slot_promenances, allocation_approach="LP",
                        allocation_diversity_bounds=allocation_diversity_bounds, ads_allocation=True, maximize_for_bids=True)

        a.fill_news_pool(news_list=news_pool, append=True)
        a.fill_ads_pool(ads_list=ads_pool, append=True)
        print(len(a.ads_pool))
        # We simulate 300 interactions for this user
        for i in range(400):
            print(i)
            a.user_arrival(u, interest_decay=True)
            if dynamic_refill:
                if i + 1 % 5 == 0:
                    # UPDATE NEWS POOL
                    random.shuffle(a.news_pool)
                    news_to_be_removed = []
                    for cat in["cibo", "gossip", "politic", "scienza", "sport", "tech"]:
                        news_count = 0
                        for j in range(len(a.news_pool)):
                            if a.news_pool[j].news_category == cat:
                                news_to_be_removed.append(a.news_pool[j])
                                news_count += 1
                        if news_count == 3:
                            break

                    a.remove_news_from_pool(news_to_be_removed)

                    for j in range(3):
                        news_num = np.random.choice([index for index in range(100000)])
                        a.insert_into_news_pool(News(news_num, "cibo-" + str(news_num)))
                        a.insert_into_news_pool(News(news_num, "gossip-" + str(news_num)))
                        a.insert_into_news_pool(News(news_num, "politic-" + str(news_num)))
                        a.insert_into_news_pool(News(news_num, "scienza-" + str(news_num)))
                        a.insert_into_news_pool(News(news_num, "sport-" + str(news_num)))
                        a.insert_into_news_pool(News(news_num, "tech-" + str(news_num)))

        result.append(a.multiple_arms_avg_reward)
        click_result.append(a.click_per_page)
    print(a.ads_weighted_beta.category_per_slot_assignment_count)
    print(a.ads_weighted_beta.category_per_slot_reward_count)
    print(len(a.ads_pool))
    print("avg time " + str(np.mean(a.times) * 1000) + str(" ms"))
    print(a.weighted_betas_matrix[0][0].category_per_slot_assignment_count)
    print(a.weighted_betas_matrix[1][2].category_per_slot_reward_count)
    print(a.weighted_betas_matrix[1][2].category_per_slot_assignment_count)
    print(a.weighted_betas_matrix[0][2].category_per_slot_reward_count)
    print(a.weighted_betas_matrix[0][2].category_per_slot_assignment_count)
    exit(3)

    plt.plot(np.mean(result, axis=0))
    plt.title("Reward - " + str(u.user_quality_measure))
    plt.show()
    plt.title("Regret - " + str(u.user_quality_measure))
    plt.plot(np.cumsum(np.max(u.user_quality_measure) - np.array(np.mean(result, axis=0))))
    plt.show()
    plt.title("Page Clicks - " + str(u.user_quality_measure))
    plt.plot(np.mean(click_result, axis=0))
    plt.show()
    result = np.mean(result, axis=0)
    click_result = np.mean(click_result, axis=0)
    file = open("reward_decay_LP_frequentfrequent.txt", "w")
    file.write(str(result[0]))
    for i in range(1, len(result)):
        file.write("," + str(result[i]))
    file.close()
    file = open("clicks_decay_LP_frequentfrequent.txt", "w")
    file.write(str(click_result[0]))
    for i in range(1, len(click_result)):
        file.write("," + str(click_result[i]))
    file.close()


