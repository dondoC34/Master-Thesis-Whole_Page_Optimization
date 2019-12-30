from news_learner import *
from synthetic_user import *
from tqdm import tqdm


def save_allocation_errors(learners_list):
    """
    Saves the allocation diversity bounds max errors for a given list of learners relative to
    ALL THE 3 DE-RANDOMIZATION TECHNIQUES rand_1, rand_2 and rand_3.
    Three files ".txt" are saved containing the just mentioned info.
    Call only if the method "measure_allocation_diversity_bounds_errors" of each learner
    in the list has been called before.
    :return: nothing
    """
    file = open("perf_rand_1111.txt", "w")
    file2 = open("perf_rand_2222.txt", "w")
    file3 = open("perf_rand_3333.txt", "w")
    for learner in learners_list:
        if i % 3 == 1:
            file.write(str(learner.rand_1_errors[0]))
            file2.write(str(learner.rand_2_errors[0]))
            file3.write(str(learner.rand_3_errors[0]))
            for k in range(1, len(learner.rand_1_errors)):
                file.write("," + str(learner.rand_1_errors[k]))
                file2.write("," + str(learner.rand_2_errors[k]))
                file3.write("," + str(learner.rand_3_errors[k]))
            file.write(",")
            file2.write(",")
            file3.write(",")
    file.close()
    file2.close()
    file3.close()


def plot_allocation_errors():
    """
    Plots an histogram containing the amount of times a de-randomization techinque did that error (in percentage).
    The plot will contain the info about EACH de-randomization techinque.
    Call only if the "save_allocation_errors" has been called in precedence and its output files have been saved.
    :return: Nothing
    """
    final_result = []
    final_result2 = []
    final_result3 = []

    for des in ["", "1", "11", "111"]:
        file = open("perf_rand_1" + des + ".txt", "r")
        result = file.read().split(",")
        result.__delitem__(-1)
        result = list(map(float, result))
        final_result += result
    file.close()

    for des in ["", "2", "22", "222"]:
        file = open("perf_rand_2" + des + ".txt", "r")
        result2 = file.read().split(",")
        result2.__delitem__(-1)
        result2 = list(map(float, result2))
        final_result2 += result2
    file.close()

    for des in ["", "3", "33", "333"]:
        file = open("perf_rand_3" + des + ".txt", "r")
        result3 = file.read().split(",")
        result3.__delitem__(-1)
        result3 = list(map(float, result3))
        final_result3 += result3
    file.close()
    res = final_result
    res2 = final_result2
    res3 = final_result3
    sns.distplot(res, hist=False)
    sns.distplot(res2, hist=False)
    sns.distplot(res3, hist=False)
    plt.legend(labels=["a", "b", "c"])
    plt.title("DeRandomization Mean Error Distribution")
    plt.show()


if __name__ == "__main__":
    """
    Four learner are going to be intialized. Each learner uses a different technique to de-randomize the LP results 
    (except for the standard learner). The avg results of "iterations" experiments are shown in terms of avg quality 
    per page and avg number of category per page.
    Furthermore, the average slot promenance per category given in output by each learner is measured.     
    """

    real_slot_promenances = [0.7, 0.8, 0.7, 0.7, 0.6, 0.5, 0.5, 0.4, 0.3, 0.2]
    categories = ["cibo", "gossip", "politic", "scienza", "sport", "tech"]
    diversity_percentage_for_category = 1.5
    promenance_percentage_value = diversity_percentage_for_category / 100 * sum(real_slot_promenances)
    allocation_diversity_bounds = (promenance_percentage_value, promenance_percentage_value) * 3
    iterations = 0
    user = SyntheticUser(23, "M", 35, "C")
    news_per_category = 100
    learner_rand_1 = NewsLearner(categories=categories, layout_slots=10,
                                 real_slot_promenances=real_slot_promenances,
                                 allocation_approach="LP",
                                 lp_rand_technique="rand_1",
                                 allocation_diversity_bounds=allocation_diversity_bounds)

    learner_rand_2 = NewsLearner(categories=categories, layout_slots=10,
                                 real_slot_promenances=real_slot_promenances,
                                 allocation_approach="LP",
                                 lp_rand_technique="rand_2",
                                 allocation_diversity_bounds=allocation_diversity_bounds)

    learner_rand_3 = NewsLearner(categories=categories, layout_slots=10,
                                 real_slot_promenances=real_slot_promenances,
                                 allocation_approach="LP",
                                 lp_rand_technique="rand_3",
                                 allocation_diversity_bounds=allocation_diversity_bounds)

    standard_learner = NewsLearner(categories=categories, layout_slots=10,
                                   real_slot_promenances=real_slot_promenances,
                                   allocation_approach="standard",
                                   allocation_diversity_bounds=allocation_diversity_bounds)

    # READ THE WEIGHTED BETA MATRIX FROM A FILE TO HAVE THE BETAS DISTRIBUTION BE DIFFERENT FROM JUST A UNIFORM
    learner_rand_1.read_weighted_beta_matrix_from_file(indexes=[(0, 0)], desinences=["0-1de_rand_testing"], folder="")
    learner_rand_2.read_weighted_beta_matrix_from_file(indexes=[(0, 0)], desinences=["0-1de_rand_testing"], folder="")
    learner_rand_3.read_weighted_beta_matrix_from_file(indexes=[(0, 0)], desinences=["0-1de_rand_testing"], folder="")
    standard_learner.read_weighted_beta_matrix_from_file(indexes=[(0, 0)], desinences=["0-1de_rand_testing"], folder="")

    # CREATE AND FILL THE NEWS POOL OF EACH LEARNER
    news_pool = []
    k = 0
    for category in categories:
        for id in range(0, news_per_category):
            news_pool.append(News(news_id=k,
                                  news_name=category + "-" + str(id)))
            k += 1

    learner_rand_1.fill_news_pool(news_pool)
    learner_rand_2.fill_news_pool(news_pool)
    learner_rand_3.fill_news_pool(news_pool)
    standard_learner.fill_news_pool(news_pool)

    # METRICS USED TO DISPLAY THE RESULTS
    page_reward_rand_1 = []
    page_reward_rand_2 = []
    page_reward_rand_3 = []
    page_reward_standard = []
    page_reward_ilp = []
    page_diversity_rand_1 = []
    page_diversity_rand_2 = []
    page_diversity_rand_3 = []
    page_diversity_ilp = []
    page_diversity_standard = []
    allocated_promenance_per_category_rand_1 = [0] * len(categories)
    allocated_promenance_per_category_rand_2 = [0] * len(categories)
    allocated_promenance_per_category_rand_3 = [0] * len(categories)
    allocated_promenance_per_category_standard = [0] * len(categories)
    allocated_promenance_per_category_ilp = [0] * len(categories)
    allocations_count_rand_1 = 0
    allocations_count_rand_2 = 0
    allocations_count_rand_3 = 0
    allocations_count_standard = 0
    allocations_count_ilp = 0

    # FOR EACH LEARNER, ALLOCATE A PAGE THEN COLLECT THE MEASURES
    for i in tqdm(range(1, iterations + 1)):

        news_category_in_page = [0] * len(categories)
        allocation_rewards = []
        if i % 3 == 0:
            allocation = learner_rand_1.find_best_allocation(user=user, update_assignment_matrices=False)
            allocations_count_rand_1 += 1
            for elem in allocation:
                click, reward = user.click_news(elem)
                allocation_rewards.append(reward)
                category_index = categories.index(elem.news_category)
                news_category_in_page[category_index] = 1
                news_slot = allocation.index(elem)
                allocated_promenance_per_category_rand_1[category_index] += real_slot_promenances[news_slot]

            page_reward_rand_1.append(sum(np.array(allocation_rewards) * np.array(real_slot_promenances)))
            page_diversity_rand_1.append(sum(news_category_in_page))

        elif i % 3 == 1:
            allocation = learner_rand_2.find_best_allocation(user=user, update_assignment_matrices=False)
            allocations_count_rand_2 += 1
            for elem in allocation:
                click, reward = user.click_news(elem)
                allocation_rewards.append(reward)
                category_index = categories.index(elem.news_category)
                news_category_in_page[category_index] = 1
                news_slot = allocation.index(elem)
                allocated_promenance_per_category_rand_2[category_index] += real_slot_promenances[news_slot]

            page_reward_rand_2.append(sum(np.array(allocation_rewards) * np.array(real_slot_promenances)))
            page_diversity_rand_2.append(sum(news_category_in_page))
        elif i % 4 == 0:
            allocation = standard_learner.find_best_allocation(user=user, update_assignment_matrices=False)
            allocations_count_standard += 1
            for elem in allocation:
                click, reward = user.click_news(elem)
                allocation_rewards.append(reward)
                category_index = categories.index(elem.news_category)
                news_category_in_page[category_index] = 1
                news_slot = allocation.index(elem)
                allocated_promenance_per_category_standard[category_index] += real_slot_promenances[news_slot]

            page_reward_standard.append(sum(np.array(allocation_rewards) * np.array(real_slot_promenances)))
            page_diversity_standard.append(sum(news_category_in_page))
        elif i % 5 == 0:
            allocation = learner_rand_2.find_best_allocation(user=user, update_assignment_matrices=False,
                                                             continuity_relaxation=False)
            allocations_count_ilp += 1
            for elem in allocation:
                click, reward = user.click_news(elem)
                allocation_rewards.append(reward)
                category_index = categories.index(elem.news_category)
                news_category_in_page[category_index] = 1
                news_slot = allocation.index(elem)
                allocated_promenance_per_category_ilp[category_index] += real_slot_promenances[news_slot]

            page_reward_ilp.append(sum(np.array(allocation_rewards) * np.array(real_slot_promenances)))
            page_diversity_ilp.append(sum(news_category_in_page))
        else:
            allocation = learner_rand_3.find_best_allocation(user=user, update_assignment_matrices=False)
            allocations_count_rand_3 += 1
            for elem in allocation:
                click, reward = user.click_news(elem)
                allocation_rewards.append(reward)
                category_index = categories.index(elem.news_category)
                news_category_in_page[category_index] = 1
                news_slot = allocation.index(elem)
                allocated_promenance_per_category_rand_3[category_index] += real_slot_promenances[news_slot]

            page_reward_rand_3.append(sum(np.array(allocation_rewards) * np.array(real_slot_promenances)))
            page_diversity_rand_3.append(sum(news_category_in_page))

    # PRINT THE COLLECTED MEASURES, AFTER AVERAGING THEM
    print("Rand_1 quality metrics:")
    print("Avg page reward: " + str(np.mean(page_reward_rand_1)))
    print("Avg page diversity: " + str(np.mean(page_diversity_rand_1)))
    print("--------------------------------")
    print("Rand_2 quality metrics:")
    print("Avg page reward: " + str(np.mean(page_reward_rand_2)))
    print("Avg page diversity: " + str(np.mean(page_diversity_rand_2)))
    print("--------------------------------")
    print("Rand_3 quality metrics:")
    print("Avg page reward: " + str(np.mean(page_reward_rand_3)))
    print("Avg page diversity: " + str(np.mean(page_diversity_rand_3)))
    print("--------------------------------")
    print("Standard quality metrics:")
    print("Avg page reward: " + str(np.mean(page_reward_standard)))
    print("Avg page diversity: " + str(np.mean(page_diversity_standard)))
    print("--------------------------------")
    print("ILP allocation quality metrics:")
    print("Avg page reward: " + str(np.mean(page_reward_ilp)))
    print("Avg page diversity: " + str(np.mean(page_diversity_ilp)))
    print("--------------------------------")
    print("Allocation category lower bounds: " + str(allocation_diversity_bounds))
    print("Rand_1 Avg promenance per category: " + str(np.array(allocated_promenance_per_category_rand_1) * 1 / allocations_count_rand_1))
    print("Rand_2 Avg promenance per category: " + str(
        np.array(allocated_promenance_per_category_rand_2) * 1 / allocations_count_rand_2))
    print("Rand_3 Avg promenance per category: " + str(
        np.array(allocated_promenance_per_category_rand_3) * 1 / allocations_count_rand_3))
    print("Standard Avg promenance per category: " + str(
        np.array(allocated_promenance_per_category_standard) * 1 / allocations_count_standard))

