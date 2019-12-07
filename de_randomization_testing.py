from news_learner import *
from synthetic_user import *
from tqdm import tqdm


if __name__ == "__main__":

    real_slot_promenances = [0.7, 0.8, 0.7, 0.7, 0.6, 0.5, 0.5, 0.4, 0.3, 0.2]
    categories = ["cibo", "gossip", "politic", "scienza", "sport", "tech"]
    diversity_percentage_for_category = 5
    promenance_percentage_value = diversity_percentage_for_category / 100 * sum(real_slot_promenances)
    allocation_diversity_bounds = (promenance_percentage_value, promenance_percentage_value) * 3
    iteration_per_learner = 1000
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

    learner_rand_1.read_weighted_beta_matrix_from_file(indexes=[(0, 0)], desinences=["0-1de_rand_testing"], folder="")
    learner_rand_2.read_weighted_beta_matrix_from_file(indexes=[(0, 0)], desinences=["0-1de_rand_testing"], folder="")
    learner_rand_3.read_weighted_beta_matrix_from_file(indexes=[(0, 0)], desinences=["0-1de_rand_testing"], folder="")
    standard_learner.read_weighted_beta_matrix_from_file(indexes=[(0, 0)], desinences=["0-1de_rand_testing"], folder="")

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

    page_reward_rand_1 = []
    page_reward_rand_2 = []
    page_reward_rand_3 = []
    page_reward_standard = []
    page_diversity_rand_1 = []
    page_diversity_rand_2 = []
    page_diversity_rand_3 = []
    page_diversity_standard = []

    for i in tqdm(range(1, iteration_per_learner + 1)):

        news_category_in_page = [0] * len(categories)
        allocation_rewards = []
        if i % 3 == 0:
            allocation = learner_rand_1.find_best_allocation(user=user, update_assignment_matrices=False)
            for elem in allocation:
                click, reward = user.click_news(elem)
                allocation_rewards.append(reward)
                category_index = categories.index(elem.news_category)
                news_category_in_page[category_index] = 1

            page_reward_rand_1.append(sum(np.array(allocation_rewards) * np.array(real_slot_promenances)))
            page_diversity_rand_1.append(sum(news_category_in_page))

        elif i % 3 == 1:
            allocation = learner_rand_2.find_best_allocation(user=user, update_assignment_matrices=False)
            for elem in allocation:
                click, reward = user.click_news(elem)
                allocation_rewards.append(reward)
                category_index = categories.index(elem.news_category)
                news_category_in_page[category_index] = 1

            page_reward_rand_2.append(sum(np.array(allocation_rewards) * np.array(real_slot_promenances)))
            page_diversity_rand_2.append(sum(news_category_in_page))
        elif i % 4 == 0:
            allocation = standard_learner.find_best_allocation(user=user, update_assignment_matrices=False)
            for elem in allocation:
                click, reward = user.click_news(elem)
                allocation_rewards.append(reward)
                category_index = categories.index(elem.news_category)
                news_category_in_page[category_index] = 1

            page_reward_standard.append(sum(np.array(allocation_rewards) * np.array(real_slot_promenances)))
            page_diversity_standard.append(sum(news_category_in_page))
        else:
            allocation = learner_rand_3.find_best_allocation(user=user, update_assignment_matrices=False)
            for elem in allocation:
                click, reward = user.click_news(elem)
                allocation_rewards.append(reward)
                category_index = categories.index(elem.news_category)
                news_category_in_page[category_index] = 1

            page_reward_rand_3.append(sum(np.array(allocation_rewards) * np.array(real_slot_promenances)))
            page_diversity_rand_3.append(sum(news_category_in_page))

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

