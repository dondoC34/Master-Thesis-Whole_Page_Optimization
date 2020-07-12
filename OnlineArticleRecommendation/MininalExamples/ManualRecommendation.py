from OnlineArticleRecommendation.Core.news_learner import NewsLearner
from OnlineArticleRecommendation.Core.ads_news import News
from random import random


news_pool = []
categories = ["food", "gossip", "politic", "science", "sport", "tech"]
real_slot_promenances = [0.9, 0.8, 0.7, 0.8, 0.5, 0.4, 0.5, 0.4, 0.3, 0.1, 0.5, 0.3, 0.4, 0.2, 0.5, 0.6, 0.2, 0.1, 0.7]
diversity_percentage_for_category = 1
promenance_percentage_value = diversity_percentage_for_category / 100 * sum(real_slot_promenances)
allocation_diversity_bounds = (promenance_percentage_value, promenance_percentage_value) * 3

# CREATE A SET OF NEWS TO FEED THE AGENT
k = 0
for category in categories:
    for id in range(1, 101):
        news_pool.append(News(news_id=k,
                              news_name=category + "-" + str(id)))
        k += 1

agent = NewsLearner(categories=categories,
                    real_slot_promenances=real_slot_promenances,
                    allocation_approach="LP",  # Use an Optimized Linear Programming formulation to build the page
                    ads_allocation=False,
                    allocation_diversity_bounds=allocation_diversity_bounds)
agent.fill_news_pool(news_list=news_pool, append=True)

for _ in range(10):  # Generate 10 pages
    page_allocation = agent.find_best_allocation(user=None)  # Do not feed a Syntethic user

    for article in page_allocation:
        if random() < 0.4:  # Assume to click the 40% of the news of a page
            agent.news_click(content=article,
                             user=None,
                             slot_nr=[page_allocation.index(article)])

for category in categories:
    # Plot posteriors for each category
    agent.weighted_betas_matrix[0][0].plot_distribution(category=category)
