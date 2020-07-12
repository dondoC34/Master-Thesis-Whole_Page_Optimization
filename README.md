# Master-Thesis-Whole_Page_Optimization
This repository constains my master thesis project concerning Whole-Page Optimization. The key idea id the automatic generation of newspapers' webpages 
through a combinatorial semi-bandit approach.  
news_learner.py includes the core of the project. It is already provided a main section in which a simple example of how to use my Agent is present.  
web_server.py implements a simple webserver hosting my Agent as well as a set of images, used to simulated a real newspaper website i adopted to collect real world data.  
The rest of the files include saved data from offline evaluation sections, html pages for the WebServer and support python file used to properly implement the Agent.

# Minimal Working Example for Simulated Experimentation
With few lines of code we can set up a simple Agent for article recommendation. 

```python
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from OnlineArticleRecommendation.Core.news_learner import NewsLearner
from OnlineArticleRecommendation.Core.ads_news import News
from OnlineArticleRecommendation.Core.synthetic_user import SyntheticUser


news_pool = []
categories = ["food", "gossip", "politic", "science", "sport", "tech"]
real_slot_promenances = [0.9, 0.8, 0.7, 0.8, 0.5, 0.4, 0.5, 0.4, 0.3, 0.1]
expected_reward = []
click_amount = []
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

# AVERAGE OVER 10 EXPERIMENTS
for k in tqdm(range(10)):
    # We create a user and set their quality metrics that we want to estimate
    u = SyntheticUser(user_id=23,
                      genre="M",
                      age=27)  # A male 27 years old user
    # We manually set its parameters, even if it is not needed
    u.user_quality_measure = [0.3, 0.65, 0.35, 0.3, 0.2, 0.1]
    agent = NewsLearner(categories=categories,
                        real_slot_promenances=real_slot_promenances,
                        allocation_approach="LP",  # Use an Optimized Linear Programming formulation to build the page
                        ads_allocation=False,
                        allocation_diversity_bounds=allocation_diversity_bounds)

    agent.fill_news_pool(news_list=news_pool, append=True) #  Fill the Articles' bucket of the Agent
    # We simulate 100 interactions for this user
    for i in range(100):
        agent.user_arrival(u, interest_decay=False)  # Assume that user's interests do not vary over time

    expected_reward.append(agent.multiple_arms_avg_reward)
    click_amount.append(agent.click_per_page)

for category in categories:
    # Plot the posteriors of each category
    agent.weighted_betas_matrix[0][0].plot_distribution(category=category)

# Plot also the expected reward of the Agent
plt.plot(np.mean(expected_reward, axis=0))
plt.title("Expected Reward")
plt.xlabel("Interaction")
plt.show()
```

# Minimal Working Example for Practical Experimentation
Associate to each News instance a filePath for the image of the article. Allocate an article page and display the corresponding articles. Each click feedback can be communicated to the Agent by means of the method *news_ckick*.

```python
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
```
