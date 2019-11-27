from news_learner import *
from synthetic_user import *
from ads_news import *


class PageCreator:
    def __init__(self, classes, real_slot_promenances, layout_slots, allocation_approach, categories):
        self.categories = categories
        self.real_slot_promenances = real_slot_promenances
        self.classes = classes
        self.learner_matrix = []
        for attribute in self.classes:
            attribute_row = []
            for _ in attribute:
                attribute_row.append(NewsLearner(real_slot_promenances=real_slot_promenances,
                                                 layout_slots=layout_slots,
                                                 allocation_approach=allocation_approach,
                                                 categories=categories))

            self.learner_matrix.append(attribute_row.copy())

    def fill_all_news_pool(self, pool):
        for learner_row in self.learner_matrix:
            for learner in learner_row:
                learner.fill_news_pool(pool, append=True)

    def user_interaction(self, user):
        pass
