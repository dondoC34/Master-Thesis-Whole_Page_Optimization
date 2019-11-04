
class News:

    def __init__(self, news_id, news_name, large=False):

        self.news_id = news_id
        self.news_name = news_name
        self.news_category = news_name.split("-")[0]
        self.sampled_quality = 0
        self.image_path = "News-AdsApp - Copia/" + news_name + ".gif"
        self.large = large

    def set_sampled_quality(self, value):
        self.sampled_quality = value

