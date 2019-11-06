
class News:

    def __init__(self, news_id, news_name):

        self.news_id = news_id
        self.news_name = news_name
        self.news_category = news_name.split("-")[0]
        self.sampled_quality = 0
        self.image_path = "News-AdsApp - Copia/" + news_name + ".gif"

    def set_sampled_quality(self, value):
        self.sampled_quality = value


class Ad:

    def __init__(self, ad_id, ad_name):

        self.ad_id = ad_id
        self.ad_name = ad_name
        self.buyer = 0
        self.sampled_quality = 0
        self.ad_category = ad_name.split("-")[0]
        self.image_path = "Ads-AdsApp/" + ad_name + ".gif"

    def set_sampled_quality(self, value):
        self.sampled_quality = value

    def set_as_buyer(self):
        self.buyer = 1

    def is_buyer(self):
        if self.buyer == 0:
            return False
        else:
            return True
