from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import string
import simplejson
from news_learner import *
from numpy import random


counter = [0]
user_codes = []
learners = []
timestamps = []
iterations = []
user_data = []
real_slot_promenances = [0.9, 0.8, 0.8, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 0.4, 0.5, 0.5, 0.4, 0.2, 0.3, 0.3, 0.1]
categories=["cibo", "gossip", "politic", "scienza", "sport", "tech"]
diversity_percentage = 7.5
diversity_percentage_for_category = diversity_percentage / 100 * sum(real_slot_promenances)
allocation_diversity_bounds = (diversity_percentage_for_category, diversity_percentage_for_category) * 3
news_pool = []
extended_news_pool = []
for _ in range(len(categories)):
    extended_news_pool.append([])

k = 0
for category in categories:
    for id in range(1, 61):
        news_pool.append(News(news_id=k,
                              news_name=category + "-" + str(id)))
        k += 1

        if category + "-" + str(id) in ["cibo-1", "cibo-6", "cibo-13", "cibo-17",
                                        "gossip-14",
                                        "politic-5", "politic-19",
                                        "scienza-6", "scienza-11",
                                        "sport-1", "sport-6", "sport-8", "sport-19", "sport-20",
                                        "tech-4", "tech-10", "tech-14", "tech-20", "cibo-41", "cibo-42",
                                        "gossip-41", "gossip-42", "politic-41", "politic-42", "politic-43", "politic-44",
                                        "politic-45", "scienza-41", "scienza-42", "scienza-43", "scienza-44", "scienza-45",
                                        "sport-41", "sport-42", "sport-43", "sport-44", "sport-45", "sport-46", "tech-41", "tech-42",
                                        "tech-43", "tech-44", "tech-45",
                                        "scienza-61", "scienza-62", "sport-61", "sport-62", "sport-63", "sport-64",
                                        "sport-65", "tech-61", "politic-61", "cibo-61", "cibo-62", "cibo-63", "cibo-64"
                                        ]:

            news_pool.__delitem__(-1)

k = 0
for category in ["sport", "cibo", "tech", "politic", "gossip", "scienza"]:
    for id in range(1, 61):

        if category + "-" + str(id) in ["cibo-1", "cibo-6", "cibo-13", "cibo-17",
                                        "gossip-14",
                                        "politic-5", "politic-19",
                                        "scienza-6", "scienza-11",
                                        "sport-1", "sport-6", "sport-8", "sport-19", "sport-20",
                                        "tech-4", "tech-10", "tech-14", "tech-20", "cibo-41", "cibo-42",
                                        "gossip-41", "gossip-42", "politic-41", "politic-42", "politic-43", "politic-44",
                                        "politic-45", "scienza-41", "scienza-42", "scienza-43", "scienza-44", "scienza-45",
                                        "sport-41", "sport-42", "sport-43", "sport-44", "sport-45", "sport-46", "tech-41", "tech-42",
                                        "tech-43", "tech-44", "tech-45",
                                        "scienza-61", "scienza-62", "sport-61", "sport-62", "sport-63", "sport-64",
                                        "sport-65", "tech-61", "politic-61", "cibo-61", "cibo-62", "cibo-63", "cibo-64"]:

            news = News(news_id=id, news_name=category + "-" + str(id))
            news_index = categories.index(news.news_category)
            extended_news_pool[news_index].append(news)


def encode_html(html_file):
    lines = open(html_file, "r").readlines()
    result = ""
    for line in lines:
        result += line

    return result.encode()


def encode_news_page(html_file, user_id, news_list):
    lines = open(html_file, "r").readlines()
    result = ""
    news_names = []
    for elem in news_list:
        news_names.append(elem.news_name)

    for line in lines:
        result += line

    result = result[0:5489 + 10] + "'" + str(user_id) + "'" + result[5489 + 10::]
    result = result[0:5662 + 12] + str(news_names) + result[5662 + 12::]
    return result.encode()

def key_gen(length):
     values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "d", "e"]
     final_values = np.random.choice(values, size=length)
     key = ""
     for i in range(length):
        key += str(final_values[i])

     return key


class RequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        if self.path.endswith("/"):
            self.send_header("content-type", "text/html")
            self.end_headers()
            response = encode_html("intro.html")
            self.wfile.write(response)
        elif self.path.endswith("/instructions"):
            self.send_header("content-type", "text/html")
            self.end_headers()
            response = encode_html("intro_instructions.html")
            self.wfile.write(response)
        elif self.path.endswith("/credits"):
            self.send_header("content-type", "text/html")
            self.end_headers()
            response = encode_html("credits.html")
            self.wfile.write(response)
        elif self.path.endswith("/image"):
            image_path = self.path.split("/")
            image_path = image_path[1] + "/" + image_path[2]
            statinfo = os.stat(image_path)
            img_size = statinfo.st_size
            self.send_header("Content-type", "image/png")
            self.send_header("Content-length", img_size)
            self.end_headers()
            f = open(image_path, 'rb')
            self.wfile.write(f.read())
            f.close()

        elif self.path.endswith("/get_started"):
            self.send_header("content-type", "text/html")
            self.end_headers()
            user_key = key_gen(16)
            while user_key in user_codes:
                user_key = key_gen(16)

            user_codes.append(user_key)
            iterations.append(0)
            user_data.append([])
            for _ in range(5):
                user_data[-1].append([])
            learners.append(NewsLearner(categories=categories,
                                        layout_slots=len(real_slot_promenances),
                                        real_slot_promenances=real_slot_promenances,
                                        allocation_approach="LP",
                                        allocation_diversity_bounds=allocation_diversity_bounds,
                                        ads_allocation=False))
            learners[-1].fill_news_pool(news_list=news_pool, append=False)
            allocation = learners[-1].find_best_allocation(interest_decay=False, user=None)
            user_data[-1][0].append(allocation.copy())
            cat_index = categories.index(allocation[0].news_category)
            allocation[0] = np.random.choice(extended_news_pool[cat_index])
            response = encode_news_page("news_page.html", user_key, allocation)
            self.wfile.write(response)
            current_time = time.time()
            timestamps.append(current_time)
            deletion_indexes = []
            for i in range(len(timestamps)):
                if current_time - timestamps[i] > 1800:
                    deletion_indexes.append(i)
            deletion_indexes.sort(reverse=True)
            for elem in deletion_indexes:
                learners.__delitem__(elem)
                timestamps.__delitem__(elem)
                user_codes.__delitem__(elem)
                iterations.__delitem__(elem)
                user_data.__delitem__(elem)

        elif self.path.endswith("/next"):
            user_key = self.path.split("/")[1]
            self.send_header("content-type", "text/html")
            self.end_headers()
            try:
                user_index = user_codes.index(user_key)
                iterations[user_index] += 1
                if iterations[user_index] < 10:
                    allocation = learners[user_index].find_best_allocation(interest_decay=False, user=None)
                    user_data[user_index][0].append(allocation.copy())
                    cat_index = categories.index(allocation[0].news_category)
                    allocation[0] = np.random.choice(extended_news_pool[cat_index])
                    response = encode_news_page("news_page.html", user_key, allocation)
                    self.wfile.write(response)
                    timestamps[user_index] = time.time()
                else:
                    response = encode_html("end_page.html")
                    self.wfile.write(response)
                    file = open("WebApp_Results/no_clustering_results.txt", "a")
                    user_data_clicks = user_data[user_index][2]
                    file.write(str(user_data_clicks[0]))
                    for i in range(1, len(user_data_clicks)):
                        file.write("," + str(user_data_clicks[i]))
                    file.write("-")
                    j = 0
                    user_data_clicked_cats = user_data[user_index][1]
                    for page_clicked_cats in user_data_clicked_cats:
                        file.write(str(page_clicked_cats[0]))
                        for i in range(1, len(page_clicked_cats)):
                            file.write("," + str(page_clicked_cats[i]))
                        j += 1
                        if j < len(user_data_clicked_cats):
                            file.write(";")
                    file.write("-")
                    j = 0
                    user_data_allocations = user_data[user_index][0]
                    for page_allocation in user_data_allocations:
                        file.write(str(page_allocation[0].news_category))
                        for i in range(1, len(page_allocation)):
                            file.write("," + str(page_allocation[i].news_category))
                        j += 1
                        if j < len(user_data_allocations):
                            file.write(";")
                    file.write("-")
                    user_data_inspection = user_data[user_index][3]
                    file.write(str(user_data_inspection[0]))
                    for i in range(1, len(user_data_inspection)):
                        file.write("," + str(user_data_inspection[i]))
                    file.write("-")
                    j = 0
                    user_data_img_times = user_data[user_index][4]
                    for page_insp_times in user_data_img_times:
                        file.write(str(page_insp_times[0]))
                        for i in range(1, len(page_insp_times)):
                            file.write("," + str(page_insp_times[i]))
                        j += 1
                        if j < len(user_data_img_times):
                            file.write(";")

                    file.write("\n")
                    file.close()

            except ValueError:
                response = encode_html("session_expired_page.html")
                self.wfile.write(response)

        elif self.path.endswith("/end"):
            self.send_header("content-type", "text/html")
            self.end_headers()
            response = encode_html("end_page.html")
            self.wfile.write(response)
        elif self.path.endswith("/expired"):
            self.send_header("content-type", "text/html")
            self.end_headers()
            response = encode_html("session_expired_page.html")
            self.wfile.write(response)
        else:
            self.send_header("content-type", "text/html")
            self.end_headers()
            response = encode_html("zanero_page.html")
            self.wfile.write(response)

    def do_POST(self):
        data_string = self.rfile.read(int(self.headers['Content-Length']))
        data = simplejson.loads(data_string)
        user_id = data["id"]
        user_clicks = data["clicked"]
        try:
            user_index = user_codes.index(user_id)
            user_alloc = user_data[user_index][0][-1]

            num_of_clicks = 0
            clicked_elements = []
            for i in range(len(user_clicks)):
                if user_clicks[i]:
                    num_of_clicks += 1
                    clicked_elements.append(user_alloc[i].news_category)
                    learners[user_index].news_click(user_alloc[i], user=None, slot_nr=[i], interest_decay=False)
                else:
                    clicked_elements.append(0)

            self.send_response(200)
            self.send_header("content-type", "text/html")
            self.end_headers()
            self.wfile.write(str(user_id).encode())

            user_data[user_index][1].append(clicked_elements.copy())
            user_data[user_index][2].append(num_of_clicks)
            user_data[user_index][3].append(data["inspection_time"])
            user_data[user_index][4].append(data["image_inspection_times"])

        except ValueError:
            self.send_response(200)
            self.send_header("content-type", "text/html")
            self.end_headers()
            self.wfile.write("expired".encode())


if __name__ == "__main__":
    PORT = 46765
    server = HTTPServer(("", PORT), RequestHandler)
    print("server running")
    server.serve_forever()
