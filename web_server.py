from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import string
import simplejson
from news_learner import *
from numpy import random
from telegram.bot import TelegramBot
import numbers
from socketserver import ThreadingMixIn
import threading

last_visit = [0.0]
last_visit_lock = threading.Lock()
user_data_lock = threading.Lock()
timestamps_lock = threading.Lock()
file_saving_lock = threading.Lock()
user_codes = []
learners = []
timestamps = []
news_pool_cat_sorted = [[], [], [], [], [], []]
big_news_pool_cat_sorted = [[], [], [], [], [], []]
iterations = []
user_data = []
real_slot_promenances = [0.9, 0.8, 0.8, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 0.4, 0.5, 0.5, 0.3]
categories = ["cibo", "gossip", "politic", "scienza", "sport", "tech"]
diversity_percentage = 9
diversity_percentage_for_category = diversity_percentage / 100 * sum(real_slot_promenances)
allocation_diversity_bounds = (diversity_percentage_for_category, diversity_percentage_for_category) * 3
news_pool = []
not_used_news = []
extended_news_pool = []
for _ in range(len(categories)):
    extended_news_pool.append([])

k = 0
for category in categories:
    for id in range(1, 91):
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
                                        "sport-65", "tech-61", "politic-61", "cibo-61", "cibo-62", "cibo-63", "cibo-64",
                                        "cibo-65", "gossip-62",
                                        "cibo-71", "cibo-72", "cibo-73", "cibo-74", "cibo-75",
                                        "gossip-71", "gossip-72", "gossip-73", "gossip-74", "gossip-75",
                                        "politic-71", "politic-72", "politic-73", "politic-74", "politic-75",
                                        "tech-71", "tech-72", "tech-73", "tech-74", "tech-75",
                                        "sport-71", "sport-72", "sport-73", "sport-74", "sport-75",
                                        "scienza-71", "scienza-72", "scienza-73", "scienza-74", "scienza-75"]:

            news_pool.__delitem__(-1)

for elem in news_pool:
    cat_index = categories.index(elem.news_category)
    news_pool_cat_sorted[cat_index].append(elem)




k = 0
for category in ["sport", "cibo", "tech", "politic", "gossip", "scienza"]:
    for id in range(1, 91):

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
                                        "sport-65", "tech-61", "politic-61", "cibo-61", "cibo-62", "cibo-63", "cibo-64",
                                        "cibo-65", "gossip-62",
                                        "cibo-71", "cibo-72", "cibo-73", "cibo-74", "cibo-75",
                                        "gossip-71", "gossip-72", "gossip-73", "gossip-74", "gossip-75",
                                        "politic-71", "politic-72", "politic-73", "politic-74", "politic-75",
                                        "tech-71", "tech-72", "tech-73", "tech-74", "tech-75",
                                        "sport-71", "sport-72", "sport-73", "sport-74", "sport-75",
                                        "scienza-71", "scienza-72", "scienza-73", "scienza-74", "scienza-75"]:

            news = News(news_id=id, news_name=category + "-" + str(id))
            news_index = categories.index(news.news_category)
            extended_news_pool[news_index].append(news)


def encode_html(html_file):
    lines = open(html_file, "r").readlines()
    result = ""
    for line in lines:
        result += line

    return result.encode()


def encode_news_page(html_file, user_id, news_list, page_nr):
    lines = open(html_file, "r").readlines()
    result = ""
    news_names = []
    for elem in news_list:
        news_names.append(elem.news_name)

    for line in lines:
        result += line

    result = result[0:656 + 2] + str(page_nr) + result[656 + 2::]
    result = result[0:4488 + 10] + "'" + str(user_id) + "'" + result[4488 + 10::]
    result = result[0:4633 + 12] + str(news_names) + result[4633 + 12::]
    return result.encode()


def extract_statistics():
    image_inspection_times = []
    clicked_categories = [0] * len(categories)
    clicks_per_page = []

    for filename in os.listdir("WebApp_Results"):
        file = open("WebApp_Results/" + filename, "r").read()
        file = file.split("-")

        clicks = file[0].split(",")
        clicks = list(map(float, clicks))
        clicks_per_page.append(clicks.copy())

        clicked_cat_per_page = file[1].split(";")
        for page in clicked_cat_per_page:
            tmp = page.split(",")
            for elem in tmp:
                if elem != "0":
                    cat_index = categories.index(elem)
                    clicked_categories[cat_index] += 1

        image_insp = file[4].split(";")
        for page in image_insp:
            tmp = page.split(",")
            tmp_insp_times = list(map(float, tmp))
            if sum(tmp_insp_times) > 0:
                tmp_insp_times = np.array(tmp_insp_times) / sum(tmp_insp_times)
            image_inspection_times. append(tmp_insp_times.copy())

    image_inspection_times = np.mean(image_inspection_times, axis=0)
    clicked_categories = np.array(clicked_categories) / sum(clicked_categories)
    clicks_per_page = np.mean(clicks_per_page, axis=0)

    return clicks_per_page, clicked_categories, image_inspection_times


def key_gen(length):
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "d", "c", "e", "f", "g", "h", "i", "l", "m", "n", "o", "p", "q",
              "r", "s", "t", "u", "v", "w", "x", "y", "z", "j", "k"]
    final_values = np.random.choice(values, size=length)
    key = ""
    for i in range(length):
        key += str(final_values[i])

    return key


class LogWriter:

    def __init__(self, filename, print):
        self.filename = filename
        self.print = print

    def write_log(self, messages):
        if self.print:
            file = open(self.filename, "a")
            curr_time = time.strftime("%H,%M,%S")
            curr_thread_name = threading.current_thread().name
            for message in messages:
                file.write(curr_time + ": " + str(curr_thread_name) + " - " + str(message) + "\n")
            file.close()


class RequestHandler(BaseHTTPRequestHandler):

    loggerBot = TelegramBot()
    logwriter = LogWriter("ServerLog.txt", True)

    def do_GET(self):
        self.send_response(200)
        if self.path == "/":
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
            last_visit_lock.acquire()
            last_visit.__delitem__(0)
            last_visit.append(time.time())
            last_visit_lock.release()
            self.send_header("content-type", "text/html")
            self.end_headers()
            user_key = key_gen(16)
            while user_key in user_codes:
                user_key = key_gen(16)

            user_data_lock.acquire()
            user_codes.append(user_key)
            iterations.append(0)
            user_data.append([])
            timestamps_lock.acquire()
            timestamps.append(time.time())
            timestamps_lock.release()
            user_index = user_codes.index(user_key)
            for _ in range(7):
                user_data[user_index].append([])
            for i in range(len(categories)):
                user_data[user_index][-1].append(news_pool_cat_sorted[i].copy())
                user_data[user_index][-2].append(extended_news_pool[i].copy())

            learners.append(NewsLearner(categories=categories,
                                        layout_slots=len(real_slot_promenances),
                                        real_slot_promenances=real_slot_promenances,
                                        allocation_approach="LP",
                                        allocation_diversity_bounds=allocation_diversity_bounds,
                                        ads_allocation=False))
            learners[user_index].fill_news_pool(news_list=news_pool, append=False)
            target_user_data = user_data[user_index]
            target_user_learner = learners[user_index]
            self.logwriter.write_log(["user " + str(user_key) + " joined.", "active users: " + str(user_codes),
                                      "current user index: " + str(user_index)])
            user_data_lock.release()
            allocation = target_user_learner.find_best_allocation(interest_decay=False, user=None)
            cat_index = categories.index(allocation[0].news_category)
            if len(target_user_data[-2][cat_index]) == 0:
                target_user_data[-2][cat_index] = extended_news_pool[cat_index].copy()
            allocation[0] = np.random.choice(target_user_data[-2][cat_index])
            target_user_data[-2][cat_index].remove(allocation[0])
            for k in range(1, len(allocation)):
                cat_index = categories.index(allocation[k].news_category)
                if len(target_user_data[-1][cat_index]) == 0:
                    target_user_data[-1][cat_index] = news_pool_cat_sorted[cat_index].copy()
                allocation[k] = np.random.choice(target_user_data[-1][cat_index])
                target_user_data[-1][cat_index].remove(allocation[k])
            target_user_data[0].append(allocation.copy())
            response = encode_news_page("news_page.html", user_key, allocation, 1)
            self.wfile.write(response)
            user_data_lock.acquire()
            timestamps_lock.acquire()
            current_time = time.time()
            deletion_indexes = []
            for i in range(len(timestamps)):
                if current_time - timestamps[i] > 1800:
                    deletion_indexes.append(i)
            deletion_indexes.sort(reverse=True)
            self.logwriter.write_log(["About to delete users: " + str(deletion_indexes), "Active Users: " + str(user_codes)])
            for elem in deletion_indexes:
                learners.__delitem__(elem)
                timestamps.__delitem__(elem)
                user_codes.__delitem__(elem)
                iterations.__delitem__(elem)
                user_data.__delitem__(elem)

            self.logwriter.write_log(["Active Users: " + str(user_codes)])
            timestamps_lock.release()
            user_data_lock.release()

            if np.random.binomial(1, 0.1) == 1:
                self.loggerBot.telegram_bot_sendtext("Number Of Active Users: " + str(len(user_codes)))

        elif self.path.endswith("/next"):
            try:
                self.send_header("content-type", "text/html")
                self.end_headers()
                user_data_lock.acquire()
                user_key = self.path.split("/")[1]
                user_index = user_codes.index(user_key)
                target_user_data = user_data[user_index]
                target_user_learner = learners[user_index]
                iterations[user_index] += 1
                target_user_iterations = iterations[user_index]
                timestamps[user_index] = time.time()
                self.logwriter.write_log(["User " + str(user_key) + "of index: " + str(user_index) + " complete page number " + str(target_user_iterations),
                                          "Active users: " + str(user_codes)])
                user_data_lock.release()

                if target_user_iterations < 10:
                    allocation = target_user_learner.find_best_allocation(interest_decay=False, user=None)
                    cat_index = categories.index(allocation[0].news_category)
                    if len(target_user_data[-2][cat_index]) == 0:
                        target_user_data[-2][cat_index] = extended_news_pool[cat_index].copy()
                    allocation[0] = np.random.choice(target_user_data[-2][cat_index])
                    target_user_data[-2][cat_index].remove(allocation[0])
                    for k in range(1, len(allocation)):
                        cat_index = categories.index(allocation[k].news_category)
                        if len(target_user_data[-1][cat_index]) == 0:
                            target_user_data[-1][cat_index] = news_pool_cat_sorted[cat_index].copy()
                        allocation[k] = np.random.choice(target_user_data[-1][cat_index])
                        target_user_data[-1][cat_index].remove(allocation[k])
                    target_user_data[0].append(allocation.copy())
                    response = encode_news_page("news_page.html", user_key, allocation, target_user_iterations + 1)
                    self.wfile.write(response)
                    self.logwriter.write_log(["Response sent to " + str(user_key)])
                else:
                    response = encode_html("end_page.html")
                    self.wfile.write(response)
                    self.logwriter.write_log(["About to save data for user " + str(user_key)])
                    file_saving_lock.acquire()
                    file = open("WebApp_Results/result" + str(len(os.listdir("WebApp_Results")) + 1) + ".txt", "w")
                    user_data_clicks = target_user_data[2]
                    self.loggerBot.telegram_bot_sendtext("New Sample!\nClient Address: " + str(self.client_address[0]) + "\nTotal Number Of Samples: " + str(len(os.listdir("WebApp_Results"))) + "\nClicks: " + str(user_data_clicks[0:10]))
                    self.logwriter.write_log(["User: " + str(user_key) + " clicks: " + str(user_data_clicks[0:10])])
                    file.write(str(user_data_clicks[0]))
                    for i in range(1, 10):
                        file.write("," + str(user_data_clicks[i]))
                    file.write("-")
                    j = 0
                    user_data_clicked_cats = target_user_data[1]
                    user_data_clicked_cats = user_data_clicked_cats[0:10]
                    for page_clicked_cats in user_data_clicked_cats:
                        file.write(str(page_clicked_cats[0]))
                        for i in range(1, len(page_clicked_cats)):
                            file.write("," + str(page_clicked_cats[i]))
                        j += 1
                        if j < len(user_data_clicked_cats):
                            file.write(";")
                    file.write("-")
                    j = 0
                    user_data_allocations = target_user_data[0]
                    user_data_allocations = user_data_allocations[0:10]
                    for page_allocation in user_data_allocations:
                        file.write(str(page_allocation[0].news_category))
                        for i in range(1, len(page_allocation)):
                            file.write("," + str(page_allocation[i].news_category))
                        j += 1
                        if j < len(user_data_allocations):
                            file.write(";")
                    file.write("-")
                    user_data_inspection = target_user_data[3]
                    file.write(str(user_data_inspection[0]))
                    for i in range(1, 10):
                        file.write("," + str(user_data_inspection[i]))
                    file.write("-")
                    j = 0
                    user_data_img_times = target_user_data[4]
                    user_data_img_times = user_data_img_times[0:10]
                    for page_insp_times in user_data_img_times:
                        file.write(str(page_insp_times[0]))
                        for i in range(1, len(page_insp_times)):
                            file.write("," + str(page_insp_times[i]))
                        j += 1
                        if j < len(user_data_img_times):
                            file.write(";")

                    file.close()
                    file_saving_lock.release()
                    self.logwriter.write_log(["Saved data for user " + str(user_key)])
                    user_data_lock.acquire()
                    user_index = user_codes.index(user_key)
                    self.logwriter.write_log(["About to remove data of user: " + str(user_key) + " of index: " + str(user_index)])
                    learners.__delitem__(user_index)
                    timestamps.__delitem__(user_index)
                    user_codes.__delitem__(user_index)
                    iterations.__delitem__(user_index)
                    user_data.__delitem__(user_index)
                    user_data_lock.release()
                    self.logwriter.write_log(["Removed data. Active users: " + str(user_codes) + " with residual iterations: " + str(iterations)])

            except ValueError:
                if user_data_lock.locked():
                    user_data_lock.release()
                if file_saving_lock.locked():
                    file_saving_lock.release()
                response = encode_html("session_expired_page.html")
                self.wfile.write(response)
            except IndexError:
                if user_data_lock.locked():
                    user_data_lock.release()
                if file_saving_lock.locked():
                    file_saving_lock.release()
                response = encode_html("zanero_page.html")
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
        elif self.path.endswith("/statistics_hdjdidiennsjdiwkakosoeprpriufncnaggagwiwoqlwlenxbhcufie"):
            clicks, cats, times = extract_statistics()
            self.loggerBot.telegram_bot_sendtext("Average clicks per page: " + str(clicks) + "\n" +
                                                 "Fraction of clicks per category: " + str(cats) + "\n" +
                                                 "Prominence estimation: " + str(times) + "\n" +
                                                 "Last visit: " + str((time.time() - last_visit[0]) / 60) + " minutes ago")
        else:
            if (not self.path.endswith("/favicon.ico")) and (not self.path.endswith("precomposed.png")) and \
               (not self.path.endswith("120x120.png")) and (not self.path.endswith("icon.png")):
                self.loggerBot.telegram_bot_sendtext("Bad Request From " + str(self.client_address[0]) + ": " + self.path)
                self.send_header("content-type", "text/html")
                self.end_headers()
                response = encode_html("zanero_page.html")
                self.wfile.write(response)

    def do_POST(self):
        data_string = self.rfile.read(int(self.headers['Content-Length']))
        data = simplejson.loads(data_string)

        try:
            user_data_lock.acquire()
            user_id = data["id"]
            user_clicks = data["clicked"]
            user_index = user_codes.index(user_id)
            user_alloc = user_data[user_index][0][-1]
            user_learner = learners[user_index]
            target_user_data = user_data[user_index]
            self.logwriter.write_log(["Received user data of " + str(user_id) + " with index " + str(user_index),
                                      "Currently active users: " + str(user_codes)])
            user_data_lock.release()

            num_of_clicks = 0
            clicked_elements = []
            for elem in user_clicks:
                if (elem is not True) and (elem is not False):
                    raise KeyError()

            for i in range(len(user_clicks)):
                if user_clicks[i]:
                    num_of_clicks += 1
                    clicked_elements.append(user_alloc[i].news_category)
                    user_learner.news_click(user_alloc[i], user=None, slot_nr=[i], interest_decay=False)
                else:
                    clicked_elements.append(0)

            target_user_data[1].append(clicked_elements.copy())
            target_user_data[2].append(num_of_clicks)
            if isinstance(data["inspection_time"], numbers.Number):
                target_user_data[3].append(data["inspection_time"])
            else:
                raise KeyError()
            for elem in data["image_inspection_times"]:
                if not isinstance(elem, numbers.Number):
                    raise KeyError()
            target_user_data[4].append(data["image_inspection_times"])

            self.logwriter.write_log(["Saved data of user " + str(user_id)])

            self.send_response(200)
            self.send_header("content-type", "text/html")
            self.end_headers()
            self.wfile.write(str(user_id).encode())
            self.logwriter.write_log(["Response sent to " + str(user_id),
                                      "Active users: " + str(user_codes)])

        except ValueError:
            user_data_lock.release()
            self.send_response(200)
            self.send_header("content-type", "text/html")
            self.end_headers()
            self.wfile.write("expired".encode())

        except KeyError:
            user_data_lock.release()
            self.send_response(200)
            self.send_header("content-type", "text/html")
            self.end_headers()
            self.wfile.write("bad_man".encode())
            self.loggerBot.telegram_bot_sendtext("Bad Post Request: " + str(data))


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


if __name__ == "__main__":
    PORT = 46765
    server = ThreadedHTTPServer(("", PORT), RequestHandler)
    print("multi-thread server running on port " + str(PORT))
    server.serve_forever()
