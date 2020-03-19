import tkinter as tk
import time
from news_learner import *
from PIL import Image, ImageTk
import sympy as sy
from sympy.stats import ContinuousRV
import numpy as np

from tkinter.ttk import *
clicked = []
# resize 500x600 large imgs


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class AdsAppEngine:

    def __init__(self, root, title, geometry, observed_thresold, num_of_pages=10):
        self.root = root
        self.news_allocation = []
        self.enter_time = 0
        self.num_of_pages = num_of_pages
        self.visited_pages = 0
        self.exit_time = 0
        self.menubar = None
        self.observed_thresold = observed_thresold
        self.clicked = [False] * 10
        self.observed = [False] * 10
        self.clicked_news = []
        self.num_of_clicks_per_page = []
        self.allocations = []
        self.times = []
        self.init_inspection_time = 0
        self.end_inspection_time = 0
        self.learner = NewsLearner(categories=["cibo", "gossip", "politic", "scienza", "sport", "tech"],
                                   layout_slots=10, real_slot_promenances=[0.9, 0.5, 0.5, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4], allocation_approach="LP",
                                   allocation_diversity_bounds=[0.20]*6, ads_allocation=False)
        self.root.title(title)
        root.configure(bg="black")
        self.root.geometry(geometry)

        # News creation: 120 news
        news_pool = []
        k = 0
        for category in ["sport", "cibo", "tech", "politic", "gossip", "scienza"]:
            for id in range(1, 41):
                news_pool.append(News(news_id=k,
                                      news_name=category + "-" + str(id)))
                k += 1

                if category + "-" + str(id) in ["cibo-1", "cibo-6", "cibo-13", "cibo-17",
                                                "gossip-14",
                                                "politic-5", "politic-19",
                                                "scienza-6", "scienza-11",
                                                "sport-1", "sport-6", "sport-8", "sport-19", "sport-20",
                                                "tech-4", "tech-10", "tech-14", "tech-20"]:
                    news_pool.__delitem__(-1)

        self.learner.fill_news_pool(news_list=news_pool, append=False)
        self.b1 = tk.Button(command=self.click_1, height=300, width=300)
        self.b2 = tk.Button(command=self.click_2, height=150, width=150)
        self.b3 = tk.Button(command=self.click_3, height=150, width=150)
        self.b4 = tk.Button(command=self.click_4, height=300, width=300)
        self.b5 = tk.Button(command=self.click_5, height=150, width=150)
        self.b6 = tk.Button(command=self.click_6, height=150, width=150)
        self.b7 = tk.Button(command=self.click_7, height=150, width=150)
        self.b8 = tk.Button(command=self.click_8, height=150, width=150)
        self.b9 = tk.Button(command=self.click_9, height=150, width=150)
        self.b10 = tk.Button(command=self.click_10, height=150, width=150)
        self.b1.grid(row=0, column=0, columnspan=2, rowspan=2, pady=3, padx=3)
        self.b2.grid(row=0, column=2, columnspan=1, pady=3, padx=3)
        self.b3.grid(row=0, column=3, columnspan=1, rowspan=1, pady=3, padx=3)
        self.b4.grid(row=1, column=2, columnspan=2, rowspan=2, pady=3, padx=3)
        self.b5.grid(row=2, column=0, columnspan=1, pady=3, padx=3)
        self.b6.grid(row=2, column=1, columnspan=1, pady=3, padx=3)
        self.b7.grid(row=3, column=0, columnspan=1, pady=3, padx=3)
        self.b8.grid(row=3, column=1, columnspan=1, pady=3, padx=3)
        self.b9.grid(row=3, column=2, columnspan=1, pady=3, padx=3)
        self.b10.grid(row=3, column=3, columnspan=1, pady=3, padx=3)
        self.b1.bind("<Enter>", self.enter_1)
        self.b1.bind("<Leave>", self.leave_1)
        self.b2.bind("<Enter>", self.enter_2)
        self.b2.bind("<Leave>", self.leave_2)
        self.b3.bind("<Enter>", self.enter_3)
        self.b3.bind("<Leave>", self.leave_3)
        self.b4.bind("<Enter>", self.enter_4)
        self.b4.bind("<Leave>", self.leave_4)
        self.b5.bind("<Enter>", self.enter_5)
        self.b5.bind("<Leave>", self.leave_5)
        self.b6.bind("<Enter>", self.enter_6)
        self.b6.bind("<Leave>", self.leave_6)
        self.b7.bind("<Enter>", self.enter_7)
        self.b7.bind("<Leave>", self.leave_7)
        self.b8.bind("<Enter>", self.enter_8)
        self.b8.bind("<Leave>", self.leave_8)
        self.b9.bind("<Enter>", self.enter_9)
        self.b9.bind("<Leave>", self.leave_9)
        self.b10.bind("<Enter>", self.enter_10)
        self.b10.bind("<Leave>", self.leave_10)
        self.menubar = tk.Menu(self.root, font=("Helvetica", 15))
        self.root.configure(menu=self.menubar)
        self.menubar.add_command(label="NEXT", command=self.next_page)
        tk.Grid.columnconfigure(self.root, 0, weight=1)
        tk.Grid.rowconfigure(self.root, 0, weight=1)
        tk.Grid.columnconfigure(self.root, 1, weight=1)
        tk.Grid.rowconfigure(self.root, 1, weight=1)
        tk.Grid.columnconfigure(self.root, 2, weight=1)
        tk.Grid.rowconfigure(self.root, 2, weight=1)
        tk.Grid.columnconfigure(self.root, 3, weight=1)
        tk.Grid.rowconfigure(self.root, 3, weight=1)
        self.allocate_new_page()
        root.mainloop()

    def click_1(self):
        self.observed[0] = True
        if not self.clicked[0]:
            self.clicked[0] = True
            self.b1.configure(bg="blue", highlightthickness=3)
        else:
            self.clicked[0] = False
            self.b1.configure(bg="white", highlightthickness=1)

    def click_2(self):
        self.observed[1] = True
        if not self.clicked[1]:
            self.clicked[1] = True
            self.b2.configure(bg="blue", highlightthickness=3)
        else:
            self.clicked[1] = False
            self.b2.configure(bg="white", highlightthickness=1)

    def click_3(self):
        self.observed[2] = True
        if not self.clicked[2]:
            self.clicked[2] = True
            self.b3.configure(bg="blue", highlightthickness=3)
        else:
            self.clicked[2] = False
            self.b3.configure(bg="white", highlightthickness=1)

    def click_4(self):
        self.observed[3] = True
        if not self.clicked[3]:
            self.clicked[3] = True
            self.b4.configure(bg="blue", highlightthickness=3)
        else:
            self.clicked[3] = False
            self.b4.configure(bg="white", highlightthickness=1)

    def click_5(self):
        self.observed[4] = True
        if not self.clicked[4]:
            self.clicked[4] = True
            self.b5.configure(bg="blue", highlightthickness=3)
        else:
            self.clicked[4] = False
            self.b5.configure(bg="white", highlightthickness=1)

    def click_6(self):
        self.observed[5] = True
        if not self.clicked[5]:
            self.clicked[5] = True
            self.b6.configure(bg="blue", highlightthickness=3)
        else:
            self.clicked[5] = False
            self.b6.configure(bg="white", highlightthickness=1)

    def click_7(self):
        self.observed[6] = True
        if not self.clicked[6]:
            self.clicked[6] = True
            self.b7.configure(bg="blue", highlightthickness=3)
        else:
            self.clicked[6] = False
            self.b7.configure(bg="white", highlightthickness=1)

    def click_8(self):
        self.observed[7] = True
        if not self.clicked[7]:
            self.clicked[7] = True
            self.b8.configure(bg="blue", highlightthickness=3)
        else:
            self.clicked[7] = False
            self.b8.configure(bg="white", highlightthickness=1)

    def click_9(self):
        self.observed[8] = True
        if not self.clicked[8]:
            self.clicked[8] = True
            self.b9.configure(bg="blue", highlightthickness=3)
        else:
            self.clicked[8] = False
            self.b9.configure(bg="white", highlightthickness=1)

    def click_10(self):
        self.observed[9] = True
        if not self.clicked[9]:
            self.clicked[9] = True
            self.b10.configure(bg="blue", highlightthickness=3)
        else:
            self.clicked[9] = False
            self.b10.configure(bg="white", highlightthickness=1)

    def enter_1(self, a):
        if not self.clicked[0]:
            self.b1.configure(bg="red", highlightthickness=3)

        if not self.observed[0]:
            self.enter_time = time.time()

    def leave_1(self, a):
        if not self.clicked[0]:
            self.b1.configure(bg="white", highlightthickness=1)

        if not self.observed[0]:
            self.exit_time = time.time()
            if self.exit_time - self.enter_time >= self.observed_thresold:
                self.observed[0] = True

    def enter_2(self, a):
        if not self.clicked[1]:
            self.b2.configure(bg="red", highlightthickness=3)

        if not self.observed[1]:
            self.enter_time = time.time()

    def leave_2(self, a):
        if not self.clicked[1]:
            self.b2.configure(bg="white", highlightthickness=1)

        if not self.observed[1]:
            self.exit_time = time.time()
            if self.exit_time - self.enter_time >= self.observed_thresold:
                self.observed[1] = True

    def enter_3(self, a):
        if not self.clicked[2]:
            self.b3.configure(bg="red", highlightthickness=3)

        if not self.observed[2]:
            self.enter_time = time.time()

    def leave_3(self, a):
        if not self.clicked[2]:
            self.b3.configure(bg="white", highlightthickness=1)

        if not self.observed[2]:
            self.exit_time = time.time()
            if self.exit_time - self.enter_time >= self.observed_thresold:
                self.observed[2] = True

    def enter_4(self, a):
        if not self.clicked[3]:
            self.b4.configure(bg="red", highlightthickness=3)

        if not self.observed[3]:
            self.enter_time = time.time()

    def leave_4(self, a):
        if not self.clicked[3]:
            self.b4.configure(bg="white", highlightthickness=1)

        if not self.observed[3]:
            self.exit_time = time.time()
            if self.exit_time - self.enter_time >= self.observed_thresold:
                self.observed[3] = True

    def enter_5(self, a):
        if not self.clicked[4]:
            self.b5.configure(bg="red", highlightthickness=3)

        if not self.observed[4]:
            self.enter_time = time.time()

    def leave_5(self, a):
        if not self.clicked[4]:
            self.b5.configure(bg="white", highlightthickness=1)

        if not self.observed[4]:
            self.exit_time = time.time()
            if self.exit_time - self.enter_time >= self.observed_thresold:
                self.observed[4] = True

    def enter_6(self, a):
        if not self.clicked[5]:
            self.b6.configure(bg="red", highlightthickness=3)

        if not self.observed[5]:
            self.enter_time = time.time()

    def leave_6(self, a):
        if not self.clicked[5]:
            self.b6.configure(bg="white", highlightthickness=1)

        if not self.observed[5]:
            self.exit_time = time.time()
            if self.exit_time - self.enter_time >= self.observed_thresold:
                self.observed[5] = True

    def enter_7(self, a):
        if not self.clicked[6]:
            self.b7.configure(bg="red", highlightthickness=3)

        if not self.observed[6]:
            self.enter_time = time.time()

    def leave_7(self, a):
        if not self.clicked[6]:
            self.b7.configure(bg="white", highlightthickness=1)

        if not self.observed[6]:
            self.exit_time = time.time()
            if self.exit_time - self.enter_time >= self.observed_thresold:
                self.observed[6] = True

    def enter_8(self, a):
        if not self.clicked[7]:
            self.b8.configure(bg="red", highlightthickness=3)

        if not self.observed[7]:
            self.enter_time = time.time()

    def leave_8(self, a):
        if not self.clicked[7]:
            self.b8.configure(bg="white", highlightthickness=1)

        if not self.observed[7]:
            self.exit_time = time.time()
            if self.exit_time - self.enter_time >= self.observed_thresold:
                self.observed[7] = True

    def enter_9(self, a):
        if not self.clicked[8]:
            self.b9.configure(bg="red", highlightthickness=3)

        if not self.observed[8]:
            self.enter_time = time.time()

    def leave_9(self, a):
        if not self.clicked[8]:
            self.b9.configure(bg="white", highlightthickness=1)

        if not self.observed[8]:
            self.exit_time = time.time()
            if self.exit_time - self.enter_time >= self.observed_thresold:
                self.observed[8] = True

    def enter_10(self, a):
        if not self.clicked[9]:
            self.b10.configure(bg="red", highlightthickness=3)

        if not self.observed[9]:
            self.enter_time = time.time()

    def leave_10(self, a):
        if not self.clicked[9]:
            self.b10.configure(bg="white", highlightthickness=1)

        if not self.observed[9]:
            self.exit_time = time.time()
            if self.exit_time - self.enter_time >= self.observed_thresold:
                self.observed[9] = True

    def allocate_new_page(self):

        self.news_allocation = self.learner.find_best_allocation(interest_decay=False, user=None)
        allocation = []
        for elem in self.news_allocation:
            allocation.append(elem.news_category)
        self.allocations.append(allocation.copy())
        root.photo1 = ImageTk.PhotoImage(Image.open(resource_path(self.news_allocation[0].image_path)).resize((300, 300), Image.ANTIALIAS))
        self.b1.configure(image=root.photo1)
        root.photo2 = ImageTk.PhotoImage(Image.open(resource_path(self.news_allocation[1].image_path)).resize((150, 150), Image.ANTIALIAS))
        self.b2.configure(image=root.photo2)
        root.photo3 = ImageTk.PhotoImage(Image.open(resource_path(self.news_allocation[2].image_path)).resize((150, 150), Image.ANTIALIAS))
        self.b3.configure(image=root.photo3)
        root.photo4 = ImageTk.PhotoImage(Image.open(resource_path(self.news_allocation[3].image_path)).resize((300, 300), Image.ANTIALIAS))
        self.b4.configure(image=root.photo4)
        root.photo5 = ImageTk.PhotoImage(Image.open(resource_path(self.news_allocation[4].image_path)).resize((150, 150), Image.ANTIALIAS))
        self.b5.configure(image=root.photo5)
        root.photo6 = ImageTk.PhotoImage(Image.open(resource_path(self.news_allocation[5].image_path)).resize((150, 150), Image.ANTIALIAS))
        self.b6.configure(image=root.photo6)
        root.photo7 = ImageTk.PhotoImage(Image.open(resource_path(self.news_allocation[6].image_path)).resize((150, 150), Image.ANTIALIAS))
        self.b7.configure(image=root.photo7)
        root.photo8 = ImageTk.PhotoImage(Image.open(resource_path(self.news_allocation[7].image_path)).resize((150, 150), Image.ANTIALIAS))
        self.b8.configure(image=root.photo8)
        root.photo9 = ImageTk.PhotoImage(Image.open(resource_path(self.news_allocation[8].image_path)).resize((150, 150), Image.ANTIALIAS))
        self.b9.configure(image=root.photo9)
        root.photo10 = ImageTk.PhotoImage(Image.open(resource_path(self.news_allocation[9].image_path)).resize((150, 150), Image.ANTIALIAS))
        self.b10.configure(image=root.photo10)
        self.init_inspection_time = time.time()

    def next_page(self):

        self.end_inspection_time = time.time()
        self.times.append(self.end_inspection_time - self.init_inspection_time)
        observed_slots = []
        not_observed_slots = []
        clicked_news = []
        not_clicked_news = []
        observed_news = []

        self.visited_pages += 1

        if self.visited_pages < self.num_of_pages:

            # collecting information on slot/news observations in the last page
            for i in range(len(self.observed)):
                if self.observed[i]:
                    observed_slots.append(i)
                    observed_news.append(self.news_allocation[i])
                else:
                    not_observed_slots.append(i)

            # collecting information on news click in the last page
            for i in range(len(self.observed)):
                if self.observed[i] and self.clicked[i]:
                    clicked_news.append(self.news_allocation[i])
                elif self.observed[i] and not self.clicked[i]:
                    not_clicked_news.append(self.news_allocation[i])

            # update the news ts learner
            tmp_clicked_news = []
            tmp_num_of_clicks = 0
            for k in range(len(self.clicked)):
                if self.clicked[k]:
                    tmp_num_of_clicks += 1
                    tmp_clicked_news.append(self.news_allocation[k].news_category)
                    self.learner.news_click(self.news_allocation[k], user=None, slot_nr=[k], interest_decay=False)
                else:
                    tmp_clicked_news.append(0)
            self.clicked_news.append(tmp_clicked_news.copy())
            self.num_of_clicks_per_page.append(tmp_num_of_clicks)

            # reset class parameters
            self.clicked = [False] * 10
            self.observed = [False] * 10
            self.b1.configure(bg="white", highlightthickness=1)
            self.b2.configure(bg="white", highlightthickness=1)
            self.b3.configure(bg="white", highlightthickness=1)
            self.b4.configure(bg="white", highlightthickness=1)
            self.b5.configure(bg="white", highlightthickness=1)
            self.b6.configure(bg="white", highlightthickness=1)
            self.b7.configure(bg="white", highlightthickness=1)
            self.b8.configure(bg="white", highlightthickness=1)
            self.b9.configure(bg="white", highlightthickness=1)
            self.b10.configure(bg="white", highlightthickness=1)

            self.allocate_new_page()

        else:
            file = open("send_me_back_to_luca.txt", "a")
            file.write(str(self.num_of_clicks_per_page[0]))
            for j in range(1, len(self.num_of_clicks_per_page)):
                file.write("," + str(self.num_of_clicks_per_page[j]))
            file.write("-")
            k = 0
            for click in self.clicked_news:
                k += 1
                file.write(str(click[0]))
                for j in range(1, len(click)):
                    file.write("," + str(click[j]))
                if k != len(self.allocations) - 1:
                    file.write(";")
            file.write("-")
            k = 0
            for alloc in self.allocations:
                k += 1
                file.write(str(alloc[0]))
                for j in range(1, len(alloc)):
                    file.write("," + str(alloc[j]))
                if k != len(self.allocations) - 1:
                    file.write(";")
            file.write("-")
            file.write(str(self.times[0]))
            for j in range(1, len(self.times)):
                file.write("," + str(self.times[j]))
            file.write("\n")
            file.close()

            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    engine = AdsAppEngine(root, "AdsApp-News", "1050x650", 0.7, num_of_pages=10)
