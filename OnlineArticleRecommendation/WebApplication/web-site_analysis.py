import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import *
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
import scipy.spatial.distance as ssd
from scipy.stats import norm, chi2, t
from tqdm import tqdm
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score, silhouette_samples
categories = ["cibo", "gossip", "politic", "scienza", "sport", "tech"]
categories_eng = ["Food", "Gossip", "Politic", "Science", "Sport", "Tech"]
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 200)


def dtw(ts1, ts2, derivative=False):
    """
    Dynamic time warping algorithm between two time series
    :param ts1: 1D list respresenting a time series
    :param ts2: 1D list respresenting a time series
    :param derivative: if derivative or classic dynamic time warping
    :return: a distance measure according to dtw
    """
    s = ts1
    t = ts2

    if derivative:
        tmp_ts1 = []
        tmp_ts2 = []
        for i in range(len(ts1) - 1):
            tmp_ts1.append(ts1[i + 1] - ts1[i])
            tmp_ts2.append(ts2[i + 1] - ts2[i])
        s = tmp_ts1
        t = tmp_ts2

    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s[i - 1] - t[j - 1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[-1][-1]


def extract_statistics(folder, ab_test=False):
    clicks_per_page = []
    page_inspection_times = []
    clicked_categories_per_page_per_user = []
    whole_experience_allocated_categories = []
    learning_agent = []
    user_guess_on_learning_agent = []
    result = []

    for filename in os.listdir(folder):
        print(filename)
        clicked_categories = [0] * len(categories)
        allocated_categories = [0] * len(categories)
        file = open(folder + "/" + filename, "r").read()
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

        clicked_categories_per_page_per_user.append(clicked_categories.copy())

        whole_exp_allocated_categories = file[2].split(";")
        for page in whole_exp_allocated_categories:
            tmp = page.split(",")
            for elem in tmp:
                cat_index = categories.index(elem)
                allocated_categories[cat_index] += 1

        whole_experience_allocated_categories.append(allocated_categories.copy())

        page_insp = file[3].split(",")
        page_insp = list(map(float, page_insp))
        page_insp = list(np.array(page_insp) / 1000)
        page_inspection_times.append(page_insp.copy())

        if ab_test:
            learning_agent.append(file[-1].split(";")[-1])
            guess = file[-1].split(";")[-2]
            if guess == "True":
                user_guess_on_learning_agent.append(1)
            else:
                user_guess_on_learning_agent.append(0)

    for i in range(len(clicks_per_page)):
        if not ab_test:
            result.append(clicks_per_page[i] + clicked_categories_per_page_per_user[i] +
                          whole_experience_allocated_categories[i] + page_inspection_times[i])
        else:
            result.append(clicks_per_page[i] + clicked_categories_per_page_per_user[i] +
                          whole_experience_allocated_categories[i] + page_inspection_times[i] + [learning_agent[i]] +
                          [user_guess_on_learning_agent[i]])

    return result


def prepare_data_frame(statistics, ab_test=False):
    if ab_test:
        frame = pd.DataFrame(data=statistics, columns=["Page-1-Clicks", "Page-2-Clicks", "Page-3-Clicks", "Page-4-Clicks",
                                                       "Page-5-Clicks", "Page-6-Clicks", "Page-7-Clicks", "Page-8-Clicks",
                                                       "Page-9-Clicks", "Page-10-Clicks", "Food-Clicks", "Gossip-Clicks",
                                                       "Politic-Clicks", "Science-Clicks", "Sport-Clicks", "Tech-Clicks",
                                                       "Food-Allocations", "Gossip-Allocations", "Politic-Allocations",
                                                       "Science-Allocations", "Sport-Allocations", "Tech-Allocations",
                                                       "Page-1-Time", "Page-2-Time", "Page-3-Time", "Page-4-Time",
                                                       "Page-5-Time", "Page-6-Time", "Page-7-Time", "Page-8-Time",
                                                       "Page-9-Time", "Page-10-Time"])
    else:
        frame = pd.DataFrame(data=statistics,
                             columns=["Page-1-Clicks", "Page-2-Clicks", "Page-3-Clicks", "Page-4-Clicks",
                                      "Page-5-Clicks", "Page-6-Clicks", "Page-7-Clicks", "Page-8-Clicks",
                                      "Page-9-Clicks", "Page-10-Clicks", "Food-Clicks", "Gossip-Clicks",
                                      "Politic-Clicks", "Science-Clicks", "Sport-Clicks", "Tech-Clicks",
                                      "Food-Allocations", "Gossip-Allocations", "Politic-Allocations",
                                      "Science-Allocations", "Sport-Allocations", "Tech-Allocations",
                                      "Page-1-Time", "Page-2-Time", "Page-3-Time", "Page-4-Time",
                                      "Page-5-Time", "Page-6-Time", "Page-7-Time", "Page-8-Time",
                                      "Page-9-Time", "Page-10-Time", "Learning_Agent", "User_Guess_Learning_Agent"])

    return frame


def elbow_knee_analysis(frame, merges, k_values, interested_columns, derivative=False):
    """
    Plot the elbow knee analysis given a dataframe and the set of values of clustering
    :param frame: pandas dataframe
    :param merges: the output of scipy clustering method: linkage
    :param k_values: the values of cluster configurations to plot
    :param interested_columns: the value according to which we want to compute the distance among clusters
    :param derivative: if true derivative dtw is used
    :return: nothing
    """
    wss_values = []
    bss_values = []

    for k in tqdm(k_values):
        centriod_distances = []
        clustering = fcluster(merges, k, "maxclust")
        frame["cluster"] = clustering
        centroids = [frame[frame["cluster"] == c][interested_columns].mean() for c in range(1, k + 1)]
        dataset_centroid = frame[interested_columns].mean()

        for i, row in frame.iterrows():
            row_distances = []
            for centroid in centroids:
                row_distances.append(dtw(row[interested_columns], centroid, derivative=derivative))
            centriod_distances.append(min(row_distances))

        wss_values.append(sum([distance**2 for distance in centriod_distances]))

        bss_distances = []
        for centroid_index in range(len(centroids)):
            bss_distances.append(len(frame[frame["cluster"] == (centroid_index + 1)]) *
                                 dtw(centroids[centroid_index], dataset_centroid, derivative=True)**2)

        bss_values.append(sum(bss_distances))

    return wss_values, bss_values


def read_dm_from_file(filename):
    file = open(filename, "r")
    lines = file.readlines()
    distance_matrix = []

    for line in lines:
        tmp_line = line.split(",")
        tmp_line = list(map(float, tmp_line))
        distance_matrix.append(tmp_line.copy())

    return np.array(distance_matrix)


def build_dm_from_frame_to_file(target_frame, interested_columns, filename, derivative=False):
    """
    Create a distance matrix according to dynamic time warping measure and save it into a file
    :param target_frame: pandas dataframe
    :param interested_columns: the columns we want to use to generate clusters
    :param filename: the file in which to save the matrix
    :param derivative: if true it uses derivative dtw
    :return: Nothing
    """
    distance_matrix = []
    frame = target_frame
    frame.reset_index(inplace=True)

    for _ in range(len(frame)):
        distance_matrix.append([0] * len(frame))

    for i in tqdm(range(len(frame) - 1)):
        for j in range(i + 1, len(frame)):
            distance = dtw(frame.loc[i, interested_columns],
                           frame.loc[j, interested_columns],
                           derivative=derivative)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    file = open(filename, "w")
    for row in distance_matrix:
        file.write(str(row[0]))
        for i in range(1, len(row)):
            file.write("," + str(row[i]))
        file.write("\n")
    file.close()


def build_dm_from_data_to_file(data, filename, derivative=False):
    """
    Creates a distance matrix according to Dynamic Time Warping algorithm and save it into a file
    :param data: pandas dataframe
    :param filename:
    :param derivative: if dynamic time warping should be derivative or normal
    :return: None
    """
    distance_matrix = []

    for _ in range(len(data)):
        distance_matrix.append([0] * len(data))

    for i in tqdm(range(len(data) - 1)):
        for j in range(i + 1, len(data)):
            distance = dtw(data[i],
                           data[j],
                           derivative=derivative)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    file = open(filename, "w")
    for row in distance_matrix:
        file.write(str(row[0]))
        for i in range(1, len(row)):
            file.write("," + str(row[i]))
        file.write("\n")
    file.close()


def normalize_categories_clicks(frame):
    for cat in categories_eng:
        frame[cat + "-Clicks-Norm"] = frame[cat + "-Clicks"] / (frame["Food-Clicks"] + frame["Gossip-Clicks"] +
                                                                frame["Sport-Clicks"] + frame["Science-Clicks"] +
                                                                frame["Tech-Clicks"] + frame["Politic-Clicks"])
    return frame


def add_categories_per_page_clicks_to_frame(frame, category, drop_threshold):
    """
    It produces 10 new columns to attach to an existing data frame. Each column i represent
    the number of clicked items with category=category on page i.
    :param frame: Frame to which attach the new columns
    :param category: Category we are interested in
    :param drop_threshold: Drops lines with total number of clicks relative to category less than the threshold
    :return: the frame with 10 new columns called category-clicks-page-i for i in [1, 10]
    """

    set_of_columns = []
    for filename in os.listdir("WebApp_Results"):
        clicks_per_page = [0] * 10

        file = open("WebApp_Results/" + filename, "r").read()
        file = file.split("-")[1]
        pages = file.split(";")
        for i in range(len(pages)):
            clicked_categories = pages[i].split(",")
            clicks_per_page[i] = clicked_categories.count(category)

        set_of_columns.append(clicks_per_page.copy())

    tmp_frame = pd.DataFrame(set_of_columns, columns=[drop_threshold[0] + "-Clicks-Page-" + str(k) for k in range(1, 11)])
    frame = pd.concat([frame, tmp_frame], axis=1)
    frame.drop(frame[(frame["Page-1-Time"] < 2) | (frame["Page-2-Time"] < 2) |
                     (frame["Page-3-Time"] < 2) | (frame["Page-4-Time"] < 2) |
                     (frame["Page-5-Time"] < 2) | (frame["Page-6-Time"] < 2) |
                     (frame["Page-7-Time"] < 2) | (frame["Page-8-Time"] < 2) |
                     (frame["Page-9-Time"] < 2) | (frame["Page-10-Time"] < 2)].index, axis=0, inplace=True)
    frame.drop(frame[(frame["Page-1-Time"] > 270) | (frame["Page-2-Time"] > 270) |
                     (frame["Page-3-Time"] > 270) | (frame["Page-4-Time"] > 270) |
                     (frame["Page-5-Time"] > 270) | (frame["Page-6-Time"] > 270) |
                     (frame["Page-7-Time"] > 270) | (frame["Page-8-Time"] > 270) |
                     (frame["Page-9-Time"] > 270) | (frame["Page-10-Time"] > 270)].index, axis=0, inplace=True)
    frame.drop(frame[frame[drop_threshold[0] + "-Clicks"] < drop_threshold[1]].index, inplace=True, axis=0)

    return frame


def snake_plot(frame, interested_columns, title, legend, xlabel, ylabel):
    """
    Plots a snake plot of each of the interested columns
    :param frame: pandas dataframe
    :param interested_columns: to be included in the plot
    :param title: the title of the plot
    :param legend: legend of the plot
    :param xlabel: label on x
    :param ylabel: label on y
    :return: Nothing
    """
    melt_frame = pd.melt(frame,
                         id_vars=['cluster'],
                         value_vars=interested_columns,
                         var_name='Attribute',
                         value_name='Value')

    sns.lineplot(x="Attribute", y="Value", hue='cluster', data=melt_frame)
    plt.xticks([i for i in range(0, 10)], [str(i) for i in range(1, 11)])
    plt.legend(legend)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_silhouette_scores(frame, range_n_clusters, diversity_matrix, merges):
    """
    Plots the silhouette scores of the frame for each of the specified clusters.
    :param frame: pandas dataframe
    :param range_n_clusters: number of cluster of which we want to plot the scores
    :param diversity_matrix: a precomputed distance matrix
    :param merges: returned by the scipy Linkage method
    :return: Nothing
    """
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.5, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(frame) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        labels = fcluster(merges, n_clusters, "maxclust")
        silhouette_avg = silhouette_score(diversity_matrix, labels, metric="precomputed")

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(diversity_matrix, labels, metric="precomputed")

        y_lower = 10
        for i in range(1, n_clusters + 1):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_xlabel("Silhouette Coefficient")
        ax1.set_ylabel("Cluster Number")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.title("Silhouette Analysis For DDTW H. Clustering")

    plt.show()


def create_image_insp_times_frame():
    image_inspection_times = []
    frame = pd.read_csv("WebSite_Results.csv")
    for filename in os.listdir("WebApp_Results"):
        file = open("WebApp_Results/" + filename, "r").read()
        file = file.split("-")

        page_clicks = file[1].split(";")
        image_insp = file[4].split(";")
        for i in range(len(image_insp)):
            tmp = image_insp[i].split(",")
            clicked_slots = page_clicks[i].split(",")
            tmp_insp_times = list(map(float, tmp))
            if sum(tmp_insp_times) > 0:
                # TRY TO INFER WHETHER THE SOURCE IS MOBILE
                reliable_insp_times = False
                for j in range(len(tmp_insp_times)):
                    if (tmp_insp_times[j] > 0) and (clicked_slots[j] == "0"):
                        reliable_insp_times = True
                        break
                if reliable_insp_times:
                    tmp_insp_times = np.array(tmp_insp_times) / sum(tmp_insp_times)
                    image_inspection_times.append(tmp_insp_times.copy())

    times_frame = pd.DataFrame(image_inspection_times, columns=["Slot-" + str(i) + "-InspTime" for i in range(1, 14)])

    return times_frame


def bootstrapping_variance_estimation(data, iterations=100):
    """
    Compute the boostrapped variance of data
    :param data: 1D list of float
    :param iterations: Iterations of the boostrapping method
    :return: the estimated variance's mean
    """
    bootstrapped_variance = []
    for i in tqdm(range(1)):
        data_at_index_i = [elem for elem in data]

        variance_estimation = []
        for _ in range(iterations):
            bootstrapped_data = []
            for _ in range(len(data_at_index_i)):
                bootstrapped_data.append(np.random.choice(data_at_index_i))
            variance_estimation.append(np.var(bootstrapped_data))

        bootstrapped_variance.append(np.mean(variance_estimation, axis=0))

    return bootstrapped_variance


def compute_p_value(frame, interested_columns=["Page-" + str(j) + "-Clicks" for j in range(1, 11)]):
    """
    For each of the columns specified in interested columns, perform a Z-test (bootstrapping the variance) and return
    the p-value for each test
    :param frame: A pandas dataframe. The frame must contain a column named learning agent, which is boolean, to discriminate
    for the test
    :param interested_columns: the columns on which we want to perform the Z-test
    :return: 1D list of floats represent the p-values of each column in interested columns
    """
    frame_learning = frame[frame["Learning_Agent"]]
    frame_not_learning = frame[frame["Learning_Agent"] == False]
    p_values = []
    var_learning = bootstrapping_variance_estimation(
        frame_learning.loc[:, interested_columns].values,
        iterations=1000)
    var_random = bootstrapping_variance_estimation(
        frame_not_learning.loc[:, interested_columns].values,
        iterations=1000)

    for i in range(len(interested_columns)):
        mean_learning = frame_learning[interested_columns[i]].mean()
        mean_random = frame_not_learning[interested_columns[i]].mean()
        Z = (mean_learning - mean_random) / np.sqrt(
            var_learning[i] / len(frame_learning) + var_random[i] / len(frame_not_learning))

        p_values.append(1 - norm.cdf(Z))

    return p_values


def chi_test_goodness_of_fit(dimension1, dimension2):
    """
    Perform a chi-squared goodness of fit test over two sample of dimension1 and dimension2 respectively.
    :param dimension1: Integer representing the dimension of the first sample
    :param dimension2: Integer representing the dimension of the second sample
    :return: The p-value corresponding to the test
    """
    total_len = (dimension1 + dimension2) / 2

    X = (dimension1 - total_len) ** 2 / total_len + (dimension2 - total_len) ** 2 / total_len
    return 1 - chi2.cdf(X, df=1)


if __name__ == "__main__":
    frame = pd.read_csv("AB_Test_Frame_wo_Outliers.csv")
    frame = frame[frame["Learning_Agent"] == True]
    print(frame.describe())
    sns.regplot(x="Science-Clicks", y="Science-Allocations", data=frame, color="green")
    plt.title("Linear Trend for Science Clicks-Allocations")
    plt.show()
