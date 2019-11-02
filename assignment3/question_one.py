import csv
import random
import math
import sys
import numpy as np
import matplotlib.pyplot as plt


class KMeansClustering:

    def __init__(self, data_file, num_iterations, k):
        self.data_file = data_file
        self.data = {}
        self.num_iteration = num_iterations
        self.k = k

    def load_data(self):
        with open(self.data_file) as d:
            for line in csv.reader(d, dialect='excel-tab'):
                key = int(line[0])
                self.data[key] = [float(line[1]), float(line[2])]

    def k_means(self):
        # These will be used for the first iteration
        representatives = self.choose_rand_unique(self.k)

        # These will be used after first iter once we calculate means
        representatives_means = []

        # Key will be the representative of that cluster, value will be list of points in that cluster
        clusters = {}

        # Add the representatives to their own clusters
        for rep in representatives:
            clusters[rep] = []
            clusters[rep].append(self.data.get(rep))

        first_time = True

        for iter in range(0, self.num_iteration):
            # Iterate over all points in the dataset
            for key in self.data.keys():

                # Don't re-add the reps to cluster on first run
                if first_time and key in clusters.keys():
                    continue

                curr_point = self.data.get(key)
                closest_rep = self.find_closest_cluster(representatives, curr_point, first_time, representatives_means)
                clusters[closest_rep].append(curr_point)

            if iter is not self.num_iteration - 1:
                # Update the representatives list
                # Basically, the representatives[i] is the index for representatives_means which will store the
                # cluster means. So if you want to get the mean value of cluster 2 you would do as follows:
                # value = representatives_means[representatives[2]]
                for j in range(0, len(representatives)):
                    rep = representatives[j]
                    representatives_means.insert(j, self.calulate_cluster_mean(clusters.get(rep)))
                    representatives[j] = j

                # Key will be the representative of that cluster, value will be list of points in that cluster
                clusters = {}

                # Add the representatives to their own clusters
                for rep in representatives:
                    clusters[rep] = []

                first_time = False
                print("Ran once")

        self.plot_clusters(clusters)

    # Add up all the x and y values and divide by number of points in cluster
    def calulate_cluster_mean(self, cluster):
        x_vals = 0
        y_vals = 0

        for point in cluster:
            x_vals += point[0]
            y_vals += point[1]

        return [(x_vals/len(cluster)), (y_vals/len(cluster))]

    # For testing
    def plot_unclustered_data(self):
        clust = dict()
        clust[1] = []

        for key in self.data.keys():
            clust[1].append(self.data.get(key))

        self.plot_clusters(clust)

    def plot_clusters(self, clusters):

        # Different color for each cluster
        colors = ["red", "black", "yellow", "green"]

        print("Found " + str(len(clusters.keys())) + " clusters!")

        i = 0

        for key in clusters.keys():
            x_python_array = []
            y_python_array = []

            for val in clusters.get(key):
                x_python_array.append(val[0])
                y_python_array.append(val[1])

            x = np.array(x_python_array)
            y = np.array(y_python_array)
            plt.scatter(x, y, c=colors[i])
            i += 1

        plt.xlabel('length')
        plt.ylabel('width')
        plt.show()

    def find_closest_cluster(self, representatives, curr_point, first_time, representatives_means):

        closest_dist = sys.maxsize
        closest_index = 0

        # If it is the first time, then the actual point values of the reps are stored in self.data dict
        # After that, once the reps are the calculated means, they will be stored in the
        # representatives_means list
        for rep in representatives:
            if first_time:
                dist = self.dist_between_points_2d(self.data.get(rep), curr_point)
            else:
                dist = self.dist_between_points_2d(representatives_means[rep], curr_point)

            if dist < closest_dist:
                closest_dist = dist
                closest_index = rep

        return closest_index

    # Calculate Euclidean distance between points
    def dist_between_points_2d(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    # Get k number of unique numbers in the range 0 to len(self.data)-1
    # This will be used to select the random clusters to start with
    def choose_rand_unique(self, k):
        return random.sample(range(0, len(self.data)-1), k)

    def apply(self):
        self.load_data()
        # self.plot_unclustered_data()
        self.k_means()


if __name__ == '__main__':
    KMeansClustering('cluster_data.txt', num_iterations=2, k=2).apply()
