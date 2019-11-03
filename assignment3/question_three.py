from sklearn.datasets import load_digits
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import confusion_matrix, fowlkes_mallows_score
from sklearn.decomposition import PCA
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


class Clustering:

    def __init__(self, k):
        self.k = k
        self.digits = load_digits()

        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
            self.digits.data,
            self.digits.target,
            random_state=0)

    def cluster_k_means(self):
        # Use kmeans++ which will help converge faster, num clusters is 10, perform 10 iterations of algo
        k_means = KMeans(init='random', n_clusters=self.k, n_init=10)
        target_prediction = k_means.fit(self.data_train).predict(self.data_test)
        self.plot_confusion_matrix(self.target_test, target_prediction, classes=self.digits.target_names,
                                   title="K-Means")
        print("K-Means fowlkes mallows score: " + str(fowlkes_mallows_score(self.target_test, target_prediction)))

    def cluster_agglomerative(self):
        agglomerative = AgglomerativeClustering(linkage='ward', n_clusters=self.k)
        target_prediction = agglomerative.fit(self.data_train).fit_predict(self.data_test)
        self.plot_confusion_matrix(self.target_test, target_prediction, classes=self.digits.target_names,
                                   title="Agglomerative")
        print("Agglomerative fowlkes mallows score: " + str(fowlkes_mallows_score(self.target_test, target_prediction)))

    def cluster_affinity_propagation(self):
        affinity_propagation = AffinityPropagation(damping=0.99)
        target_prediction = affinity_propagation.fit(self.data_train).predict(self.data_test)

        print("Target Prediction: \n" + str(target_prediction))

        #todo uncomment
        # self.plot_confusion_matrix(self.target_test, target_prediction, classes=self.digits.target_names,
        #                            title="Affinity Propagation")
        # print("Affinity Propagation fowlkes mallows score: " + str(fowlkes_mallows_score(self.data_train, target_prediction)))

    # Function taken from Scikit:
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    def plot_confusion_matrix(self, y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes,
               yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    def cluster_all(self):
        self.cluster_k_means()
        self.cluster_agglomerative()
        self.cluster_affinity_propagation()
        plt.show()


if __name__ == '__main__':
    cluster = Clustering(k=10)
    cluster.cluster_all()
