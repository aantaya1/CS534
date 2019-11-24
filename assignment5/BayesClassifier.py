import csv
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
np.set_printoptions(threshold=sys.maxsize)


class BayesClassifier:
    def __init__(self, k=1):

        # For laplace smoothing
        self.k = k

        # Store the probability or spam or ham given a word
        self.spam_probabilities = dict()
        self.ham_probabilities = dict()

        # Total number of spam and ham messages
        self.num_spam = 0
        self.num_ham = 0

        # For mapping between index and actual word
        self.mapper = dict()

    def feature_label_from_csv(self, filename):
        first_line = True

        labels = []
        features = []

        with open(filename) as d:

            loop_cnt = 0

            for line in csv.reader(d):

                # Add all the words to a dict where key=index, value=word
                if first_line:
                    for idx in range(0, len(line) - 2):
                        self.mapper[idx] = str(line[idx+1])
                    first_line = False
                    loop_cnt += 1
                    continue

                # Add the labels
                labels.insert(loop_cnt, int(line[len(line)-1]))

                feature = []

                # Iterate over every word on line
                for idx in range(1, len(line)-2):
                    feature.append(int(line[idx]))

                features.append(feature)

                loop_cnt += 1

        X = np.array(features)
        y = np.array(labels)

        return X, y

    def fit(self, X, y):
        spam_occurrences = dict()
        ham_occurrences = dict()

        num_spam = 0
        num_ham = 0

        spam_word = 0
        ham_word = 0

        # Count the number of spam messages
        for val in y:
            if int(val) is 0:
                num_spam += 1
            else:
                num_ham += 1

        for row_index in range(0, len(X)):
            row = X[row_index]
            for idx in range(0, len(row)):

                # If the word is present (1) and the message is spam (0)
                if int(row[idx]) is 1 and int(y[row_index]) is 0:
                    spam_word += 1
                    word = self.mapper.get(idx)

                    # Add or increment the spam count for this word (i.e. how many times it has been
                    # found in a message that is spam)
                    if word not in spam_occurrences.keys():
                        spam_occurrences[word] = 1
                    else:
                        spam_occurrences[word] = spam_occurrences.get(word) + 1

                # If the word is present (1) and the message is ham (1)
                if int(row[idx]) is 1 and int(y[row_index]) is 1:
                    ham_word += 1
                    word = self.mapper.get(idx)

                    # Add or increment the ham count for this word (i.e. how many times it has been
                    # found in a message that is ham)
                    if word not in ham_occurrences.keys():
                        ham_occurrences[word] = 1
                    else:
                        ham_occurrences[word] = ham_occurrences.get(word) + 1

        # Calculate the probability of each word being in a spam message
        for key in self.mapper.keys():
            word = self.mapper.get(key)

            if word not in spam_occurrences.keys():
                spam_occurrences[word] = 0

            if word not in ham_occurrences.keys():
                ham_occurrences[word] = 0

            # Use laplas smoothing, where num classes is 2
            # Get probability of spam given that word
            word_spam_prob = (spam_occurrences[word] + self.k) / (num_spam + (self.k * 2))
            self.spam_probabilities[word] = word_spam_prob

            # Get probability of ham given that word
            word_ham_prob = (ham_occurrences[word] + self.k) / (num_ham + (self.k * 2))
            self.ham_probabilities[word] = word_ham_prob

        self.num_ham = num_ham
        self.num_spam = num_spam

    def predict(self, X):

        predicted_labels = []

        total_messages = self.num_spam + self.num_ham
        total_probability_spam = self.num_spam / total_messages
        total_probability_ham = self.num_ham / total_messages

        for row in X:
            probability_spam = 1
            probability_ham = 1

            # Multiple the probabilities of being spam or ham for each individual word (numerator)
            # Also calculate the probability of seeing each word (denominator)
            for idx in range(0, len(row)):

                if int(row[idx]) is 0:
                    continue

                word = self.mapper.get(idx)

                if word in self.spam_probabilities:
                    probability_spam *= self.spam_probabilities[word]

                if word in self.ham_probabilities:
                    probability_ham *= self.ham_probabilities[word]

            # Multiply by the probability of having spam or ham
            probability_ham *= total_probability_ham
            probability_spam *= total_probability_spam

            # Whichever probability is more likely will be the final classification
            if probability_ham >= probability_spam:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)

        return np.array(predicted_labels)


def calculate_f_score(predicted_labels, actual_labels):
    num_tp = 0
    num_tn = 0
    num_fp = 0
    num_fn = 0

    if (len(predicted_labels) is not len(actual_labels)):
        print("calculate_f_score(): Size of labels are not the same size!")
        return -1

    num_labels = len(predicted_labels)

    # Go through every label and get the number of true positive, false positive, etc.
    for idx in range(0, num_labels):
        pred_val = int(predicted_labels[idx])
        actual_val = int(actual_labels[idx])

        if pred_val is 1:  # Positive values
            if pred_val == actual_val:
                num_tp += 1
            elif pred_val != actual_val:
                num_fp += 1
        elif pred_val is 0:  # Negative values
            if pred_val == actual_val:
                num_tn += 1
            elif pred_val != actual_val:
                num_fn += 1

    # Calculate pre and rec
    pre = num_tp / (num_tp + num_fp)
    rec = num_tp / (num_tp + num_fn)

    # Calculate the actual f-measure now
    f_measure = (2 * pre * rec) / (pre + rec)

    return f_measure


if __name__ == '__main__':

    # How we will split the data set for training/testing
    TRAINING_SPLIT = 0.8

    ##### Use Subjects Dataset #####

    # Set k value for laplace smoothing
    bayes = BayesClassifier(k=1)

    # Convert the csv to numpy array
    X, y = bayes.feature_label_from_csv("dbworld_subjects_stemmed.csv")

    # Create 80/20 split for training/testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=TRAINING_SPLIT)

    # Train/fit the model (i.e. calculate the probs for each word)
    bayes.fit(X_train, y_train)

    # Run prediction on test set (20% of original set)
    labels = bayes.predict(X_test)

    print('********My NB (Subjects)********')
    print('Predicted: \t\t' + str(labels))
    print('Actual:    \t\t' + str(y_test))
    print('f-score:   \t\t' + str(calculate_f_score(labels, y_test)))

    # Run same data sets with sklearn models
    bayes_sklearn = MultinomialNB()
    bayes_sklearn.fit(X_train, y_train)
    labels = bayes_sklearn.predict(X_test)

    print('********Sklearn NB (Subjects)********')
    print('Predicted:\t\t' + str(labels))
    print('Actual:   \t\t' + str(y_test))
    print('f-score:   \t\t' + str(calculate_f_score(labels, y_test)))

    ##### Use Bodies Dataset #####

    # Set k value for laplace smoothing
    bayes = BayesClassifier(k=1)

    # Convert the csv to numpy array
    X, y = bayes.feature_label_from_csv("dbworld_bodies_stemmed.csv")

    # Create 80/20 split for training/testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=TRAINING_SPLIT)

    # Train/fit the model (i.e. calculate the probs for each word)
    bayes.fit(X_train, y_train)

    # Run prediction on test set (20% or original set)
    labels = bayes.predict(X_test)

    print('********My NB (Bodies)********')
    print('Predicted: \t\t' + str(labels))
    print('Actual:    \t\t' + str(y_test))
    print('f-score:   \t\t' + str(calculate_f_score(labels, y_test)))

    # Run same data sets with sklearn models
    bayes_sklearn = MultinomialNB()
    bayes_sklearn.fit(X_train, y_train)
    labels = bayes_sklearn.predict(X_test)

    print('********Sklearn NB (Bodies)********')
    print('Predicted:\t\t' + str(labels))
    print('Actual:   \t\t' + str(y_test))
    print('f-score:   \t\t' + str(calculate_f_score(labels, y_test)))
