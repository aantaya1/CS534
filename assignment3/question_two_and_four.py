import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_rows', None)

# Apply three classification algorithms to the same ckd_data.zip dataset as in Problem 2. (15 points)
# a. Support Vector Machine with the linear kernel and default parameters (sklearn.svm.SVC).
# b. Support Vector Machine with the RBF kernel and default parameters.
# c. Random forest with default parameters (sklearn.ensemble.RandomForestClassifier).
# Assess all three classification algorithms using the following protocol:
# i. Use 80% of each class data to train your classifier and the remaining 20% to test it.
# ii. Report the f-measure of the algorithm’s performance on the training and test sets.


class ChronicKidneyDiseaseClassification:
    def __init__(self):
        self.data = self.load_data()
        train_data, target = self.prepare_data()

        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
            train_data, target, random_state=0, train_size=0.8)

    def load_data(self):

        data = []
        start = False

        with open('ckd_data/chronic_kidney_disease_full.arff') as d:
            for line in csv.reader(d):
                if '@data' in line:
                    start = True
                    continue
                if start:
                    data.append(line)

        return data

    def prepare_data(self):
        df = pd.DataFrame(self.data)

        for key in df.keys():
            df[key].replace({'?': np.nan, '\t?': np.nan}, inplace=True)

        # Red Blood Cells
        df[5] = df[5].map({'normal': int(1), 'abnormal': int(0)})

        # Pus Cell (nominal)
        df[6] = df[6].map({'normal': int(1), 'abnormal': int(0)})

        # Pus Cell clumps (nominal)
        df[7] = df[7].map({'present': int(1), 'notpresent': int(0)})

        # Bacteria (nominal)
        df[8] = df[8].map({'present': int(1), 'notpresent': int(0)})

        # Hypertension (nominal)
        df[18] = df[18].map({'yes': int(1), 'no': int(0)})

        # Diabetes Mellitus (nominal)
        df[19] = df[19].map({'yes': int(1), 'no': int(0)})

        # Coronary Artery Disease (nominal)
        df[20] = df[20].map({'yes': int(1), 'no': int(0)})

        # Appetite (nominal)
        df[21] = df[21].map({'good': int(1), 'poor': int(0)})

        # Pedal Edema(nominal)
        df[22] = df[22].map({'yes': int(1), 'no': int(0)})

        # Anemia (nominal)
        df[23] = df[23].map({'yes': int(1), 'no': int(0)})

        # Classification
        df[24] = df[24].map({'ckd': int(1), 'notckd': int(0)})

        # Replace unknown values with column mean
        for key in df.keys():
            if int(key) > 23:
                continue
            df[key].astype(str).str.strip('\t\n')
            try:
                median = round(df[key].median(), 2)
                df[key].fillna(median, inplace=True)
            except TypeError as e:
                print(e)

        # For some reason these are giving issues... remove them
        df.drop([37, 76, 133, 214, 230, 369, 400, 401], axis=0, inplace=True)

        target_classifications = df[24]
        df.drop(columns=[24, 25], inplace=True)

        # Convert everything to float
        for key in df.keys():
            df[key] = df[key].astype('float')

        return df, target_classifications

    def standardize_data(self):
        pass

    def regularize_data(self):
        pass

    def apply_logistic_regression(self, regularization=True):
        pass

    def apply_linear_svm(self):
        svc = SVC(kernel='linear').fit(self.data_train.to_numpy(), self.target_train.to_numpy())
        test_predicted = svc.predict(self.data_test.to_numpy())
        train_predicted = svc.predict(self.data_train.to_numpy())

        test_score = f1_score(self.target_test, test_predicted)
        train_score = f1_score(self.target_train, train_predicted)

        print("SVM (Linear Kernel) F-Measure Score (train): " + str(train_score))
        print("SVM (Linear Kernel) F-Measure Score (test): " + str(test_score))

    def apply_rbf_svm(self):
        # Gamma is auto by default. Set explicitly to remove warnings at runtime.
        svc = SVC(kernel='rbf', gamma='auto').fit(self.data_train.to_numpy(), self.target_train.to_numpy())
        test_predicted = svc.predict(self.data_test.to_numpy())
        train_predicted = svc.predict(self.data_train.to_numpy())

        test_score = f1_score(self.target_test, test_predicted)
        train_score = f1_score(self.target_train, train_predicted)

        print("SVM (RBF Kernel) F-Measure Score (train): " + str(train_score))
        print("SVM (RBF Kernel) F-Measure Score (test): " + str(test_score))

    def apply_random_forest(self):
        # n_estimator is 10 by default. Set explicitly to remove warnings at runtime.
        random_forest_classifier = RandomForestClassifier(n_estimators=10).fit(self.data_train.to_numpy(), self.target_train.to_numpy())
        test_predicted = random_forest_classifier.predict(self.data_test.to_numpy())
        train_predicted = random_forest_classifier.predict(self.data_train.to_numpy())

        test_score = f1_score(self.target_test, test_predicted)
        train_score = f1_score(self.target_train, train_predicted)

        print("Random Forest Classifier F-Measure Score (train): " + str(train_score))
        print("Random Forest Classifier F-Measure Score (test): " + str(test_score))


if __name__ == '__main__':
    classifier = ChronicKidneyDiseaseClassification()
    classifier.apply_linear_svm()
    classifier.apply_rbf_svm()
    classifier.apply_random_forest()
