import csv
import pandas as pd
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


class ChronicKidneyDiseaseClassification:
    def __init__(self):
        self.data = self.load_data()
        self.replace_unknown_vals_with_average()

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

    def replace_unknown_vals_with_average(self):
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

        # Note: Replace unknown values with column mean
        for key in df.keys():
            if int(key) > 23:
                continue
            print(key)
            df[key].astype(str).str.strip('\t\n')
            try:
                median = df[key].median()
                df[key].fillna(median, inplace=True)
            except TypeError as e:
                print(e)
                for value in df[key].values:
                    print(value)

        # For some reason there are an extra two rows...remove them
        df.drop([400, 401], axis=0, inplace=True)

        self.data = df

    def standardize_data(self):
        pass

    def regularize_data(self):
        pass

    def apply_logistic_regression(self, regularization=True):
        pass

    def apply_linear_svm(self):
        pass

    def apply_rbf_svm(self):
        pass

    def apply_random_forest(self):
        pass


if __name__ == '__main__':
    classifier = ChronicKidneyDiseaseClassification()
    classifier.apply_linear_svm()
    print(classifier.data)
