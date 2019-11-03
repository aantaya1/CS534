class ChronicKidneyDiseaseClassification:
    def __init__(self):

        pass

    def load_data(self):
        pass

    def replace_unknown_vals_with_average(self):
        pass

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
