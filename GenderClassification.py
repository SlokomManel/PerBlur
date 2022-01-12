# from MovieLensData import load_user_item_matrix, load_gender_vector, load_user_item_matrix_100k, load_user_item_matrix_1m, load_gender_vector_1m
import MovieLensData as MD
import Classifiers
from Utils import one_hot
import Utils
import numpy as np
from Utils import feature_selection, normalize, chi2_selection, normalize2
from sklearn.feature_selection import f_regression, f_classif, chi2
import matplotlib.pyplot as plt
import Models
import pandas as pd


def one_million(classifier):
    """
    :param classifier: this function takes the original user-item matrix as input in addition to T == a vactor of users' gender
    The user-item matrix needs to be normalized
    :return: a classification score
    """
    X = MD.load_user_item_matrix_1m_all()  # max_user=max_user, max_item=max_item)

    T = MD.load_gender_vector_1m()  # max_user=max_user)
    X = Utils.normalize(X)

    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    print("before", X_train.shape)
    print(X_train.shape)

    classifier(X_train, T_train)
    from sklearn.linear_model import LogisticRegression
    random_state = np.random.RandomState(0)
    # model = Models.Dominant_Class_Classifier()
    model = LogisticRegression(penalty='l2', C=1.0,
                               random_state=random_state)  # penalty='l2', C=545.5594781168514, random_state=random_state) #
    # from sklearn.svm import SVC
    # model = SVC(kernel='linear', probability=True, random_state=random_state)
    # from sklearn.dummy import DummyClassifier
    # model = DummyClassifier(strategy='most_frequent')

    model.fit(X_train, T_train)

    Utils.ROC_plot(X_test, T_test, model)  # ROC_plot


def one_million_obfuscated(classifier):
    """
        :param classifier: this function takes the original and obfuscated user-item matrices as input in addition to T == a vactor of users' gender
        The user-item matrix needs to be normalized
        :return: a classification score
        """
    # Read the needed inputs
    T = MD.load_gender_vector_1m()  # max_user=max_user)
    X1 = MD.load_user_item_matrix_1m_all()
    X2 = MD.load_user_item_matrix_1m_masked(file_index=1)

    print(X1.shape, X2.shape, T.shape)
    # Normalization
    X1 = Utils.normalize(X1)
    X2 = Utils.normalize(X2)

    print(list(X1[0, :]))
    print(list(X2[0, :]))

    # Classification
    from sklearn.linear_model import LogisticRegression
    # from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.naive_bayes import GaussianNB
    # from sklearn.naive_bayes import MultinomialNB

    random_state = np.random.RandomState(0)
    model = LogisticRegression(penalty='l2', random_state=random_state)  # C=545.5594781168514,
    # model = SVC(kernel='linear', probability=True, random_state=random_state)
    # model = RandomForestClassifier()
    # model = GaussianNB()
    # model = MultinomialNB()
    Utils.ROC_cv_obf(X1, X2, T, model)


## LastFM data
def hyperTunig_SVM():
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix

    X = LFM.load_user_item_matrix_lfm_All()  # max_user=max_user, max_item=max_item)
    T = LFM.load_gender_vector_lfm()

    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    # defining parameter range
    param_grid = {'classifier__penalty': [0.1, 1],  # , 10, 100
                  'classifier__C': [1, 0.1, 0.01],  # , 0.001, 0.0001
                  'kernel': ['linear']}

    grid = GridSearchCV(SVC(), param_grid, scoring='roc_auc', refit=True, cv=10, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, T_train)

    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)
    grid_predictions = grid.predict(X_test)

    # print classification report
    print(classification_report(T_test, grid_predictions))


def HyperTuning_Logreg():
    # read dataset
    X = LFM.load_user_item_matrix_lfm_All()  # max_user=max_user, max_item=max_item)
    T = LFM.load_gender_vector_lfm()
    X = Utils.normalizze(X)

    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]
    # Grid Search
    logreg = LogisticRegression()
    param = {"C": [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1], "penalty": ["l1", "l2"]}
    clf = GridSearchCV(logreg, param, scoring='roc_auc', refit=True, cv=10)
    clf.fit(X_train, T_train)
    print('Best roc_auc: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_))


def lastFM(classifier):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import chi2 as CHI2
    from sklearn.feature_selection import VarianceThreshold
    X = LFM.load_user_item_matrix_lfm_All()  # max_user=max_user, max_item=max_item)
    T = LFM.load_gender_vector_lfm()  # max_user=max_user)
    # X = Utils.features_square(X, T)
    """find the correct / needed features"""
    # X = feature_selection(X, T, Utils.select_male_female_different)
    # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    # X = chi2_selection(X, T)
    # X = Utils.features_square(X, T)
    """
    X = Utils.features_square(X, T)
    # Save the new LastFM data
    X_new = X.copy()
    output_file = "lastFM/Selected_Features/"
    with open(output_file + "LastFM_10K_Features" + ".csv", 'w') as f:
        for index_user, user in enumerate(X_new):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(int(np.round(rating))) + "\n")
    """
    X = Utils.normalizze(X)

    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]
    # X_train, _ = Utils.random_forest_selection(X_train, T_train)

    print("before", X_train.shape)
    print(X_train.shape)

    classifier(X_train, T_train)
    from sklearn.linear_model import LogisticRegression
    random_state = np.random.RandomState(0)
    # model = Models.Dominant_Class_Classifier()
    model = LogisticRegression(penalty='l2', C=0.1,
                               random_state=random_state)  # penalty='l2', C=545.5594781168514, random_state=random_state) #
    from sklearn.svm import SVC
    # model = SVC(kernel='linear', probability=True, random_state=random_state)
    # from sklearn.dummy import DummyClassifier
    # model = DummyClassifier(strategy='most_frequent')
    # model = RandomForestClassifier()

    model.fit(X_train, T_train)
    Utils.ROC_plot(X_test, T_test, model)  # ROC_plot


def LFM_obfuscated(classifier):
    # X2 = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
    T = LFM.load_gender_vector_lfm()  # max_user=max_user)
    X1 = LFM.load_user_item_matrix_lfm_All()
    X2 = LFM.load_user_item_matrix_lfm_masked(file_index=64)  # max_user=max_user, max_item=max_item)
    # X1 = Utils.features_square(X1, T)
    # X2 = Utils.features_square(X2, T)
    # X2 = X1
    print(X1.shape, X2.shape, T.shape)
    X1 = Utils.normalizze(X1)
    X2 = Utils.normalizze(X2)

    X_train, T_train = X1[0:int(0.8 * len(X1))], T[0:int(0.8 * len(X1))]
    X_test, T_test = X2[int(0.8 * len(X2)):], T[int(0.8 * len(X2)):]

    print(list(X1[0, :]))
    print(list(X2[0, :]))
    # print(X)
    print("before", X_train.shape)
    # X = Utils.remove_significant_features(X, T)
    # _, X_train = Utils.random_forest_selection(X_train, T_train)
    # X = feature_selection(X, T, Utils.select_male_female_different)
    print(X_train.shape)
    from sklearn.linear_model import LogisticRegression
    # from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    # from sklearn import svm
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.naive_bayes import GaussianNB
    # from sklearn.naive_bayes import MultinomialNB

    random_state = np.random.RandomState(0)
    # model = LogisticRegression (penalty='l2', random_state=random_state) # C=545.5594781168514,
    model = SVC(kernel='linear', probability=True, random_state=random_state)
    Utils.ROC_cv_obf(X1, X2, T, model)

    ################## Flixster ################


def flixster(classifier):
    import FlixsterDataSub as FDS
    # X, T, _ = FD.load_flixster_data_subset(file="Flixster/With_Fancy_KNN/subset_FX_O.dat")#subset_2000.txt")
    X = FDS.load_user_item_matrix_FX_All()
    T = FDS.load_gender_vector_FX()

    X = Utils.normalizze(X)

    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    print("before", X_train.shape)
    print(X_train.shape)

    classifier(X_train, T_train)
    # from sklearn.linear_model import LogisticRegression
    random_state = np.random.RandomState(0)
    # model = Models.Dominant_Class_Classifier()
    # model = LogisticRegression(penalty='l2', C=1.0, random_state=random_state) # penalty='l2', C=545.5594781168514, random_state=random_state) #
    from sklearn.svm import SVC
    model = SVC(kernel='linear', probability=True, random_state=random_state)
    # from sklearn.dummy import DummyClassifier
    # model = DummyClassifier(strategy='most_frequent')

    model.fit(X_train, T_train)

    Utils.ROC_plot(X_test, T_test, model)  # ROC_plot


def flixster_obfuscated(classifier):
    """import FlixsterData as FD
    X1, T, _ = FD.load_flixster_data_subset()
    X2,_,_ = FD.load_flixster_data_subset_masked(file_index=12)  # max_user=max_user, max_item=max_item)"""
    import FlixsterDataSub as FDS
    X1 = FDS.load_user_item_matrix_FX_All()
    T = FDS.load_gender_vector_FX()
    X2 = FDS.load_user_item_matrix_FX_masked(file_index=60)

    # X1 = FD.load_user_item_matrix_FD_All()
    # X2 = FD.load_user_item_matrix_FD_masked()
    # T = np.loadtxt('FX_Users6000_Gender.txt', dtype=int)
    # X2 = X1
    print(X1.shape, X2.shape)

    # X1, T = Utils.balance_data(X1, T)
    # X2, T2 = Utils.balance_data(X2, T)
    X1 = Utils.normalizze(X1)
    X2 = Utils.normalizze(X2)
    X_train, T_train = X1[0:int(0.8 * len(X1))], T[0:int(0.8 * len(X1))]
    X_test, T_test = X2[int(0.8 * len(X2)):], T[int(0.8 * len(X2)):]
    print(list(X1[0, :]))
    print(list(X2[0, :]))
    # print(X)
    print("before", X_train.shape)
    # X = Utils.remove_significant_features(X, T)
    # X_train, _ = Utils.random_forest_selection(X_train, T_train)
    # X = feature_selection(X, T, Utils.select_male_female_different)
    print(X_train.shape)
    from sklearn.linear_model import LogisticRegression
    random_state = np.random.RandomState(0)
    # model = LogisticRegression(penalty='l2', random_state=random_state)

    from sklearn.svm import SVC
    model = SVC(kernel='linear', probability=True, random_state=random_state)

    Utils.ROC_cv_obf(X1, X2, T, model)

    # model = LogisticRegression(penalty='l2', random_state=random_state)
    # model.fit(X_train, T_train)
    # Utils.ROC_plot(X_test, T_test, model)

if __name__ == '__main__':
    # load the data, It needs to be in the form N x M where N_i is the ith user and M_j is the jth item. Y, the target,
    # is the gender of every user
    import timeit
    start = timeit.default_timer()

#     one_million(Classifiers.log_reg) # Classifiers.svm_classifier
    one_million_obfuscated(Classifiers.log_reg) # svm_classifier
    # LFM_obfuscated(Classifiers.log_reg)  # log_reg svm_classifier
    # lastFM (Classifiers.log_reg)
    # hyperTunig_SVM ()
    # HyperTuning_Logreg()
    # flixster (Classifiers.svm_classifier)
    # flixster_obfuscated (Classifiers.svm_classifier) #log_reg svm_classifier
    # Sub_FX_obfuscated (Classifiers.log_reg)
    # one_million_obfuscated_ML_FD (Classifiers.log_reg)
    # libimseti_obfuscated(Classifie  rs.log_reg)
    stop = timeit.default_timer()
    print('Time: ', stop - start)


