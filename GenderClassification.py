#from MovieLensData import load_user_item_matrix, load_gender_vector, load_user_item_matrix_100k, load_user_item_matrix_1m, load_gender_vector_1m
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

"""
def flixster(classifier):
    import FlixsterDataSub as FDS
    #X, T, _ = FD.load_flixster_data_subset(file="Flixster/With_Fancy_KNN/subset_FX_O.dat")#subset_2000.txt")
    X = FDS.load_user_item_matrix_FX_All()
    T = FDS.load_gender_vector_FX()
    #X = Utils.normalize(X)
    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    # print(X)
    print("before", X_train.shape)
    # X = Utils.remove_significant_features(X, T)
    # X_train, _ = Utils.random_forest_selection(X_train, T_train)
    # X = feature_selection(X, T, Utils.select_male_female_different)
    print(X_train.shape)

    # X = Utils.normalize(X)
    # X = Utils.standardize(X)
    # X = chi2_selection(X, T)

    classifier(X_train, T_train)
    from sklearn.linear_model import LogisticRegression
    random_state = np.random.RandomState(0)
    # model = Models.Dominant_Class_Classifier()
    model = LogisticRegression(penalty='l2', random_state=random_state)
    model.fit(X_train, T_train)
    Utils.ROC_plot(X_test, T_test, model)


def flixster_obfuscated(classifier):
    # import FlixsterData as FD
    # X1, T, _ = FD.load_flixster_data_subset()
    # X2,_,_ = FD.load_flixster_data_subset_masked(file_index=12)  # max_user=max_user, max_item=max_item)
    import FlixsterDataSub as FDS
    X1 = FDS.load_user_item_matrix_FX_All()
    T = FDS.load_gender_vector_FX()
    X2 = FDS.load_user_item_matrix_FX_masked(file_index=6)

    #X1 = FD.load_user_item_matrix_FD_All()
    #X2 = FD.load_user_item_matrix_FD_masked()
    #T = np.loadtxt('FX_Users6000_Gender.txt', dtype=int)
    # X2 = X1
    print(X1.shape, X2.shape)

    # X1, T = Utils.balance_data(X1, T)
    # X2, T2 = Utils.balance_data(X2, T)
    # X1 = Utils.normalize(X1)
    # X2 = Utils.normalize(X2)
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
    model = LogisticRegression(penalty='l2', random_state=random_state)

    Utils.ROC_cv_obf(X1, X2, T, model)

    model = LogisticRegression(penalty='l2', random_state=random_state)
    # model.fit(X_train, T_train)
    # Utils.ROC_plot(X_test, T_test, model)
"""

def one_million(classifier):
    X = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
    #X = MD.load_user_item_matrix_1m_limited_ratings(limit=1)
    #X = MD.load_user_item_matrix_1m_binary()

    # X = MD.load_user_genre_matrix_100k_obfuscated()
    T = MD.load_gender_vector_1m()  # max_user=max_user)
    #X, T = Utils.balance_data(X, T)

    #X = Utils.normalize(X)
    X = feature_selection(X, T, Utils.select_male_female_different) #, Utils.select_male_female_different
    #X = chi2_selection(X, T)
    #X = Utils.remove_significant_features(X, T)
    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    # print(X)
    print("before", X_train.shape)
    # X = Utils.remove_significant_features(X, T)
    #X_train, _ = Utils.random_forest_selection(X_train, T_train)
    # X = feature_selection(X, T, Utils.select_male_female_different)
    print(X_train.shape)

    # X = Utils.normalize(X)
    # X = Utils.standardize(X)
    # X = chi2_selection(X, T)

    classifier(X_train, T_train)
    from sklearn.linear_model import LogisticRegression
    random_state = np.random.RandomState(0)
    #model = Models.Dominant_Class_Classifier()
    model = LogisticRegression(penalty='l2', random_state=random_state)
    model.fit(X_train, T_train)

    Utils.ROC_plot(X_test, T_test, model)


def one_million_obfuscated(classifier):
    #X2 = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
    T = MD.load_gender_vector_1m()  # max_user=max_user)
    X1 = MD.load_user_item_matrix_1m_all()
    X2 = MD.load_user_item_matrix_1m_masked(file_index=162)  # max_user=max_user, max_item=max_item)

    #X2 = X1
    print(X1.shape, X2.shape, T.shape)

    #X1, T = Utils.balance_data(X1, T)
    #X2, T2 = Utils.balance_data(X2, T)
    #X1 = Utils.normalize(X1)
    #X2 = Utils.normalize(X2)
    X_train, T_train = X1[0:int(0.8 * len(X1))], T[0:int(0.8 * len(X1))]
    X_test, T_test = X2[int(0.8 * len(X2)):], T[int(0.8 * len(X2)):]

    print(list(X1[0,:]))
    print(list(X2[0,:]))
    # print(X)
    print("before", X_train.shape)

    print(X_train.shape)
    from sklearn.linear_model import LogisticRegression
    #from sklearn.svm import SVC
    #from sklearn.ensemble import RandomForestClassifier
    #from sklearn.naive_bayes import GaussianNB
    #from sklearn.naive_bayes import MultinomialNB


    random_state = np.random.RandomState(0)
    model = LogisticRegression(penalty='l2', random_state=random_state)
    #model = SVC(kernel='linear', probability=True, random_state=random_state)
    #model = RandomForestClassifier()
    #model = GaussianNB()
    #model = MultinomialNB()
    Utils.ROC_cv_obf(X1, X2, T, model)

    model = LogisticRegression(penalty='l2', random_state=random_state)
    #model = RandomForestClassifier()
    #model = GaussianNB()
    #model = MultinomialNB()
    #model.fit(X_train, T_train)
    #Utils.ROC_plot(X_test, T_test, model)

"""
def Sub_FX_obfuscated(classifier):
    T = FDS.load_gender_vector_FX()  # max_user=max_user)
    X1 = FDS.load_user_item_matrix_FX_All()
    X2 = FDS.load_user_item_matrix_FX_masked(file_index=26)  # max_user=max_user, max_item=max_item)

    print(X1.shape, X2.shape, T.shape)

    X_train, T_train = X1[0:int(0.8 * len(X1))], T[0:int(0.2 * len(X1))]
    X_test, T_test = X2[int(0.8 * len(X2)):], T[int(0.2 * len(X2)):]

    print(list(X1[0,:]))
    print(list(X2[0,:]))
    print("before", X_train.shape)
    print(X_train.shape)
    from sklearn.linear_model import LogisticRegression

    random_state = np.random.RandomState(0)
    model = LogisticRegression(penalty='l2', random_state=random_state)

    Utils.ROC_cv_obf(X1, X2, T, model)

    model = LogisticRegression(penalty='l2', random_state=random_state)
"""

def one_hundert_k(classifier):
    X = MD.load_user_item_matrix_100k_Complet()  # max_user=max_user, max_item=max_item)
    #X = MD.load_user_genre_matrix_100k()
    T = MD.load_gender_vector_100k()  # max_user=max_user)
    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    # print(X)
    print(X_train.shape)
    # X = Utils.remove_significant_features(X, T)
    #X_train = Utils.random_forest_selection(X_train, T_train)
    # X = feature_selection(X, T, Utils.select_male_female_different)
    print(X_train.shape)

    # X = Utils.normalize(X)
    # X = Utils.standardize(X)
    # X = chi2_selection(X, T)

    classifier(X_train, T_train)


def one_hundert_k_obfuscated(classifier):
    T = MD.load_gender_vector_100k()  # max_user=max_user)
    X1 = MD.load_user_item_matrix_100k()
    X2 = MD.load_user_item_matrix_100k_masked(file_index=30)  # max_user=max_user, max_item=max_item)
    # X2 = X1
    print(X1.shape, X2.shape, T.shape)

    X_train, T_train = X1[0:int(0.8 * len(X1))], T[0:int(0.8 * len(X1))]
    X_test, T_test = X2[int(0.8 * len(X2)):], T[int(0.8 * len(X2)):]

    print(list(X1[0, :]))
    print(list(X2[0, :]))
    # print(X)
    print(X_train.shape)
    from sklearn.linear_model import LogisticRegression

    random_state = np.random.RandomState(0)
    model = LogisticRegression(penalty='l2', random_state=random_state)

    Utils.ROC_cv_obf(X1, X2, T, model)

    model = LogisticRegression(penalty='l2', random_state=random_state)


if __name__ == '__main__':
    # load the data, It needs to be in the form N x M where N_i is the ith user and M_j is the jth item. Y, the target,
    # is the gender of every user
    import timeit
    start = timeit.default_timer()

    #max_user = 6040
    #max_item = 3952
    #X = MD.load_user_item_matrix_1m()#max_user=max_user, max_item=max_item)
    #T = MD.load_gender_vector_1m()#max_user=max_user)

    #print(X.shape, T.shape)
    #OH_T = [one_hot(int(x), 2) for x in T]
    #Classifiers.log_reg(X, T)
    #Classifiers.MLP_classifier(X, T, max_item)
    #one_hundert_k (Classifiers.log_reg)
    #one_hundert_k_obfuscated (Classifiers.log_reg)
    #one_million(Classifiers.log_reg)
    one_million_obfuscated(Classifiers.log_reg)
    #flixster (Classifiers.log_reg)
    #flixster_obfuscated (Classifiers.log_reg)
    #Sub_FX_obfuscated (Classifiers.log_reg)
    #one_million_obfuscated_ML_FD (Classifiers.log_reg)
    #libimseti_obfuscated(Classifie  rs.log_reg)
    stop = timeit.default_timer()
    print('Time: ', stop - start)