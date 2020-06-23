"""
Blur(So)me is extension of previous work proposed by Windenberg et al., (BlurMe: Inferring and Obfuscating
User Gender Based on Ratings ) and Strucks et al., (BlurM(or)e: Revisiting Gender Obfuscation
in the User-Item Matrix)

This code is extending previous github repository done by Christopher Strucks (Github Link: https://github.com/STrucks/BlurMore)

In Blur(S)me you need to :
    + Generate json file: "Confidence score" from imputation/knn/few_observed_entries
    + You will read the json file
"""

import json
import json
import MovieLensData as MD
import numpy as np
import Utils
import Classifiers
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def blurMe_1m():
    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    rating_mode = list(['highest', 'avg', 'pred'])[1]
    top = -1
    p = 0.05
    dataset = ['ML', 'Fx', 'Li'][1]
    if dataset == 'ML':
        X = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
        T = MD.load_gender_vector_1m()  # max_user=max_user)
    elif dataset == 'Fx':
        """import FlixsterData as FD
        X, T, _ = FD.load_flixster_data_subset()"""
        ###X, T = FD.load_flixster_data()
        # X1 = FD.load_user_item_matrix_FD_All ()
        # X, T = FD.load_flixster_data()
        # T = np.loadtxt('FX_Users6000_Gender.txt', dtype=int)
        import FlixsterDataSub as FDS
        X = FDS.load_user_item_matrix_FX_Train()
        T = FDS.load_gender_vector_FX()
    else:
        import LibimSeTiData as LD
        X, T, _ = LD.load_libimseti_data2()
    # X = Utils.normalize(X)

    avg_ratings = np.zeros(shape=X.shape[0])
    for index, user in enumerate(X):
        ratings = []
        for rating in user:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[index] = 0
        else:
            avg_ratings[index] = np.average(ratings)

    """ AVERAGE ACROSS MOVIE
    avg_ratings = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = []
        for rating in X[:, item_id]:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[item_id] = 0
        else:
            avg_ratings[item_id] = np.average(ratings)
    """

    # 1: get the set of most correlated movies, L_f and L_m:
    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]
    print("lists")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    cv = StratifiedKFold(n_splits=10)
    coefs = []
    avg_coefs = np.zeros(shape=(len(X_train[1]),))

    random_state = np.random.RandomState(0)
    for train, test in cv.split(X_train, T_train):
        x, t = X_train[train], T_train[train]
        model = LogisticRegression(penalty='l2', random_state=random_state)
        model.fit(x, t)
        # rank the coefs:
        ranks = ss.rankdata(model.coef_[0])
        coefs.append(ranks)
        # print(len(model.coef_[0]),len(X_train[0]))
        avg_coefs += model.coef_[0]

    coefs = np.average(coefs, axis=0)
    coefs = [[coefs[i], i + 1, avg_coefs[i]] for i in range(len(coefs))]
    coefs = np.asarray(list(sorted(coefs)))

    if top == -1:
        values = coefs[:, 2]
        var_val = np.min(np.abs(values))
        index_zero = np.where(np.abs(values) == np.min(np.abs(values)))
        top_male = index_zero[0][0]
        top_female = index_zero[0][-1]
        L_m = coefs[:top_male, 1]
        R_m = 2835 - coefs[:top_male, 0]  # 3952
        C_m = np.abs(coefs[:top_male, 2])
        L_f = coefs[coefs.shape[0] - top_female:, 1]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0] - top_female:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0] - top_female:, 2]
        C_f = list(reversed(np.abs(C_f)))

    else:
        L_m = coefs[:top, 1]
        R_m = 2835 - coefs[:top, 0]  # 3952
        C_m = np.abs(coefs[:top, 2])
        L_f = coefs[coefs.shape[0] - top:, 1]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0] - top:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0] - top:, 2]
        C_f = list(reversed(np.abs(C_f)))

    # print(R_f)
    # L_m = L_m[:500]
    # L_f= L_f[:500]
    print("L_f: ", L_f)
    print("L_m: ", L_m)
    """print(len(L_f))
    L_ff = L_f.copy()
    L_ff = pd.DataFrame(L_ff)
    L_ff.to_csv('L_ff.csv', index = False)
    print("------")
    print(len(L_m))
    L_mm = L_m.copy()
    L_mm = pd.DataFrame(L_mm)
    L_mm.to_csv('L_mm.csv', index = False)"""

    """
    id_index, index_id = MD.load_movie_id_index_dict()
    movies = []
    with open("ml-1m/movies.dat", 'r') as f:
        for line in f.readlines():
            movies.append(line.replace("\n", ""))

    for index, val in enumerate(L_m[0:10]):
        print(index, movies[id_index[int(val)]], C_m[index])
    for index, val in enumerate(L_f[0:10]):
        print(index, movies[id_index[int(val)]], C_f[index])


    movie_dict = MD.load_movie_id_dictionary_1m()
    print("males")
    for id in L_m:
        print(movie_dict[int(id)])

    print("females")
    for id in L_f:
        print(movie_dict[int(id)])
    """
    print("obfuscation")
    # Now, where we have the two lists, we can start obfuscating the data:
    # X = MD.load_user_item_matrix_1m()
    X_obf = np.copy(X)

    # X = Utils.normalize(X)
    # X_obf = Utils.normalize(X_obf)
    prob_m = []  # [p / sum(C_m) for p in C_m]
    prob_f = []  # [p / sum(C_f) for p in C_f]
    print("obfuscation")
    for index, user in enumerate(X):
        print(index)
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index = 0
        # print(k)
        if T[index] == 1:
            added = 0
            safety_counter = 0
            while added < k and safety_counter < 100:
                # select a random movie:
                if sample_mode == 'random':
                    movie_id = L_m[np.random.randint(0, len(L_m))]
                elif sample_mode == 'sampled':
                    movie_id = L_m[np.random.choice(range(len(L_m)), p=prob_m)]
                elif sample_mode == 'greedy':
                    movie_id = L_m[greedy_index]
                    greedy_index += 1
                    if greedy_index >= len(L_m):
                        safety_counter = 100
                if X_obf[index, int(movie_id) - 1] == 0:
                    if rating_mode == 'higest':
                        X_obf[index, int(movie_id) - 1] = 5
                    elif rating_mode == 'avg':
                        X_obf[index, int(movie_id) - 1] = avg_ratings[int(index)]
                    added += 1
                safety_counter += 1
        elif T[index] == 0:
            added = 0
            safety_counter = 0
            while added < k and safety_counter < 100:
                # select a random movie:
                if sample_mode == 'random':
                    movie_id = L_f[np.random.randint(0, len(L_f))]
                elif sample_mode == 'sampled':
                    movie_id = L_f[np.random.choice(range(len(L_f)), p=prob_f)]
                elif sample_mode == 'greedy':
                    movie_id = L_f[greedy_index]
                    greedy_index += 1
                    if greedy_index >= len(L_f):
                        safety_counter = 100
                if X_obf[index, int(movie_id) - 1] == 0:
                    if rating_mode == 'higest':
                        X_obf[index, int(movie_id) - 1] = 5
                    elif rating_mode == 'avg':
                        X_obf[index, int(movie_id) - 1] = avg_ratings[int(index)]
                    added += 1
                safety_counter += 1

    # output the data in a file:
    output_file = ""
    if dataset == 'ML':
        output_file = "ml-1m/BlurMe/Top-500/"
        with open(
                output_file + "All_blurme_obfuscated_" + str(p) + "_" + sample_mode + "_" + rating_mode + "_top" + str(
                        top) + ".dat", 'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    elif dataset == 'Fx':
        output_file = "Flixster/BlurMe/"
        with open(output_file + "TrainingSet_FX_blurme_obfuscated_" + str(
                p) + "_" + sample_mode + "_" + rating_mode + "_top" + str(
            top) + ".dat", 'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    else:
        with open("libimseti/LST_blurme_obfuscated_" + str(p) + "_" + sample_mode + "_" + rating_mode + "_top" + str(
                top) + ".dat", 'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    return X_obf


"""elif dataset == 'Fx':
            import FlixsterData as FD
            output_file = "Flixster/BlurMe/"
            user_id2index, user_index2id = FD.load_user_id_index_dict()
            movie_id2index, movie_index2id = FD.load_movie_id_index_dict()

            with open(output_file + "All_FX_blurme_obfuscatedV1_" + str(p) + "_" + sample_mode + "_" + rating_mode + "_top" + str(
                    top) + ".dat", 'w') as f:
                for index_user, user in enumerate(X_obf):
                    for index_movie, rating in enumerate(user):
                        if rating > 0:
                            f.write(str(user_index2id[index_user]) + "::" + str(movie_index2id[index_movie]) + "::" + str(
                                int(np.round(rating))) + "::000000000\n")"""


def blurMe_100k():
    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    rating_mode = list(['highest', 'avg', 'pred'])[1]

    # 1: get the set of most correlated movies, L_f and L_m:
    X = MD.load_user_item_matrix_100k()  # max_user=max_user, max_item=max_item)
    avg_ratings = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = []
        for rating in X[:, item_id]:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[item_id] = 0
        else:
            avg_ratings[item_id] = np.average(ratings)

    T = MD.load_gender_vector_100k()  # max_user=max_user)
    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    cv = StratifiedKFold(n_splits=10)
    coefs = []

    for train, test in cv.split(X_train, T_train):
        x, t = X_train[train], T_train[train]
        random_state = np.random.RandomState(0)
        model = LogisticRegression(penalty='l2', random_state=random_state)
        model.fit(x, t)
        coefs.append(model.coef_)

    coefs = np.average(coefs, axis=0)[0]
    coefs = [[coefs[i], i + 1] for i in range(len(coefs))]
    coefs = np.asarray(list(sorted(coefs)))
    L_m = coefs[:10, 1]
    L_f = coefs[coefs.shape[0] - 10:, 1]
    L_f = list(reversed(L_f))

    print(L_f)
    print("------")
    print(L_m)

    """
    movie_dict = MD.load_movie_id_dictionary_1m()
    print("males")
    for id in L_m:
        print(movie_dict[int(id)])

    print("females")
    for id in L_f:
        print(movie_dict[int(id)])
    """

    # Now, where we have the two lists, we can start obfuscating the data:
    X = MD.load_user_item_matrix_100k()
    X_obf = MD.load_user_item_matrix_100k()
    p = 0.1
    prob_m = [p / sum(L_m) for p in L_m]
    prob_f = [p / sum(L_f) for p in L_f]
    for index, user in enumerate(X):
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index = 0
        # print(k)
        if T[index] == 1:
            added = 0
            safety_counter = 0
            while added < k and safety_counter < 100:
                # select a random movie:
                if sample_mode == 'random':
                    movie_id = L_m[np.random.randint(0, len(L_m))]
                elif sample_mode == 'sampled':
                    movie_id = L_m[np.random.choice(range(len(L_m)), p=prob_m)]
                elif sample_mode == 'greedy':
                    movie_id = L_m[greedy_index]
                    greedy_index += 1
                    if greedy_index >= len(L_m):
                        safety_counter = 100
                if X_obf[index, int(movie_id) - 1] == 0:
                    if rating_mode == 'higest':
                        X_obf[index, int(movie_id) - 1] = 5
                    elif rating_mode == 'avg':
                        X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id)]
                    added += 1
                safety_counter += 1
        elif T[index] == 0:
            added = 0
            safety_counter = 0
            while added < k and safety_counter < 100:
                # select a random movie:
                if sample_mode == 'random':
                    movie_id = L_f[np.random.randint(0, len(L_f))]
                elif sample_mode == 'sampled':
                    movie_id = L_f[np.random.choice(range(len(L_f)), p=prob_f)]
                elif sample_mode == 'greedy':
                    movie_id = L_f[greedy_index]
                    greedy_index += 1
                    if greedy_index >= len(L_f):
                        safety_counter = 100
                if X_obf[index, int(movie_id) - 1] == 0:
                    if rating_mode == 'higest':
                        X_obf[index, int(movie_id) - 1] = 5
                    elif rating_mode == 'avg':
                        X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id)]
                    added += 1
                safety_counter += 1

    # output the data in a file:
    with open("ml-100k/blurme_obfuscated_" + str(p) + "_" + sample_mode + "_" + rating_mode + ".dat", 'w') as f:
        f.write("user_id,item_id,rating")
        for index_user, user in enumerate(X_obf):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(str(index_user + 1) + "," + str(index_movie + 1) + "," + str(int(rating)) + "\n")
    return X_obf

def blurMePP():
    top = -1
    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    removal_mode = list(['random', 'strategic'])[1]
    # id_index, index_id = MD.load_movie_id_index_dict()
    notice_factor = 2
    p = 0.05
    dataset = ['ML', 'Fx', 'Li'][1]
    if dataset == 'ML':
        # X = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
        # T = MD.load_gender_vector_1m()  # max_user=max_user)
        X = MD.load_user_item_matrix_100k()
        T = MD.load_gender_vector_100k()
    elif dataset == 'Fx':
        """import FlixsterData as FD
        #X, T, _ = FD.load_flixster_data_subset()
        X, T, _ = FD.load_flixster_data_subset_trainingSet()"""
        import FlixsterDataSub as FDS
        X = FDS.load_user_item_matrix_FX_Train()
        T = FDS.load_gender_vector_FX()
    else:
        import LibimSeTiData as LD
        X, T, _ = LD.load_libimseti_data_subset()
    # X = Utils.normalize(X)
    avg_ratings = np.zeros(shape=X.shape[1])
    initial_count = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = []
        for rating in X[:, item_id]:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[item_id] = 0
        else:
            avg_ratings[item_id] = np.average(ratings)
        initial_count[item_id] = len(ratings)
    max_count = initial_count * notice_factor
    # 1: get the set of most correlated movies, L_f and L_m:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    cv = StratifiedKFold(n_splits=10)
    coefs = []
    avg_coefs = np.zeros(shape=(len(X[1]),))

    random_state = np.random.RandomState(0)
    for train, test in cv.split(X, T):
        x, t = X[train], T[train]
        model = LogisticRegression(penalty='l2', random_state=random_state)
        model.fit(x, t)
        # rank the coefs:
        ranks = ss.rankdata(model.coef_[0])
        coefs.append(ranks)
        # print(len(model.coef_[0]),len(X_train[0]))
        avg_coefs += model.coef_[0]

    coefs = np.average(coefs, axis=0)
    coefs = [[coefs[i], i + 1, avg_coefs[i]] for i in range(len(coefs))]
    coefs = np.asarray(list(sorted(coefs)))
    if top == -1:
        values = coefs[:, 2]
        index_zero = np.where(np.abs(values) == np.min(np.abs(values)))
        top_male = index_zero[0][0]
        top_female = index_zero[0][-1]
        L_m = coefs[:top_male, 1]
        R_m = 2835 - coefs[:top_male, 0]  # 3952
        C_m = np.abs(coefs[:top_male, 2])
        L_f = coefs[coefs.shape[0] - top_female:, 1]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0] - top_female:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0] - top_female:, 2]
        C_f = list(reversed(np.abs(C_f)))

    else:
        L_m = coefs[:top, 1]
        R_m = 2835 - coefs[:top, 0]  # 3952
        C_m = np.abs(coefs[:top, 2])
        L_f = coefs[coefs.shape[0] - top:, 1]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0] - top:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0] - top:, 2]
        C_f = list(reversed(np.abs(C_f)))

    """print(len(L_f))
    L_ff = L_f.copy()
    L_ff = pd.DataFrame(L_ff)
    L_ff.to_csv('L_ff_impute_Opposite.csv', index=False)
    print("------")
    print(len(L_m))
    L_mm = L_m.copy()
    L_mm = pd.DataFrame(L_mm)
    L_mm.to_csv('L_mm_impute_Opposite.csv', index=False)"""
    # Now, where we have the two lists, we can start obfuscating the data:
    # X = MD.load_user_item_matrix_1m()
    # np.random.shuffle(X)
    # print(X.shape)

    X_obf = np.copy(X)
    total_added = 0
    for index, user in enumerate(X):
        print(index)
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index_m = 0
        greedy_index_f = 0
        # print(k)
        added = 0
        if T[index] == 1:
            safety_counter = 0
            while added < k and safety_counter < 1000:
                if greedy_index_m >= len(L_m):
                    safety_counter = 1000
                    continue
                if sample_mode == 'greedy':
                    movie_id = L_m[greedy_index_m]
                if sample_mode == 'random':
                    movie_id = L_m[np.random.randint(0, len(L_m))]
                greedy_index_m += 1
                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id) - 1]])
                if rating_count > max_count[int(movie_id) - 1]:
                    continue
                if X_obf[index, int(movie_id) - 1] == 0:
                    X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id) - 1]
                    added += 1
                safety_counter += 1
        elif T[index] == 0:
            safety_counter = 0
            while added < k and safety_counter < 1000:
                if greedy_index_f >= len(L_f):
                    safety_counter = 1000
                    continue
                if sample_mode == 'greedy':
                    movie_id = L_f[greedy_index_f]
                if sample_mode == 'random':
                    movie_id = L_f[np.random.randint(0, len(L_f))]
                greedy_index_f += 1
                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id) - 1]])
                if rating_count > max_count[int(movie_id) - 1]:
                    continue

                if X_obf[index, int(movie_id) - 1] == 0:
                    X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id) - 1]
                    added += 1
                safety_counter += 1
        total_added += added

    # Now remove ratings from users that have more than 200 ratings equally:
    """nr_many_ratings = 0
    for user in X:
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            nr_many_ratings += 1
    nr_remove = total_added / nr_many_ratings

    for user_index, user in enumerate(X):
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0], size=(int(nr_remove),),
                                                      replace=False)
            X_obf[user_index, to_be_removed_indecies] = 0"""

    nr_many_ratings = 0
    for user in X:
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            nr_many_ratings += 1
    print(nr_many_ratings)
    nr_remove = total_added / nr_many_ratings

    for user_index, user in enumerate(X):
        print("user: ", user_index)
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            index_m = 0
            index_f = 0
            rem = 0
            if T[user_index] == 1:
                safety_counter = 0
                # We note that if we add safety_counter < 1000 in the while we have a higher accuracy than if we keep it in the if
                while (rem < nr_remove) and safety_counter < 1000:
                    if index_f >= len(L_f):
                        safety_counter = 1000
                        continue

                    if removal_mode == "random":
                        to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0],
                                                                  size=(int(nr_remove),),
                                                                  replace=False)  # , replace=False)
                    if removal_mode == "strategic":
                        to_be_removed_indecies = L_f[index_f]
                    index_f += 1

                    if X_obf[user_index, int(to_be_removed_indecies) - 1] != 0:
                        X_obf[user_index, int(to_be_removed_indecies) - 1] = 0
                        rem += 1
                    safety_counter += 1

            elif T[user_index] == 0:

                while (rem < nr_remove) and safety_counter < 1000:
                    if index_m >= len(L_m):  # and safety_counter < 1000:
                        safety_counter = 1000
                        continue

                    if removal_mode == "random":
                        to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0],
                                                                  size=(int(nr_remove),),
                                                                  replace=False)  # , replace=False)
                    # X_obf[user_index, to_be_removed_indecies] = 0

                    if removal_mode == "strategic":
                        to_be_removed_indecies = L_m[index_m]
                    index_m += 1

                    if X_obf[user_index, int(to_be_removed_indecies) - 1] != 0:
                        X_obf[user_index, int(to_be_removed_indecies) - 1] = 0
                        rem += 1
                    safety_counter += 1

    # finally, shuffle the user vectors:
    # np.random.shuffle(X_obf)
    # output the data in a file:
    output_file = ""
    if dataset == 'ML':
        output_file = "ml-100k/BlurMore/"  # "ml-1m/BlurMore/"
        with open(output_file + "TrainingSet_blurmepp_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(
                notice_factor) + "_" + str(removal_mode) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(
                            str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                                int(np.round(rating))) + "::000000000\n")

    elif dataset == 'Fx':
        output_file = "Flixster/BlurMore/AverageRating/"
        with open(output_file + "TrainingSet_FX_blurmepp_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(
                notice_factor) + "_" + str(removal_mode) + ".dat",  # + "_" + str(removal_mode) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    else:
        with open("libimseti/LST_blurmepp_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(notice_factor) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    return X_obf


"""    elif dataset == 'Fx':
        import FlixsterData as FD
        output_file = "Flixster/BlurMore/"
        user_id2index, user_index2id = FD.load_user_id_index_dict()
        movie_id2index, movie_index2id = FD.load_movie_id_index_dict()

        with open(output_file + "TrainingSet_FX_blurmepp_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(notice_factor) + ".dat", #+ "_" + str(removal_mode) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(user_index2id[index_user]) + "::" + str(movie_index2id[index_movie]) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")
"""


# -----------------------------------------------

def blurSome():
    # NN_TrainingSet_Before_Opposite_Gender_KNN_fancy_imputation_1m_k_30
    # NN_All_Before_Opposite_Gender_KNN_fancy_imputation_1m_k_30
    ########
    # NN_All_Before_allUsers_KNN_fancy_imputation_1m_k_30
    # NN_TrainingSet_Before_allUsers_KNN_fancy_imputation_1m_k_30
    """
    ## Step 1: Create personalized list of indicative items per user based on the saved json file from imputation
    item_choice = {}
    with open('ml-1m/With_Fancy_KNN/test_Confidence_Score_Items_Selection/NN_TrainingSet_Before_allUsers_KNN_fancy_imputation_1m_k_30.json') as json_file:
        data = json.load(
            json_file)
    len_dict = {}
    for key, value in data.items():
        # print (value)
        length = []
        for v in value:
            # print (len(v))
            length.append(len(v))
        len_dict[int(key)] = length"""

    top = -1
    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    removal_mode = list(['random', 'strategic'])[1]
    # id_index, index_id = MD.load_movie_id_index_dict()
    notice_factor = 2
    p = 0.1
    dataset = ['ML', 'Fx', 'Li'][1]
    if dataset == 'ML':
        X = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
        T = MD.load_gender_vector_1m()  # max_user=max_user)
        # X_filled = MD.load_user_item_matrix_1m_complet()
        """X = MD.load_user_item_matrix_100k_train()
        T = MD.load_gender_vector_100k()
        #X_filled = MD.load_user_item_matrix_100k_Complet ()"""
    elif dataset == 'Fx':
        import FlixsterData as FD
        # X, T = FD.load_flixster_data_subset()
        """X = FD.load_user_item_matrix_FD()
        X1, T = FD.load_flixster_data()
        X, T, _ = FD.load_flixster_data_subset()
        X_filled = FD.load_user_item_matrix_FD_All()"""
        import FlixsterDataSub as FDS
        X = FDS.load_user_item_matrix_FX_All()
        T = FDS.load_gender_vector_FX()
        X_filled = FDS.load_user_item_FX_Complet()
    else:
        import LibimSeTiData as LD
        X, T, _ = LD.load_libimseti_data_subset()
    # X = Utils.normalize(X)

    avg_ratings = np.zeros(shape=X.shape[1])
    initial_count = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = []
        for rating in X[:, item_id]:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[item_id] = 0
        else:
            avg_ratings[item_id] = np.average(ratings)
        initial_count[item_id] = len(ratings)
    max_count = initial_count * notice_factor
    # 1: get the set of most correlated movies, L_f and L_m:

    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    cv = StratifiedKFold(n_splits=10)
    coefs = []
    avg_coefs = np.zeros(shape=(len(X[1]),))

    random_state = np.random.RandomState(0)
    for train, test in cv.split(X, T):
        x, t = X[train], T[train]
        model = LogisticRegression(penalty='l2', random_state=random_state)
        model.fit(x, t)
        # rank the coefs:
        ranks = ss.rankdata(model.coef_[0])
        coefs.append(ranks)
        # print(len(model.coef_[0]),len(X_train[0]))
        avg_coefs += model.coef_[0]

    coefs = np.average(coefs, axis=0)
    coefs = [[coefs[i], i + 1, avg_coefs[i]] for i in range(len(coefs))]
    coefs = np.asarray(list(sorted(coefs)))
    if top == -1:
        values = coefs[:, 2]
        index_zero = np.where(np.abs(values) == np.min(np.abs(values)))
        top_male = index_zero[0][0]
        top_female = index_zero[0][-1]
        L_m = coefs[:top_male, 1]
        R_m = 3952 - coefs[:top_male, 0]  # 3952
        C_m = np.abs(coefs[:top_male, 2])
        L_f = coefs[coefs.shape[0] - top_female:, 1]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0] - top_female:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0] - top_female:, 2]
        C_f = list(reversed(np.abs(C_f)))

    else:
        L_m = coefs[:top, 1]
        R_m = 3952 - coefs[:top, 0]  # 3952
        C_m = np.abs(coefs[:top, 2])
        L_f = coefs[coefs.shape[0] - top:, 1]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0] - top:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0] - top:, 2]
        C_f = list(reversed(np.abs(C_f)))

    """L_mm = list(map(int, L_m))
    L_mm = list(map(lambda x: x-1, L_mm))#[:500]
    L_ff = list(map(int, L_f))
    L_ff= list(map(lambda x: x-1, L_ff))#[:500]

    for z in range(len(X)):
        print(z)
        values = len_dict[z]
        lst_j = []
        # list of neighbors ordered / ranked by weight for user i
        user_item = list(np.argsort(values))#[::-1])
        #lst = X_filled [z]
        #lst = list(map(lst.__getitem__, user_item))
        if (len(user_item) == len(values)):
            p = 0
            while p < len(values):
                if T [z] == 0:
                    f= user_item.pop(0) #np.argmin (lst)
                    if f in L_ff:
                        if f not in lst_j:
                            lst_j.append(f)

                elif T [z]== 1:
                    m = user_item.pop(0)
                    if m in L_mm:
                        if m not in lst_j:
                            lst_j.append(m)
                p+= 1
            item_choice [z] = lst_j
    print("item_choice: ", item_choice)

    with open(#Flixster/With_Fancy_KNN/FX_NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice_Top50IndicativeItems.json   ml-100k/With_Fancy_KNN/ml100k_NN_All_AllUsers_Neighbors_Weight_K_30_item_choice_AllIndicativeItems.json
            "Flixster/With_Fancy_KNN/FX_NN_TrainingSet_2370_AllUsers_Neighbors_Weight_K_30_item_choice_Top50IndicativeItems.json",
            "w") as fp:
        json.dump(item_choice, fp, cls=NpEncoder)"""

    # NN_All_AllUsers_Neighbors_Weight_K_30_item_choice_AllIndicativeItems
    # NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice_AllIndicativeItems

    # NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice | for top-100 most indicative items
    # NN_All_AllUsers_Neighbors_Weight_K_30_item_choice | for top-100 most indicative items

    # NN_All_AllUsers_Neighbors_Weight_K_30_item_choice_Top75IndicativeItems_SortFromLowToHigh

    # NN_All_AllUsers_Neighbors_Weight_K_30_item_choice_Top75IndicativeItems
    # NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice_Top75IndicativeItems
    # ml-1m/user_based_imputation/With_Fancy_KNN/test_Confidence_Score_Items_Selection/NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice_Top75IndicativeItems_SortFromLowToHigh.json'

    # ml-100k/With_Fancy_KNN/ml100k_NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice_Top50IndicativeItems.json

    with open(
            'Flixster/With_Fancy_KNN/FX_NN_All_2370_AllUsers_Neighbors_Weight_K_30_item_choice_Top75IndicativeItems.json') as json_file:
        item_choice = json.load(
            # ml-1m/user_based_imputation/With_Fancy_KNN/test_Confidence_Score_Items_Selection/NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice_Top50IndicativeItems.json
            json_file)

    # Now, where we have the two lists, we can start obfuscating the data:
    X_obf = np.copy(X)
    total_added = 0
    for index, user in enumerate(X):
        print(index)
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index = 0
        added = 0
        mylist = list(item_choice.values())
        safety_counter = 0

        while added < k and safety_counter < 1000:
            if greedy_index >= len(mylist[index]):
                safety_counter = 1000
                continue
            if sample_mode == 'greedy':
                vec = mylist[index]
                movie_id = vec[greedy_index]
            if sample_mode == 'random':
                movie_id = vec[np.random.randint(0, len(vec))]
            greedy_index += 1
            rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, movie_id]])
            if rating_count > max_count[movie_id]:
                continue

            if X_obf[index, movie_id] == 0:
                X_obf[index, movie_id] = X_filled[
                    index, movie_id]  # avg_ratings[int(movie_id)] #X_filled[index, movie_id]
                added += 1
            safety_counter += 1
        total_added += added

    # Now remove ratings from users that have more than 200 ratings equally:
    # L_ff = [978,3844,1897,1345,2261,1769,906,3281,2635,94,3606,461,469,3117,2151,546,2365,2857,1041,314,28,932,1480,3077,2331,2621,1648,3896,1804,3515,1566,928,915,3716,222,616,3565,279,3133,3224,1658,294,1654,3724,176,248,141,2820,372,2334,3211,918,2416,1286,2369,3682,2613,562,3539,1207,2853,329,1518,532,959,2625,2379,4,211,2335,1188,3325,1645,3902,117,283,1806,2598,425,653,669,3655,1967,2690,1015,1954,3556,3686,661,1620,1150,3079,3145,3445,2495,3534,3449,1615,1464,2498,1969,8,3595,2132,445,261,1236,2774,270,3662,1099,3625,82,971,1264,2639,3921,153,1423,1020,215,3450,267,199,2358,1762,839,1086,2091,2145,2664,1962,345,1896,2805,2448,1059,3870,3249,2972,2014,1111,2881,3340,3461,11,3857,1483,668,17,2085,5,72,3668,3528,2974,788,2788,2205,3439,3604,2532,3240,2141,2816,2506,3707,1856,2463,1678,1699,2862,3594,386,245,3702,1226,3864,1088,587,3620,1549,3441,206,2160,428,1585,3192,2267,1394,405,2057,531,3247,3672,477,2443,438,2144,2726,2786,2907,1959,1151,1301,3567,3105,2080,224,2241,3577,1582,2348,3328,2677,3936,3866,3391,3512,950,3200,2903,1463,423,3911,2337,3438,2364,3454,1172,875,3318,2291,1479,146,2280,41,105,1894,479,2946,2990,1449,2681,3679,3171,1385,47,414,3699,1571,927,2891,2346,2442,2406,1186,1031,2733,1011,2549,1812,1013,709,3024,3399,3649,3102,2104,2496,1836,880,3418,3206,2248,2800,3126,2741,552,208,3459,2995,3185,1043,338,780,3497,2710,3483,2307,1232,1047,1327,2798,3165,3720,3029,900,1921,3457,2052,2949,2163,2952,3122,337,1104,3093,920,2316,309,2042,3153,1441,3302,911,3284,2630,3124,2659,2122,203,1688,3741,778,695,2899,2259,197,805,2305,247,2158,2154,29,955,1546,3605,917,538,1941,49,1268,2064,3266,272,2461,781,2926,2050,3090,1129,994,1831,1171,2015,3481,3839,3478,554,1900,3396,1874,1717,3167,754,1183,3161,1816,3051,2481,2173,52,3586,594,1814,3635,3923,3825,334,2876,1983,3204,330,3054,2661,3208,362,3944,2739,1066,464,1189,1162,2852,2870,455,1704,3264,269,3071,159,575,3274,519,3326,3821,2975,1600,1021,3613,3250,2815,3518,2973,2202,3791,2232,3795,26,131,2555,3446,3924,2128,470,2166,2020,2505,733,3675,1422,3735,1081,1807,1824,1917,3492,1547,2043,350,2533,1904,97,1991,2920,2071,671,2940,2888,2900,2875,1321,1444,1300,558,2864,961,807,2155,997,1875,3877,2285,3781,1407,1729,1605,1084,252,3771,102,1926,3786,327,1169,2989,1346,2320,3727,420,2479,246,216,1631,3705,3063,152,2219,1310,3708,3041,2964,3287,1640,2062,621,2572,408,3428,2514,1949,3317,3663,910,2094,1586,1984,88,1744,3183,3514,1447,2589,1523,3040,2643,1357,39,3290,242,316,447,1,449,1397,3055,612,2395,1672,2161,2125,3222,2713,2569,2732,818,2727,507,1863,586,560,2511,3046,3726,1759,3134,3405,1290,3104,2001,149,2931,2704,3562,2388,1323,2206,2290,3072,2718,186,1263,1132,1857,237,3917,491,1832,3503,3476,2797,1032,1906,2190,1777,465,2245,1642,2719,1593,581,3776,2779,2860,1893,473,3690,2963,2799,1487,838,86,73,3926,3081,2324,3199,99,1468,3737,2435,2336,407,360,753,2609,434,2951,3355,2918,3039,2269,2752,1217,3074,3467,2133,3178,65,2056,1076,1336,2548,2565,3269,2574,650,2494,1152,2738,3584,2101,3881,429,3775,1756,3011,332,1785,3813,2699,3637,495,2345,3780,37,3137,3014,926,3550,3951,1421,1010,2722,2114,140,2784,1956,3861,2355,1419,87,422,1656,3932,1235,3420,964,2763,173,3798,383,1258,2485,1529,320,2762,1833,40,2469,2126,165,3739,1570,2692,2693,3841,2240,277,1621,2773,358,1862,3316,2227,3186,1657,3471,2872,3004,217,708,1337,1534,1659,3173,2953,1429,1809,3427,3070,209,1260,3412,3878,2421,3874,3624,2966,3808,846,1660,48,3255,2182,293,1934,3390,2238,348,1381,1278,2748,2789,2992,1114,2037,1509,3300,662,1034,2347,467,1860,2708,3108,435,490,715,1944,2361,474,2579,1976,1988,2573,639,809,2636,2444,1709,2683,3033,343,3432,2539,263,2583,2396,2927,2142,3833,2197,1004,2923,844,3797,3906,1299,182,506,494,3064,14,2991,533,170,3843,3768,1784,746,3092,1409,280,3912,3554,3810,2937,392,3155,229,2554,3949,2892,1457,1624,452,1899,2886,2622,2414,3873,596,1767,367,1527,1023,69,1027,776,2827,476,195,2008,1064,914,2000,3591,595,1686,3611,2801,1442,3409,3076,919,42,3389,2778,1245,2623,2873,1438,1616,129,3194,389,2782,872,3715,1225,687,1456,697,2247,3146,2666,692,864,3945,3060,3232,2257,3834,192,1573,1909,1993,649,1291,1242,2083,359,271,462,331,3627,2148,2812,798,3698,886,3664,2310,318,1758,2925,617,2321,3931,3714,2749,877,1093,2817,3653,3005,1683,81,177,1349,904,1022,2714,2649,278,593,200,357,3230,1943,1788,909,553,2735,2351,2868,3667,2299,262,2585,2281,1101,1112,3089,3410,767,3470,1223,3020,3547,2418,218,3367,1240,3479,1200,365,80,1507,3433,363,2465,1406,3696,1005,3618,2531,1938,722,1815,1340,2051,376,1961,898,126,1204,2590,1653,3527,3286,148,2772,1755,3486,3083,1362,2159,3706,899,1990,1428,2716,580,3757,1453,2065,2352,2688,516,2790,341,3268,132,179,982,3235,2665,3303,736,799,2657,2439,2760,1913,1130,2385,1957,949,1008,2028,371,3257,3296,3114,513,1139,2006,3732,1026,3783,1561,3435,2650,3042,381,613,3526,2468,1535,1024,3837,2266,670,3901,2180,3684,2381,1216,1316,2146,2487,2405,2436,1030,3767,921,2667,3357,266,3660,905,291,3665,974,54,1177,896,1533,3047,3941,1951,1613,647,2454,1380,232,1212,3787,3044,3501,1446,101,1730,2203,1770,3922,1955,3363,2466,50,1922,913,2912,543,2846,1384,2614,3488,2408,1602,1234,1440,2342,837,2066,1410,1839,2038,1269,1619,3456,2399,1028,2084,3197,2695,2467,2097,1328,1829,2233,2397,2551,220,2010,2761,3442,25,3140,1666,2007,2956,2186,446,1080,2417,2823,2368,2431,3121,2370,1117,2047,2359,783,230,2239,940,1500,3847,1203,3723,2265,339,1902,448,3058,1801,3157,3370,1085,1277,3130,2884,239,1120,602,1252,2917,1580,23,3785,2818,1285,3894,824,2156,1267,3671,1916,1126,1107,2586,1427,3097,459,1211,375,1194,496,3061,3557,2109,1854,1417,762,2523,2400,2255,1398,3507,1796,134,1840,1669,2638,2019,3661,1919,3341,891,2765,1825,3158,317,1471,2535,571,1332,2456,1180,1526,396,1035,187,3258,3883,599,74,2750,1662,2849,3701,3350,500,2211,1068,1219,725,2318,3855,1937,1594,3840,3884,3246,1272,3049,3131,1311,1296,489,238,2553,2284,2783,1313,2550,79,1850,1685,243,3641,301,748,135,44,881,609,406,3676,160,998,3068,2096,3181,2568,3885,136,3156,3217,1522,2445,3003,968,1297,615,2813,2814,3493,3892,302,1039,3119,2474,3166,1465,3103,698,441,2003,555,1486,2908,1720,3440,2477,2184,1197,1055,3345,2915,1324,577,2021,1353,902,2423,3395,2072,2833,960,2715,1014,3245,2608,3596,3223,2168,945,3553,2709,1617,608,78,3848,12,525,168,113,3761,2382,3366,1073,3406,2930,3352,3359,2258,1931,264,3742,106,2575,1537,1053,3644,3027,1887,2679,1351,108,184,3915,369,2844,2934,3242,2804,3889,2947,2279,2945,534,559,3551,3160,437,1209,2402,1655,3711,3853,2070,1279,3718,1191,1834,1562,2831,2928,3854,3474,3612,2106,1985,2641,939,295,3642,1805,1116,2282,3095,2982,2226,3304,2136,3542,249,2357,3209,1270,1078,3495,1811,2260,3491,985,1178,3353,2194,3067,2658,2392,3288,2153,1083,3529,19,1293,2127,2746,2855,2313,1972,2545,1167,2803,3802,2660,1622,972,1668,1575,2617,3578,288,1173,2576,1514,1497,1199,3677,3434,2039,934,2304,2527,1674,347,2834,834,3538,3050,98,866,1981,257,2839,1681,3136,521,688,213,2314,3043,62,879,2429,2559,1572,2044,3169,457,2524,2962,3685,2686,188,2969,2878,139,1641,201,3426,2562,3592,398,3869,1000,606,821,1936,2373,2309,887,2476,214,3899,2685,3354,2054,2825,2387,3254,3164,634,2076,3631,2619,3453,853,1994,1488,3305,1220,2618,1185,2293,3614,415,2944,769,3460,1476,2137,1511,2582,3437,1595,114,2302,2678,1097,1822,2263,2540,2914,1935,540,806,2475,2440,2193,2983,2777,2768,1623,1436,254,1545,3499,2674,167,3472,2002,1548,3463,3601,2372,2916,779,3704,391,3536,3794,3293,1773,751,3784,3343,1966,3532,867,2053,2102,1404,3640,2829,3179,3574,3188,929,411,2251,2271,814,1636,1556,3086,865,1133,3336,907,703,385,3639,3579,2933,830,2879,570,840,2354,659,903,2464,3852,2294,3657,2736,1253,1515,2905,1724,2697,3482,2055,1334,1771,2528,3132,1461,1485,3335,2998,966,3828,1163,1484,3597,710,3517,1747,2808,2599,1661,3609,3872,2858,3331,1221,656,775,1042,3537,2446,110,2993,2503,3907,3890,3856,3829,3815,3750,3656,3650,3630,3621,3607,3589,3583,3582,3561,3560,3558,3541,3530,3455,3411,3383,3382,3369,3356,3348,3332,3323,3315,3291,3279,3278,3237,3234,3231,3229,3227,3226,3195,3193,3191,3170,3080,3065,3059,3023,3009,2980,2958,2957,2954,2910,2909,2895,2845,2838,2832,2703,2698,2684,2680,2604,2603,2601,2595,2592,2588,2584,2564,2547,2543,2508,2489,2438,2319,2274,2270,2230,2229,2228,2225,2224,2222,2220,2217,2216,2199,2198,2030,1868,1847,1843,1838,1828,1823,1819,1818,1813,1808,1803,1802,1800,1790,1789,1787,1786,1781,1778,1776,1775,1774,1768,1766,1765,1763,1761,1757,1751,1745,1742,1740,1738,1737,1736,1723,1716,1712,1710,1708,1706,1705,1700,1698,1697,1691,1638,1637,1634,1628,1618,1607,1579,1578,1577,1576,1568,1560,1559,1557,1540,1536,1530,1524,1521,1512,1506,1505,1492,1491,1481,1478,1469,1467,1462,1452,1451,1448,1443,1435,1434,1433,1424,1418,1403,1402,1400,1386,1368,1338,1319,1318,1314,1309,1308,1239,1229,1195,1182,1166,1159,1158,1157,1156,1155,1146,1145,1143,1141,1140,1137,1122,1118,1110,1109,1108,1106,1075,1074,1072,1065,1052,1048,1045,1001,995,983,979,894,890,883,873,871,857,856,855,845,825,822,819,817,816,812,797,795,794,777,774,773,772,770,768,763,758,752,740,739,738,730,727,723,721,717,713,699,693,690,689,686,683,677,676,675,655,654,651,646,641,636,629,625,624,622,620,604,591,404,403,402,400,399,395,323,286,285,284,221,143,138,133,127,115,109,91,51,3404,1930,2847,3935,1372,3034,3703,1569,1330,374,481,2110,1821,427,2082,2470,3413,1513,2460,3736,1077,719,2919,3408,804,2441,384,597,46,3570,1940,191,67,2668,3733,2099,1367,986,2776,147,3632,2939,3001,1946,36,2078,1625,1096,497,2655,2986,1923,1564,193,1598,1797,1017,198,3730,2338,3744,2169,307,1359,912,836,3748,3600,346,3364,2218,1474,1877,712,957,2705,2516,1590,3740,3516,123,1415,3480,3282,2242,236,3443,2948,3387,353,1190,1105,56,737,3759,205,843,726,2691,832,664,660,2031,3909,1344,984,3522,2534,583,2339,2611,2254,1373,3747,1925,1343,1820,2717,421,3919,3251,313,235,366,1050,3835,3572,678,121,2214,1070,827,529,84,1714,3376,3202,2213,975,868,1364,484,2821,701,584,530,3888,3380,3312,1115,142,520,2556,1727,3447,512,2404,1632,234,2074,3109,2210,1360,787,3219,1870,3321,3220,2742,2484,2277,2235,1852,1630,1510,859,644,579,282,1134,3599,3057,897,3277,2824,2275,1835,212,1425,3850,3085,251,1553,3314,3378,790,874,757,1908,1764,601,2292,3886,3763,1275,352,472,851,789,1165,842,3772,64,2172,1375,3236,2955,3859,401,1430,1082,2308,501,90,3212,1849,3401,1131,884,876,1596,3778,619,792,3275,2541,2720,3800,3746,30,3373,3904,1715,1439,2419,682,3238,2123,1470,742,3790,2967,666,1363,2984,923,397,250,75,3563,1878,311,3846,1722,1176,119,3084,1555,1558,2538,3423,226,2411,1508,1036,2482,1795,3377,3283,3184,9,561,185,2317,576,931,536,289,3337,696,3026,658,3533,1652,1888,1886,3575,564,2840,3508,3115,2176,1040,1830,756,1791,1842,1792,433,962,440,3566,578,1997,3253,3398,895,3765,1408,2024,706,3458,2988,3142,2978,1341,2629,2996,2023,3271,1945,189,981,1525,3722,823,3876,1651,598,1939,439,2453,3203,605,2209,600,2612,657,1565,627,3210,2192,618,672,3149,2567,1181,3647,987,3731,2061,2325,1315,679,2026,2745,2563,732,382]
    # L_mm = [1929,2942,299,3494,231,2721,2867,431,3678,1046,3688,996,1025,3069,2252,1339,2828,233,1873,2970,3633,2427,3468,1882,3113,2088,556,1366,3394,171,2447,2836,60,324,892,2488,3918,2731,13,1414,2615,2086,2593,3628,3831,1612,2898,3804,786,2401,1231,3015,3022,3289,761,2497,963,1187,163,3920,3619,2360,3327,3799,1396,2610,3638,3030,1090,1037,2902,1389,3035,3903,2034,3770,1626,2676,1713,2702,741,523,194,2929,2029,1303,340,2843,2130,2490,1971,3697,999,3934,3372,3546,3811,1702,83,2422,1006,1733,1208,3764,2806,2045,2243,1063,3806,2687,2526,3929,3078,828,1982,3817,3947,6,1233,829,1003,1390,15,3189,3509,813,10,764,922,3127,3125,3163,3248,2380,3908,1257,953,2859,1726,364,1977,2046,3379,3358,1416,2349,273,1391,1251,2025,2150,2606,3198,2922,759,207,3265,2362,504,3535,1387,705,3036,3801,888,2152,1092,3444,1611,2311,175,549,3421,2253,3548,3180,2669,1029,535,158,2412,2663,1794,2932,2522,1262,2111,370,2060,2822,1317,3375,1663,3429,2409,3272,2011,1283,2729,3190,485,475,32,1837,3225,511,3946,2366,3053,390,1695,3654,648,3760,3276,537,3513,1100,2288,2341,1007,1880,3540,589,1948,1610,2771,2634,2124,502,7,1876,3569,1597,3626,58,2118,3580,1305,1411,1701,1649,321,1051,458,493,1677,3756,947,2795,95,2135,1379,1563,1920,2501,988,3608,2802,2654,936,989,157,1693,1135,3933,2662,901,3544,3385,1306,544,417,305,2706,1161,1973,937,2249,124,2856,3710,3239,3868,1711,1215,3177,3310,1218,3469,2105,3430,3091,2371,410,2004,3002,1963,673,2433,2116,2403,1287,2507,2682,3593,992,2882,3213,2901,2863,2723,585,3152,3241,3725,2457,1288,419,528,1911,2520,3424,2068,333,2089,35,2093,3018,1752,2670,3298,2492,3738,3306,3342,1266,735,2525,1437,2959,1494,2330,151,3871,1019,1581,3623,3196,3525,3392,24,77,1748,488,463,1280,2478,1517,2434,1635,2890,3388,3466,1583,2188,3062,2149,942,3694,2092,322,3689,1302,2131,3201,1503,1552,2753,3658,1542,156,2500,2725,3928,258,569,2764,1405,259,946,361,2296,3415,3261,3646,517,2262,1227,1680,1732,3214,3925,3118,2391,3422,3818,592,1089,509,2430,150,3393,225,1912,2303,2648,2389,3729,2518,2140,328,3788,505,2164,3858,785,1399,1550,1895,1250,2032,1644,3139,303,3116,3075,100,1202,803,1689,1261,2871,3826,1079,1342,3462,3308,2696,3603,2410,3863,1281,2504,1125,2117,1907,3082,3531,2499,3809,2458,933,2087,2647,292,2943,240,2751,3475,1673,1779,1734,388,665,1329,326,728,1237,1347,1858,1978,2018,1222,2058,1879,3473,1365,290,637,2770,2384,3052,413,3506,181,1676,1541,1246,2941,1667,976,335,116,3743,2390,567,3789,3824,518,3088,2147,522,2580,3943,2143,3490,1567,1995,885,623,2787,3543,103,3865,1892,2437,3031,1646,2950,1980,1627,811,3820,680,707,3617,1228,1431,1392,3927,2017,3680,3521,1551,2755,1249,2883,1113,2208,498,2491,2377,451,1298,342,2407,2889,3319,2009,1192,387,3008,1675,3860,492,3006,925,3564,1866,2712,849,349,432,2420,355,2398,2913,2178,541,2165,850,1970,456,344,1690,3900,76,3361,145,162,2600,2885,956,1265,1371,1378,3262,1213,3910,1496,2521,3752,3648,3135,3162,766,1707,3914,2353,1589,1965,2048,2081,3159,2769,104,2921,2095,2340,1885,2759,3338,2204,2393,952,1087,174,1504,436,2791,2287,861,747,256,3549,2356,1731,3017,711,1841,1793,948,1915,1499,3916,1238,514,2826,574,1739,835,547,1867,1393,2642,800,724,1255,1304,2781,2231,1855,3588,2620,2472,3930,3233,2938,2306,111,3477,2656,1851,2897,3504,1947,3683,3147,3659,2874,2627,2272,1498,2022,1844,3000,1782,2455,3099,1859,55,1395,96,3803,1282,2322,3402,969,2200,704,426,453,3938,1703,2819,1058,2107,2113,2780,841,2987,630,3256,178,1292,3218,2295,3717,645,1124,3713,1016,1670,1889,2965,908,1592,3602,2036,3111,1350,1128,43,3334,444,1725,2904,944,1890,268,550,3172,1060,416,1067,2473,2837,3709,1307,377,1999,2367,1103,2327,2529,2376,1289,590,1144,298,2581,2743,2196,2558,304,1154,3322,1516,2793,1901,3734,3681,1377,3386,2215,2672,2323,325,3758,2597,631,2694,394,409,2329,3037,714,935,3309,1543,379,3048,2936,2264,1374,3094,1918,2530,2700,1647,3816,1136,524,2737,640,1361,3168,1276,3774,1696,499,1193,296,2807,3555,2183,3107,3669,1355,210,1883,3545,3511,542,3749,3252,3762,319,1665,373,2979,2515,1998,2542,483,1650,3830,2644,3175,3895,862,486,202,424,2027,1348,1986,889,2098,2607,526,1719,1205,508,2976,1214,718,480,2841,638,2005,1687,1256,3087,793,1458,3838,1845,2587,1826,3431,276,2100,169,3339,1952,1413,3836,1584,1753,2244,3489,120,2566,3407,3329,2866,351,2544,2971,3221,1388,255,3,2578,1455,1382,1604,45,2297,2181,1164,815,973,860,3270,2924,3505,3875,2536,1184,2616,20,125,3371,3652,1933,2851,3502,3360,1454,3007,744,180,3307,3129,893,674,3819,1772,63,1960,1230,1996,3182,1376,1910,118,1750,2179,1012,3073,2519,2121,3691,3891,2425,1881,3721,3267,2673,2326,1472,223,1817,2375,2960,2628,1591,3849,2374,2256,3330,667,3417,450,568,1914,3032,1783,2906,3769,1033,2300,1009,3766,3021,3629,3347,430,1170,1248,2570,2893,3950,2734,1273,3365,2594,1160,3651,3634,3243,228,2171,1884,3782,3693,603,2170,2033,22,1853,2689,980,3622,965,1254,2189,607,3123,1420,2424,1210,3260,760,1049,3311,3013,1095,3793,3812,2728,1629,2139,2792,1284,2480,2869,3728,183,572,3520,2212,3285,2848,107,3299,1018,3568,164,749,1142,1244,2785,3773,1633,1356,3012,487,796,685,1320,3552,1601,2591,3645,2134,2626,2471,3228,1224,2041,1201,1094,3106,2069,2328,260,2809,3351,3110,89,1827,2865,3148,1123,3016,336,810,3448,3898,3263,750,219,2432,354,2842,2428,1325,2177,3301,2195,3066,1247,1271,1964,196,3719,967,3259,2981,1179,1002,833,1927,2221,2675,1958,1426,2560,1243,1974,743,2651,2162,539,1532,869,3500,161,1432,265,2637,2246,1979,852,274,2546,1370,1493,3712,771,1450,460,1614,204,2757,3498,3100,356,2754,702,3822,2775,3384,2268,2624,801,3692,1639,1924,3862,3590,694,2961,1643,3045,3897,2977,1354,144,3942,2413,1606,3751,3753,2602,858,3571,1445,2090,2012,3510,2234,2605,1574,3814,3674,681,970,2854,1196,3792,882,478,3700,21,2740,281,3244,1519,1865,1520,3695,3842,563,3324,454,628,3559,632,3452,3138,2968,3636,3523,3670,3845,510,1528,2276,1898,582,443,3368,954,1953,3154,1846,1539,610,3419,2877,3414,2796,1069,2283,1728,1942,2059,2157,2077,2273,1608,1603,287,943,2073,2747,2363,1259,3187,3581,2645,1489,3573,2767,2333,3362,1352,2756,2653,2835,380,33,1322,1119,1062,3496,2040,241,315,3832,31,3344,958,3174,1241,2707,1588,1295,808,70,3893,3381,2332,3273,1692,1721,3777,3524,2794,2896,3937,3112,2063,720,2577,3010,626,663,2301,3403,1746,2250,1799,1473,112,16,2730,3141,2450,916,2997,614,2016,826,863,784,2510,1369,2,2236,3484,306,85,2067,1326,3436,1490,2312,1609,2452,253,2744,1679,1294,1903,3400,2711,2493,1358,2383,3056,2112,1531,3144,2185,1057,3333,130,878,1312,3485,2115,3913,3143,1684,3745,3416,297,3823,3425,66,3939,1056,2724,3851,643,1754,2237,2571,2108,2880,2187,3150,1735,2049,2640,1153,3320,1412,2701,310,1333,3096,731,977,1206,1477,2103,27,1175,3867,684,765,565,545,1544,3940,2138,190,3615,1975,2502,1682,1198,2289,3673,782,3464,3295,3025,2459,642,59,716,3349,2985,3587,652,1459,154,38,1671,1502,2887,3952,734,2810,3294,2386,3128,1950,1871,2758,2513,2035,2766,2561,1482,1044,34,1538,1928,1475,548,2167,1554,1148,3346,2175,2378,2631,2075,3098,3827,1335,393,2517,611,3879,1127,2509,3487,1810,551,2632,244,1587,466,2120,2343,1174,1147,2830,2512,68,3880,3807,515,3616,2894,3101,2344,300,1098,2911,2174,2633,122,870,791,3019,941,2223,93,3292,471,1061,3313,2286,1760,1968,368,2298,1383,566,2671,482,3882,2483,2935,2191,2129,1599,2013,3215,3120,1743,2350,3028,924,831,1694,1168,2646,2119,3779,1891,166,418,503,3374,1798,2394,2537,71,700,1992,2278,2486,3585,938,2652,1741,1401,635,991,3905,1495,2201,172,2861,1780,1989,3755,2079,854,3576,1102,2552,2557,1091,3216,557,3887,3151,2811,1331,2451,3207,1869,1664,930,1749,3610,2850,2449,3948,92,1038,3397,1138,3754,442,227,3687,2415,1872,1274,1987,312,1905,1460,2999,378,990,1864,2426,1848,3666,1466,53,633,2462,468,755,729,3176,1121,1718,128,18,2315,588,993,57,3643,3805,3280,848,527,3205,3038,308,412,1149,2994,951,1071,3598,3519,61,1861,2207,275,3465,1932,573,3796,155,847,820,802,691,1054,2596,745,3451,137,3297,1501,382,732,2563,2745,2026,679,1315,2325,2061,3731,987,3647,1181,2567,3149,672,618,2192,3210,627,1565,657,2612,600,2209,605,3203,2453,439,1939,598,1651,3876,823,3722,1525,981,189,1945,3271,2023,2996,2629,1341,2978,3142,2988,3458,706,2024,1408,3765,895,3398,3253,1997,578,3566,440,962,433,1792,1842,1791,756,1830,1040,2176,3115,3508,2840,564,3575,1886,1888,1652,3533,658,3026,696,3337,289,536,931,576,2317,185,561,9,3184,3283,3377,1795,2482,1036,1508,2411,226,3423,2538,1558,1555,3084,119,1176,1722,3846,311,1878,3563,75,250,397,923,2984,1363,666,2967,3790,742,1470,2123,3238,682,2419,1439,1715,3904,3373,30,3746,3800,2720,2541,3275,792,619,3778,1596,876,884,1131,3401,1849,3212,90,501,2308,1082,1430,401,3859,2955,3236,1375,2172,64,3772,842,1165,789,851,472,352,1275,3763,3886,2292,601,1764,1908,757,874,790,3378,3314,1553,251,3085,3850,1425,212,1835,2275,2824,3277,897,3057,3599,1134,282,579,644,859,1510,1630,1852,2235,2277,2484,2742,3220,3321,1870,3219,787,1360,2210,3109,2074,234,1632,2404,512,3447,1727,2556,520,142,1115,3312,3380,3888,530,584,701,2821,484,1364,868,975,2213,3202,3376,1714,84,529,827,1070,2214,121,678,3572,3835,1050,366,235,313,3251,3919,421,2717,1820,1343,1925,3747,1373,2254,2611,2339,583,2534,3522,984,1344,3909,2031,660,664,832,2691,726,843,205,3759,737,56,1105,1190,353,3387,2948,3443,236,2242,3282,3480,1415,123,3516,3740,1590,2516,2705,957,712,1877,1474,2218,3364,346,3600,3748,836,912,1359,307,2169,3744,2338,3730,198,1017,1797,1598,193,1564,1923,2986,2655,497,1096,1625,2078,36,1946,3001,2939,3632,147,2776,986,1367,2099,3733,2668,67,191,1940,3570,46,597,384,2441,804,3408,2919,719,1077,3736,2460,1513,3413,2470,2082,427,1821,2110,481,374,1330,1569,3703,3034,1372,3935,2847,1930,3404]

    nr_many_ratings = 0
    for user in X:
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            nr_many_ratings += 1
    print(nr_many_ratings)
    nr_remove = total_added / nr_many_ratings

    for user_index, user in enumerate(X):
        print("user: ", user_index)
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            index_m = 0
            index_f = 0
            rem = 0
            if T[user_index] == 1:
                safety_counter = 0
                # We note that if we add safety_counter < 1000 in the while we have a higher accuracy than if we keep it in the if
                while (rem < nr_remove) and safety_counter < 1000:
                    if index_f >= len(L_f):  # and safety_counter < 1000:
                        safety_counter = 1000
                        continue

                    if removal_mode == "random":
                        to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0],
                                                                  size=(int(nr_remove),),
                                                                  replace=False)  # , replace=False)
                    if removal_mode == "strategic":
                        to_be_removed_indecies = L_f[index_f]
                    index_f += 1

                    if X_obf[user_index, int(to_be_removed_indecies) - 1] != 0:
                        X_obf[user_index, int(to_be_removed_indecies) - 1] = 0
                        rem += 1
                    safety_counter += 1

            elif T[user_index] == 0:

                while (rem < nr_remove) and safety_counter < 1000:
                    if index_m >= len(L_m):  # and safety_counter < 1000:
                        safety_counter = 1000
                        continue

                    if removal_mode == "random":
                        to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0],
                                                                  size=(int(nr_remove),),
                                                                  replace=False)  # , replace=False)
                    # X_obf[user_index, to_be_removed_indecies] = 0

                    if removal_mode == "strategic":
                        to_be_removed_indecies = L_m[index_m]
                    index_m += 1

                    if X_obf[user_index, int(to_be_removed_indecies) - 1] != 0:
                        X_obf[user_index, int(to_be_removed_indecies) - 1] = 0
                        rem += 1
                    safety_counter += 1

    # Now remove ratings from users that have more than 200 ratings equally:
    """nr_many_ratings = 0
    for user in X:
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            nr_many_ratings += 1
    nr_remove = total_added / nr_many_ratings

    for user_index, user in enumerate(X):
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0], size=(int(nr_remove),),
                                                      replace=False)
            X_obf[user_index, to_be_removed_indecies] = 0 """

    # finally, shuffle the user vectors:
    # np.random.shuffle(X_obf)
    # output the data in a file:
    output_file = ""
    if dataset == 'ML':  # _AllIndicItems || Top100IndicativeItems_
        output_file = "ml-1m/BlurSome/Top-50/"  # ml-1m/BlurSome/Top-75_HighRatings/
        with open(
                output_file + "TrainingSet_blurSome_AverageML1M_obfuscated_Top50IndicativeItems_" + sample_mode + "_" + str(
                        p) + "_" + str(notice_factor) + ".dat",  # "_" + str(removal_mode) +
                'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(
                            str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                                int(np.round(rating))) + "::000000000\n")


    elif dataset == 'Fx':
        output_file = "Flixster/BlurSome/Top75/"
        with open(output_file + "All_FX_Top75_Predicted_blurSome_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(
                notice_factor) + "_" + str(removal_mode) + ".dat",  # + "_" + str(removal_mode) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    else:
        with open("libimseti/LST_blurmepp_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(notice_factor) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    return X_obf


def BlurMeimputation():
    top = -1
    # sample_mode can be either totally random selection of items or random from list_F or list_M or greedy from list_F and list_M
    sample_mode = \
    list(['totally_random', 'popular_items', 'critical_items', 'highest', 'random', 'sampled', 'greedy', 'imputation'])[
        7]
    removal_mode = list(['random', 'strategic'])[1]
    # id_index, index_id = MD.load_movie_id_index_dict()
    notice_factor = 2
    p = 0.1  # 0.3
    dataset = ['ML', 'Fx', 'Li'][1]
    if dataset == 'ML':
        """X = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
        #X_features = MD.load_user_item_matrix_1m_Features()
        T = MD.load_gender_vector_1m()  # max_user=max_user)
        X_filled = MD.load_user_item_matrix_1m_complet()"""
        X = MD.load_user_item_matrix_100k()
        T = MD.load_gender_vector_100k()
        X_filled = MD.load_user_item_matrix_100k_Complet()
    elif dataset == 'Fx':
        import FlixsterData as FD
        # X, T = FD.load_flixster_data_subset()
        """X = FD.load_user_item_matrix_FD()
        X1, T = FD.load_flixster_data()"""
        X, T, _ = FD.load_flixster_data_subset()
        X_filled = FD.load_user_item_matrix_FD_All()
    else:
        import LibimSeTiData as LD
        X, T, _ = LD.load_libimseti_data_subset()
    # X = Utils.normalize(X)
    avg_ratings = np.zeros(shape=X.shape[1])
    initial_count = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = []
        for rating in X[:, item_id]:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[item_id] = 0
        else:
            avg_ratings[item_id] = np.average(ratings)
        initial_count[item_id] = len(ratings)
    max_count = initial_count * notice_factor
    # 1: get the set of most correlated movies, L_f and L_m:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    cv = StratifiedKFold(n_splits=10)
    coefs = []
    avg_coefs = np.zeros(shape=(len(X[1]),))

    random_state = np.random.RandomState(0)
    for train, test in cv.split(X, T):
        x, t = X[train], T[train]
        model = LogisticRegression(penalty='l2', random_state=random_state)
        model.fit(x, t)
        # rank the coefs:
        ranks = ss.rankdata(model.coef_[0])
        coefs.append(ranks)
        # print(len(model.coef_[0]),len(X_train[0]))
        avg_coefs += model.coef_[0]

    coefs = np.average(coefs, axis=0)
    coefs = [[coefs[i], i + 1, avg_coefs[i]] for i in range(len(coefs))]
    coefs = np.asarray(list(sorted(coefs)))
    if top == -1:
        values = coefs[:, 2]
        index_zero = np.where(values == np.min(np.abs(values)))
        top_male = index_zero[0][0]
        top_female = index_zero[0][-1]
        L_m = coefs[:top_male, 1]
        R_m = 3952 - coefs[:top_male, 0]
        C_m = np.abs(coefs[:top_male, 2])
        L_f = coefs[coefs.shape[0] - top_female:, 1]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0] - top_female:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0] - top_female:, 2]
        C_f = list(reversed(np.abs(C_f)))

    else:
        L_m = coefs[:top, 1]
        R_m = 3952 - coefs[:top, 0]
        C_m = np.abs(coefs[:top, 2])
        L_f = coefs[coefs.shape[0] - top:, 1]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0] - top:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0] - top:, 2]
        C_f = list(reversed(np.abs(C_f)))

    # create list of all items for totally_random strategy
    L = list(range(1, 3953))

    # list of popular items in Movielens 1M == 590
    L_pop = [1, 2, 6, 10, 11, 16, 17, 21, 24, 25, 32, 34, 36, 39, 45, 47, 50, 62, 70, 95, 104, 110, 111, 112, 141, 150,
             151, 153, 160, 161, 163, 165, 173, 185, 196, 198, 208, 223, 231, 235, 246, 253, 260, 265, 266, 288, 292,
             293, 296, 300, 316, 317, 318, 329, 337, 339, 344, 349, 350, 353, 356, 357, 364, 367, 368, 377, 380, 434,
             435, 440, 441, 442, 454, 457, 466, 471, 474, 480, 497, 500, 508, 509, 527, 529, 539, 541, 543, 551, 552,
             553, 555, 586, 587, 588, 589, 590, 592, 593, 594, 595, 597, 608, 610, 648, 653, 673, 708, 733, 736, 745,
             750, 778, 780, 785, 788, 800, 832, 852, 858, 866, 898, 899, 902, 903, 904, 908, 910, 912, 913, 914, 919,
             920, 923, 924, 953, 968, 969, 1019, 1022, 1028, 1029, 1032, 1035, 1036, 1037, 1042, 1060, 1073, 1077, 1079,
             1080, 1084, 1088, 1089, 1090, 1092, 1093, 1094, 1095, 1097, 1101, 1127, 1129, 1136, 1148, 1172, 1179, 1183,
             1186, 1188, 1193, 1196, 1197, 1198, 1199, 1200, 1201, 1203, 1204, 1206, 1207, 1208, 1210, 1213, 1214, 1215,
             1219, 1220, 1221, 1222, 1225, 1228, 1230, 1231, 1233, 1234, 1235, 1240, 1242, 1244, 1246, 1247, 1249, 1250,
             1252, 1253, 1256, 1258, 1259, 1261, 1262, 1263, 1265, 1266, 1267, 1269, 1270, 1271, 1272, 1275, 1276, 1278,
             1282, 1284, 1285, 1287, 1288, 1291, 1292, 1293, 1294, 1296, 1299, 1302, 1304, 1307, 1320, 1321, 1333, 1339,
             1343, 1345, 1356, 1357, 1358, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1380, 1387, 1391, 1393,
             1394,
             1396, 1407, 1408, 1466, 1479, 1485, 1500, 1513, 1517, 1527, 1544, 1552, 1562, 1569, 1573, 1580, 1584, 1586,
             1587, 1597, 1608, 1610, 1617, 1625, 1639, 1641, 1645, 1653, 1663, 1673, 1674, 1676, 1682, 1690, 1704, 1711,
             1721, 1722, 1729, 1732, 1747, 1748, 1777, 1779, 1784, 1805, 1831, 1876, 1883, 1885, 1909, 1912, 1917, 1918,
             1921, 1923, 1947, 1952, 1953, 1954, 1957, 1958, 1959, 1960, 1961, 1962, 1965, 1967, 1968,
             1994, 1997, 2000, 2001, 2002, 2003, 2005, 2006, 2011, 2012, 2013, 2018, 2019, 2020, 2021, 2023, 2028, 2046,
             2054, 2058, 2064, 2067, 2076, 2078, 2080, 2081, 2085, 2087, 2094, 2100, 2105, 2108, 2109, 2115, 2124, 2133,
             2140, 2144, 2145, 2150, 2160, 2161, 2167, 2174, 2193, 2194, 2243, 2248, 2268, 2273, 2278, 2288, 2289, 2291,
             2294, 2300, 2302, 2321, 2324, 2329, 2333, 2336, 2352, 2353, 2355, 2359, 2366, 2369, 2371, 2384, 2391, 2393,
             2394, 2395, 2396, 2405, 2406, 2407, 2424, 2427, 2433, 2455, 2470, 2478, 2490, 2501, 2502, 2527, 2528, 2529,
             2539, 2541, 2542, 2571, 2572, 2580, 2581, 2598, 2599, 2605, 2616, 2617, 2628, 2640, 2641, 2657, 2662, 2664,
             2671, 2683, 2686, 2688, 2692, 2694, 2699, 2700, 2701, 2706, 2707, 2710, 2712, 2716, 2717, 2722, 2723, 2724,
             2746, 2761, 2762, 2763, 2770, 2791, 2792, 2795, 2797, 2804, 2826, 2858, 2871, 2872, 2881, 2890, 2908, 2915,
             2916, 2918, 2944, 2947, 2948, 2949, 2951, 2959, 2968, 2976, 2985, 2987, 2997, 3005,
             3006, 3020, 3033, 3037, 3039, 3044, 3052, 3060, 3070, 3072, 3081, 3082, 3087, 3095, 3098, 3100, 3101, 3104,
             3105, 3107, 3108, 3113, 3114, 3147, 3148, 3160, 3174, 3175, 3176, 3210, 3247, 3252, 3253, 3255, 3256, 3257,
             3263, 3273, 3298, 3300, 3301, 3354, 3360, 3361, 3362, 3363, 3386, 3396, 3408, 3418, 3421, 3424, 3435, 3438,
             3448, 3450, 3471, 3476, 3479, 3481, 3489, 3499, 3504, 3510, 3524, 3526, 3527, 3535, 3543, 3552, 3555, 3578,
             3591, 3608, 3614, 3623, 3624, 3638, 3671, 3683, 3685, 3697, 3698, 3699, 3702, 3703, 3704, 3717, 3740, 3745,
             3751, 3752, 3753, 3755, 3763, 3785, 3793, 3809, 3863, 3868, 3893, 3897, 3911, 3948]

    # list of critical items same way as Shuffle-NNN
    L_critical = [1, 2, 3, 6, 7, 10, 11, 12, 19, 20, 21, 25, 32, 33, 34, 36, 37, 39, 44, 45, 47, 48, 50, 52, 56, 62, 65,
                  66, 67, 70, 77, 83, 88, 95, 96, 100, 103, 105, 107, 108, 110, 111, 112, 114, 116, 117, 119, 127, 130,
                  132, 133, 134, 136, 138, 139, 142, 144, 150, 153, 157, 158, 159, 162, 165, 168, 171, 172, 173, 176,
                  177, 178, 182, 186, 194, 196, 204, 205, 208, 210, 211, 214, 216, 217, 223, 224, 225, 226, 228, 229,
                  230, 231, 232, 235, 238, 240, 249, 251, 253, 256, 258, 260, 261, 264, 266, 269, 270, 275, 276, 277,
                  288, 291, 292, 293, 295, 296, 300, 306, 307, 308, 310, 311, 312, 313, 315, 316, 318, 324, 326, 327,
                  328, 329, 333, 338, 339, 344, 345, 349, 350, 353, 355, 356, 357, 358, 359, 361, 364, 365, 366, 367,
                  368, 370, 373, 374, 377, 379, 380, 381, 383, 384, 387, 396, 398, 409, 410, 411, 412, 413, 421, 427,
                  433, 434, 440, 442, 446, 448, 452, 454, 456, 457, 458, 460, 465, 467, 470, 474, 477, 480, 481, 484,
                  486, 487, 493, 494, 496, 497, 500, 502, 505, 507, 511, 514, 515, 517, 519, 520, 521, 522, 527, 529,
                  530, 532, 534, 539, 541, 552, 557, 558, 559, 567, 572, 578, 579, 580, 583, 584, 585, 586, 587, 588,
                  589, 590, 592, 593, 594, 595, 596, 597, 599, 600, 601, 602, 603, 605, 607, 608, 611, 614, 616, 623,
                  626, 628, 634, 635, 637, 643, 644, 648, 650, 651, 652, 653, 657, 658, 660, 665, 666, 668, 669, 670,
                  672, 673, 674, 679, 682, 687, 690, 695, 696, 698, 701, 703, 705, 706, 708, 712, 717, 720, 729, 730,
                  733, 734, 736, 741, 744, 745, 749, 750,
                  756, 758, 763, 764, 769, 774, 776, 778, 780, 782, 783, 784, 785, 786, 787, 792, 793, 796, 799, 803,
                  805, 808, 811, 814, 815, 818, 820, 827, 828, 832, 837, 838, 842, 846, 847, 852, 853, 858, 859, 860,
                  861, 878, 880, 882, 884, 887, 891, 893, 895, 896, 899, 900, 903, 904, 905, 907, 908, 911, 912, 913,
                  914, 915, 916, 919, 920, 921, 922, 923, 924, 925, 930, 931, 938, 939, 941, 943, 945, 946, 953, 955,
                  959, 960, 962, 964, 967, 969, 972, 975,
                  976, 977, 981, 985, 988, 993, 994, 1004, 1005, 1011, 1013, 1018, 1022, 1023, 1025, 1028, 1029, 1030,
                  1031, 1032, 1033, 1035, 1036, 1037, 1038, 1040, 1044, 1056, 1057, 1064, 1070, 1071, 1073, 1076, 1079,
                  1080, 1084, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1097, 1101, 1102, 1103, 1105, 1107, 1115,
                  1116, 1118, 1127, 1128, 1129, 1130, 1131, 1132, 1135, 1136, 1139, 1142, 1145, 1148, 1149, 1150, 1152,
                  1160, 1165, 1172, 1178, 1181, 1183, 1186, 1189, 1190, 1193, 1196, 1197, 1198, 1199, 1200, 1201, 1204,
                  1206, 1207, 1208, 1210, 1212, 1213, 1214, 1215, 1218, 1219, 1220, 1221, 1222, 1223, 1225, 1228, 1230,
                  1231, 1234, 1235, 1240, 1242, 1244, 1246, 1247, 1248, 1249, 1250, 1252, 1253, 1254, 1257, 1258, 1259,
                  1261, 1262, 1263, 1265, 1266, 1267, 1270, 1271, 1272, 1274, 1275, 1276, 1277, 1278, 1279, 1281, 1282,
                  1283, 1284, 1285, 1286, 1288, 1289, 1291, 1293, 1294, 1297, 1300, 1301, 1302, 1304, 1307, 1320, 1321,
                  1333, 1334, 1339, 1340, 1343, 1345, 1346, 1347, 1350, 1355, 1356, 1357, 1358, 1359, 1360, 1363, 1364,
                  1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1383, 1385, 1386, 1387, 1388,
                  1389, 1391, 1393, 1394, 1396, 1407, 1415, 1416, 1419, 1428, 1429, 1432, 1433, 1434, 1442, 1465, 1466,
                  1470, 1479, 1484, 1485, 1486, 1495, 1498, 1500, 1501, 1503, 1507, 1508, 1510, 1514, 1515, 1517, 1519,
                  1522, 1527, 1528, 1534, 1542, 1544, 1548, 1551, 1552, 1553, 1555, 1558, 1561, 1562, 1566, 1569, 1573,
                  1575, 1580, 1584, 1587, 1588, 1590, 1591, 1595, 1597, 1599, 1608, 1610, 1614, 1616, 1617, 1622, 1625,
                  1626, 1630, 1632, 1633, 1640, 1641, 1644, 1645, 1652, 1653, 1658, 1659, 1663, 1664, 1666, 1668, 1669,
                  1671, 1672, 1673, 1674, 1675, 1676, 1680, 1681, 1682, 1686, 1687, 1688, 1690, 1694, 1695, 1701, 1704,
                  1709, 1717, 1721, 1722, 1726, 1727, 1729, 1732,
                  1735, 1739, 1741, 1747, 1748, 1749, 1752, 1762, 1764, 1769, 1770, 1772, 1773, 1777, 1780, 1782, 1784,
                  1787, 1788, 1791, 1792, 1795, 1796, 1805, 1806, 1811, 1816, 1820, 1822, 1824, 1826, 1831, 1835, 1841,
                  1842, 1850, 1851, 1852, 1854, 1856, 1857, 1862, 1865, 1866, 1872, 1876, 1882, 1883, 1886, 1887, 1891,
                  1896, 1897, 1901, 1902, 1903, 1905, 1907, 1908, 1909, 1914, 1917, 1918, 1921, 1923, 1930, 1936, 1938,
                  1941, 1943, 1946, 1947, 1949,
                  1952, 1953, 1954, 1955, 1958, 1959, 1961, 1962, 1964, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974,
                  1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1990, 1991, 1992, 1993, 1994,
                  1995, 1996, 1997, 2000, 2001, 2002, 2003, 2005, 2006, 2007, 2008, 2010, 2011, 2012, 2018, 2020, 2021,
                  2023, 2024, 2026, 2028, 2035, 2037, 2038, 2040, 2042, 2044, 2046, 2051, 2054, 2055, 2058, 2063, 2064,
                  2065, 2071, 2072, 2075,
                  2078, 2080, 2081, 2082, 2084, 2085, 2087, 2089, 2090, 2092, 2094, 2096, 2100, 2101, 2105, 2107, 2108,
                  2115, 2118, 2119, 2121, 2122, 2123, 2124, 2125, 2127, 2129, 2134, 2137, 2138, 2140, 2142, 2143, 2144,
                  2145, 2148, 2149, 2150, 2151, 2157, 2161, 2167, 2169, 2173, 2174, 2179, 2183, 2186, 2188, 2191, 2192,
                  2193, 2194, 2196, 2202, 2203, 2205, 2206, 2207, 2208, 2209, 2210, 2214, 2215, 2219, 2221, 2223, 2232,
                  2233, 2234, 2235, 2236, 2241, 2243, 2244, 2246, 2248, 2249, 2254, 2260, 2262, 2266, 2268, 2273, 2275,
                  2277, 2278, 2279, 2280, 2281, 2282, 2286, 2288, 2291, 2292, 2294, 2295, 2296, 2297, 2302, 2306, 2309,
                  2312, 2315, 2316, 2318, 2320, 2321, 2326, 2332, 2335, 2338, 2339, 2342, 2344, 2345, 2352, 2353, 2354,
                  2355, 2360, 2363, 2366, 2371, 2372, 2373, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2389,
                  2391, 2393, 2394, 2395, 2396, 2398, 2399, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2409, 2410, 2411,
                  2412, 2416, 2417, 2418, 2420, 2421, 2422, 2423, 2424, 2431, 2432, 2444, 2446, 2447, 2452, 2455, 2456,
                  2459, 2468, 2469, 2470, 2471, 2475, 2476, 2478, 2481, 2484, 2486, 2491, 2493, 2495, 2496, 2502, 2503,
                  2504, 2511, 2513, 2515, 2516, 2517, 2519, 2521, 2525, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2539,
                  2545, 2548, 2553, 2554, 2557, 2563, 2565, 2566, 2571, 2573, 2576, 2578, 2581, 2584, 2586, 2590, 2599,
                  2602, 2608, 2610, 2611, 2612, 2613, 2616, 2617, 2620, 2622, 2624, 2625, 2626, 2628, 2634, 2637, 2638,
                  2640, 2641, 2642, 2643, 2644, 2646, 2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2657, 2661,
                  2662, 2663, 2664, 2671, 2673, 2679, 2681, 2682, 2683, 2685, 2687, 2688, 2693, 2694, 2696, 2699,
                  2700, 2702, 2703, 2704, 2705, 2706, 2707, 2710, 2712, 2715, 2716, 2717, 2729, 2731, 2738, 2740, 2742,
                  2744, 2746, 2749, 2750, 2754, 2755, 2758, 2760, 2761, 2762, 2763, 2769, 2770, 2771, 2773, 2778, 2779,
                  2782, 2788, 2789, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2804, 2805, 2806, 2808, 2811,
                  2812, 2815, 2816, 2817, 2818, 2821, 2823, 2825, 2828, 2830, 2831, 2840, 2843, 2844, 2848, 2850,
                  2852, 2854, 2855, 2857, 2858, 2862, 2863, 2865, 2867, 2869, 2875, 2878, 2879, 2880, 2890, 2892, 2893,
                  2895, 2899, 2902, 2903, 2906, 2909, 2915, 2916, 2918, 2921, 2922, 2930, 2933, 2934, 2935, 2936, 2937,
                  2940, 2941, 2944, 2946, 2947, 2948, 2949, 2950, 2951, 2953, 2955, 2959, 2968, 2971, 2972, 2976, 2977,
                  2984, 2985, 2986, 2987, 2989, 2990, 2991, 2993, 2996, 2997, 3003, 3004, 3005, 3012, 3014, 3018,
                  3019, 3021, 3026, 3027, 3032, 3033, 3034, 3039, 3042, 3043, 3051, 3053, 3057, 3058, 3062, 3063, 3066,
                  3068, 3069, 3081, 3082, 3087, 3088, 3097, 3098, 3099, 3100, 3104, 3107, 3109, 3114, 3119, 3120, 3122,
                  3123, 3126, 3129, 3130, 3136, 3139, 3140, 3146, 3147, 3154, 3164, 3168, 3175, 3178, 3179, 3185, 3187,
                  3204, 3210, 3212, 3215, 3216, 3220, 3222, 3223, 3228, 3229, 3233, 3236, 3238, 3239, 3240, 3245,
                  3246, 3247, 3253, 3255, 3256, 3260, 3263, 3264, 3265, 3275, 3277, 3281, 3282, 3284, 3287, 3288, 3290,
                  3292, 3293, 3295, 3298, 3299, 3300, 3304, 3305, 3307, 3312, 3315, 3318, 3321, 3322, 3323, 3324, 3328,
                  3330, 3334, 3335, 3336, 3337, 3345, 3346, 3353, 3358, 3359, 3360, 3361, 3363, 3365, 3372, 3377, 3379,
                  3380, 3382, 3384, 3386, 3388, 3392, 3395, 3396, 3397, 3398, 3402, 3404, 3406, 3407, 3408, 3412, 3413,
                  3415, 3416, 3418, 3421, 3424, 3430, 3431, 3432, 3433, 3434, 3435, 3437, 3438, 3439, 3440, 3441, 3443,
                  3444, 3445, 3448, 3450, 3453, 3456, 3458, 3462, 3463, 3464, 3470, 3471, 3472, 3479, 3481, 3482, 3483,
                  3485, 3486, 3488, 3489, 3490, 3492, 3493, 3494, 3499, 3501, 3503, 3508, 3513, 3517, 3518, 3522, 3524,
                  3525, 3526, 3527, 3532, 3534, 3542, 3547, 3550, 3552, 3553, 3557, 3568, 3569, 3573, 3574, 3575, 3578,
                  3579, 3591, 3600, 3601, 3602, 3603, 3605, 3608, 3610, 3613, 3617, 3623, 3628, 3631, 3633, 3635, 3636,
                  3638, 3639, 3641, 3642, 3643, 3644, 3648, 3651, 3654, 3656, 3657, 3658, 3659, 3661, 3662, 3663, 3664,
                  3665, 3666, 3668, 3669, 3671, 3677, 3678, 3680, 3681, 3687, 3688, 3689, 3690, 3692, 3694, 3695, 3696,
                  3697, 3698, 3699, 3700, 3702, 3703, 3704, 3709, 3712, 3718, 3730, 3734, 3738, 3742, 3744, 3746, 3748,
                  3753, 3754, 3755, 3757, 3759, 3765, 3766, 3767, 3768, 3771, 3772,
                  3773, 3774, 3775, 3776, 3778, 3781, 3782, 3787, 3792, 3793, 3798, 3800, 3803, 3805, 3807, 3812, 3814,
                  3819, 3821, 3827, 3830, 3832, 3835, 3836, 3837, 3838, 3839, 3848, 3852, 3857, 3866, 3868, 3869, 3874,
                  3875, 3876, 3878, 3887, 3888, 3889, 3893, 3896, 3897, 3899, 3900, 3906, 3911, 3916, 3917, 3918, 3919,
                  3921, 3922, 3924, 3926, 3927, 3930, 3931, 3938, 3939, 3940, 3941, 3942, 3943, 3944, 3945, 3948]

    # list of highly rated items
    L_high = [989, 3881, 1830, 3382, 787, 3280, 3607, 3233, 3172, 3656, 3245, 53, 2503, 2905, 2019, 318, 858, 745, 50,
              527, 1148, 2309, 1795, 2480, 439, 557, 3517, 3888, 578, 922, 1198, 904, 1178, 260, 1212, 750, 3338, 720,
              1207, 3435, 912, 670, 2762, 3030, 668, 1204, 2930,
              913, 1193, 923, 3307, 1250, 908, 1262, 3022, 1223, 1221, 3089, 1423, 593, 326, 3134, 1252, 2028, 3429,
              1136, 128, 3410, 1267, 2324, 3470, 1131, 1147, 2731, 1234, 2858, 2571, 1284, 2360, 2186, 1197, 1233, 1260,
              898, 910, 2839, 2931, 953, 1203, 930, 1196, 1254,
              2937, 1172, 1224, 2357, 3091, 899, 905, 3469, 296, 1213, 3679, 541, 903, 2203, 1945, 1217, 1272, 1132,
              363, 926, 608, 3634, 1276, 1189, 1225, 969, 1278, 1117, 598, 1002, 214, 3730, 951, 1299, 919, 1247, 678,
              1208, 954, 3468, 950, 2804, 3462, 110, 2859, 1949, 3801,
              1237, 3196, 2925, 306, 556, 3077, 2329, 3897, 1104, 2692, 2609, 1617, 3114, 1219, 3683, 916, 1248, 1304,
              1361, 1256, 1358, 2351, 928, 3038, 942, 1664, 3789, 1218, 1927, 1242, 2935, 2920, 3629, 1236, 246, 1228,
              649, 2208, 111, 911, 2197, 1288, 1283, 1704, 669, 1280,
              955, 2726, 1269, 3222, 1066, 1235, 1900, 2330, 1214, 3147, 914, 1240, 2066, 3035, 3000, 945, 1449, 1,
              1300, 2300, 2493, 1230, 1293, 1253, 3090, 2501, 3265, 2788, 1537, 1089, 3095, 1201, 1291, 1950, 1303,
              2396, 909, 1200, 2997, 1780, 1294, 1939, 602, 1036, 2918, 3949, 1952,
              1211, 1222, 1287, 1263, 47, 947, 293, 3578, 1251, 1258, 457, 3811, 759, 933, 307, 1953, 116, 1199, 1259,
              906, 1084, 1944, 1111, 994, 58, 2936, 1266, 1090, 3088, 1387, 741, 2624, 2607, 2612, 356, 599, 3083, 3365,
              3192, 446, 800, 2010, 3334, 1244, 2927, 3359, 1231, 1086,
              3741, 1206, 2959, 936, 232, 965, 2132, 2064, 3198, 150, 1307, 3911, 924, 1545, 1046, 2067, 162, 29, 3006,
              589, 1947, 213, 2682, 2677, 2966, 28, 1264, 121, 1961, 1226, 3421, 1446, 1610, 3224, 1963, 2973, 581, 971,
              2761, 3671, 3849, 615, 1719, 3037, 3508, 3281, 2819,
              3341, 3097, 1041, 1942, 3507, 1281, 2917, 1931, 1362, 2336, 3783, 1192, 1935, 17, 3363, 3819, 1209, 1210,
              3814, 3808, 3467, 1394, 3504, 2728, 1080, 2248, 2972, 3471, 2908, 3007, 1292, 2313, 1934, 2206, 2940,
              2194, 3498, 1797, 1246, 1245, 1249, 2944, 1175, 966, 3164,
              139, 624, 2198, 3353, 1741, 3609, 3305, 1316, 3057, 3277, 878, 134, 853, 3065, 1470, 1434, 130, 632, 1420,
              746, 456, 2670, 3737, 584, 2999, 1076, 3126, 717, 2358, 1915, 1832, 3092, 3522, 2444, 3530, 1842, 2494,
              701, 3229, 1238, 3232, 1851, 1901, 2575, 774, 3601, 2811,
              2438, 398, 792, 1871, 396, 3647, 1827, 2909, 1139, 758, 497, 920, 1296, 1674, 2951, 1177, 3929, 1348,
              1270, 1956, 1099, 2983, 1023, 3075, 1243, 943, 3735, 3347, 1759, 1077, 2732, 529, 949, 1096, 1960, 927,
              1301, 1946, 2648, 3739, 940, 1289, 3350, 3152, 3062, 2020, 1079, 2791, 2947, 3742, 3362, 324, 1273, 308,
              1097, 915, 3654, 1419, 2289, 3475, 2686, 2398, 778, 3182, 3728, 2970, 41, 3306, 1411, 1185, 36, 2932,
              3096, 1788, 1997, 1571, 2565, 1265, 3201, 932, 3148]

    # Now, where we have the two lists, we can start obfuscating the data:
    # X = MD.load_user_item_matrix_1m()
    # np.random.shuffle(X)
    # print(X.shape)
    X_obf = np.copy(X)
    total_added = 0

    for index, user in enumerate(X):

        print(index)
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index_m = 0
        greedy_index_f = 0
        # print(k)
        added = 0
        if T[index] == 1:
            safety_counter = 0
            while added < k and safety_counter < 100:
                if greedy_index_m >= len(L_m):
                    safety_counter = 1000
                    continue
                if sample_mode == 'greedy':
                    movie_id = L_m[greedy_index_m]
                if sample_mode == 'random':
                    movie_id = L_m[np.random.randint(0, len(L_m))]
                if sample_mode == 'totally_random':
                    movie_id = L[np.random.randint(0, len(L))]
                if sample_mode == 'imputation':
                    movie_id = L_m[greedy_index_m]
                if sample_mode == 'popular_items':
                    movie_id = L_pop[np.random.randint(0, len(L_pop))]
                if sample_mode == 'critical_items':
                    movie_id = L_critical[np.random.randint(0, len(L_critical))]
                if sample_mode == 'highest':
                    movie_id = L_high[np.random.randint(0, len(L_high))]
                greedy_index_m += 1
                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id) - 1]])
                if rating_count > max_count[int(movie_id) - 1]:
                    continue

                if X_obf[index, int(movie_id) - 1] == 0:
                    X_obf[index, int(movie_id) - 1] = X_filled[index, int(movie_id) - 1]
                    added += 1
                safety_counter += 1
        elif T[index] == 0:
            safety_counter = 0
            while added < k and safety_counter < 1000:
                if greedy_index_f >= len(L_f):
                    safety_counter = 1000
                    continue
                if sample_mode == 'greedy':
                    movie_id = L_f[greedy_index_f]
                if sample_mode == 'random':
                    movie_id = L_f[np.random.randint(0, len(L_f))]
                if sample_mode == 'totally_random':
                    movie_id = L[np.random.randint(0, len(L))]
                if sample_mode == 'imputation':
                    movie_id = L_f[greedy_index_f]
                if sample_mode == 'popular_items':
                    movie_id = L_pop[np.random.randint(0, len(L_pop))]
                if sample_mode == 'critical_items':
                    movie_id = L_critical[np.random.randint(0, len(L_critical))]
                if sample_mode == 'highest':
                    movie_id = L_high[np.random.randint(0, len(L_high))]
                greedy_index_f += 1
                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id) - 1]])
                if rating_count > max_count[int(movie_id) - 1]:
                    continue

                if X_obf[index, int(movie_id) - 1] == 0:
                    X_obf[index, int(movie_id) - 1] = X_filled[index, int(movie_id) - 1]
                    added += 1
                safety_counter += 1
        total_added += added

    # Now remove ratings from users that have more than 200 ratings equally:
    # The removal can be either "Random" so just uncomment the commented code
    # Removal strategy == Random
    """nr_many_ratings = 0
    for user in X:
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            nr_many_ratings += 1
    nr_remove = total_added/nr_many_ratings

    for user_index, user in enumerate(X):
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:,0], size=(int(nr_remove),), replace=False)
            X_obf[user_index, to_be_removed_indecies] = 0
    """
    # Removal strategy == Greedy
    nr_many_ratings = 0
    for user in X:
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            nr_many_ratings += 1
    print(nr_many_ratings)
    nr_remove = total_added / nr_many_ratings

    for user_index, user in enumerate(X):
        print("user: ", user_index)
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            index_m = 0
            index_f = 0
            rem = 0
            if T[user_index] == 1:
                safety_counter = 0
                # We note that if we add safety_counter < 1000 in the while we have a higher accuracy than if we keep it in the if
                while (rem < nr_remove) and safety_counter < 1000:
                    if index_f >= len(L_f):
                        safety_counter = 1000
                        continue

                    if removal_mode == "random":
                        to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0],
                                                                  size=(int(nr_remove),),
                                                                  replace=False)  # , replace=False)
                    if removal_mode == "strategic":
                        to_be_removed_indecies = L_f[index_f]
                    index_f += 1

                    if X_obf[user_index, int(to_be_removed_indecies) - 1] != 0:
                        X_obf[user_index, int(to_be_removed_indecies) - 1] = 0
                        rem += 1
                    safety_counter += 1

            elif T[user_index] == 0:

                while (rem < nr_remove) and safety_counter < 1000:
                    if index_m >= len(L_m):  # and safety_counter < 1000:
                        safety_counter = 1000
                        continue

                    if removal_mode == "random":
                        to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0],
                                                                  size=(int(nr_remove),),
                                                                  replace=False)  # , replace=False)
                    # X_obf[user_index, to_be_removed_indecies] = 0

                    if removal_mode == "strategic":
                        to_be_removed_indecies = L_m[index_m]
                    index_m += 1

                    if X_obf[user_index, int(to_be_removed_indecies) - 1] != 0:
                        X_obf[user_index, int(to_be_removed_indecies) - 1] = 0
                        rem += 1
                    safety_counter += 1

    # finally, shuffle the user vectors:
    # np.random.shuffle(X_obf)
    # output the data in a file:
    output_file = ""
    if dataset == 'ML':
        output_file = "ml-100k/BlurMore/Imput_Matrix_AllUsers/"
        with open(output_file + "All_blurmepp_Imputation_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(
                notice_factor) + "_" + str(removal_mode) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(
                            str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                                int(np.round(rating))) + "::000000000\n")

    elif dataset == 'Fx':
        import FlixsterData as FD
        output_file = "Flixster/BlurMore/Imput_Matrix_AllUsers/"
        user_id2index, user_index2id = FD.load_user_id_index_dict()
        movie_id2index, movie_index2id = FD.load_movie_id_index_dict()

        with open(output_file + "All_FX_blurmepp_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(
                notice_factor) + "_" + str(removal_mode) + ".dat",  # + "_" + str(removal_mode) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(user_index2id[index_user]) + "::" + str(movie_index2id[index_movie]) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    else:
        with open("libimseti/LST_blurmepp_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(notice_factor) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    return X_obf



# blurMe_100k()
# blurMe_1m()
# blurMePP()
blurSome()
# BlurMeimputation ()