import numpy as np
import pandas as pd
from fancyimpute import KNN
import MovieLensData as MD
import json
import matplotlib.pyplot as plt

X = MD.load_user_item_matrix_1m()
X[X == 0] = np.nan
X_filled_knn_1m = KNN(k=30).fit_transform(X)

X_filled_knn = np.rint(X_filled_knn_1m)

output_file = "ml-1m/With_Fancy_KNN/"
with open(output_file + "All_Users_KNN_fancy_imputation_ML1M_k_30.dat",
          'w') as f:
    for index_user, user in enumerate(X_filled_knn):
        for index_movie, rating in enumerate(user):
            if rating > 0:
                f.write(
                    str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                        int(np.round(rating))) + "::000000000\n")