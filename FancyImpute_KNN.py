"""
This function will help in the creation of the personalized list via the generation of confidence score + imputed matrix
you need to go to KNN-knn_impute_few_observed. I uploaded few_observed_entries.py file with the needed added line codes 
that you need to adapt to your few_observed_entries.py file 
"""
import numpy as np
import pandas as pd
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
import MovieLensData as MD
# import FlixsterData as FD
import FlixsterDataSub as FDS
# import LastFMData as LFM
import json
import matplotlib.pyplot as plt
import GoodBookData as GBD



# X = MD.load_user_item_matrix_1m_trainingSet()
# X = FDS.load_user_item_matrix_FX_TrainingSet()
# X = LFM.load_user_item_matrix_lfm_Train()
X = GBD.load_user_item_matrix_GB_TrainingSet()
#X= X[:100, :100]
#X = MD.load_user_item_matrix_100k()
X [X == 0] = np.nan
X_filled_knn_1m = KNN(k=30).fit_transform(X)
print (X_filled_knn_1m)
X_filled_knn = np.rint(X_filled_knn_1m)

output_file = "/Users/mslokom/Documents/RecSys_News/goodbook/"
with open(output_file + "TrainingSet_users_KNN_fancy_imputation_GBD_k_30.dat",
                  'w') as f:
        for index_user, user in enumerate(X_filled_knn):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(
                        str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

