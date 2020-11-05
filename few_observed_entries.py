# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## when we do imputation (file: FancyImpute_KNN.py--> KNN --> few_observed_entries.py)

from __future__ import absolute_import, print_function, division
import time
import MovieLensData as MD
import json
import numpy as np
from six.moves import range
import pandas as pd
from .common import knn_initialize
import matplotlib.pyplot as plt
import json
#from json import JSONEncoder
"""
def gender_neighbors (arr, TT):
    neighbors_SG = []
    neighbors_DG = []
    for i in range(len(arr)):
        # print("i", i)
        if TT[arr[i]] != TT[i]:
            # print("TT [k_nearest_indices [i]]", TT [k_nearest_indices [i]])
            # print("TT [i]", TT [i])
            # print("---------------------neighbors_DG---------------")
            np.delete(arr, i)
            neighbors_DG.append(i)
            # print("------------------Done------------------")
        else:
            # print("------------------neighbors_SG------------------")
            neighbors_SG.append(i)
    return  neighbors_DG, neighbors_SG


def arr_to_dic(arr):
    res = {}

    for id, v in enumerate(arr):
        #print(id, v)
        #k = "u"+ str(id)
        res[str(id)] = v

    with open('k_nearest.json', 'w') as fp:
        json.dump(res, fp)
    #return res
"""

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

def knn_impute_few_observed(
        X, missing_mask, k, verbose=False, print_interval=100):
    """
    Seems to be the fastest kNN implementation. Pre-sorts each rows neighbors
    and then filters these sorted indices using each columns mask of
    observed values.

    Important detail: If k observed values are not available then uses fewer
    than k neighboring rows.

    Parameters
    ----------
    X : np.ndarray
        Matrix to fill of shape (n_samples, n_features)

    missing_mask : np.ndarray
        Boolean array of same shape as X

    k : int

    verbose : bool
    """
    print("Hellooooo")
    T = MD.load_gender_vector_1m()
    start_t = time.time()
    n_rows, n_cols = X.shape
    # put the missing mask in column major order since it's accessed
    # one column at a time
    missing_mask_column_major = np.asarray(missing_mask, order="F")
    observed_mask_column_major = ~missing_mask_column_major
    X_column_major = X.copy(order="F")
    X_row_major, D, effective_infinity = \
        knn_initialize(X, missing_mask, verbose=verbose)
    # get rid of infinities, replace them with a very large number
    D_sorted = np.argsort(D, axis=1)
    inv_D = 1.0 / D
    D_valid_mask = D < effective_infinity
    valid_distances_per_row = D_valid_mask.sum(axis=1)

    # trim the number of other rows we consider to exclude those
    # with infinite distances
    D_sorted = [
        D_sorted[i, :count]
        for i, count in enumerate(valid_distances_per_row)
    ]

    dot = np.dot
    k_nearest_indices_filter = {}
    for i in range(n_rows):

        missing_row = missing_mask[i, :]
        missing_indices = np.where(missing_row)[0]
        row_weights = inv_D[i, :]
        if verbose and i % print_interval == 0:
            print("Imputing row %d/%d with %d missing, elapsed time: %0.3f" % (
                    i +1,
                    n_rows,
                    len(missing_indices),
                    time.time() - start_t))
        candidate_neighbor_indices = D_sorted[i]#[:30]
        user_filter = []
        #user_filter.append(list(candidate_neighbor_indices))  # list(candidate_neighbor_indices)
        #k_nearest_indices_filter[str(i)] = user_filter.copy()
        for j in missing_indices:
            observed = observed_mask_column_major[:, j]
            sorted_observed = observed[candidate_neighbor_indices]
            observed_neighbor_indices = candidate_neighbor_indices[sorted_observed]
            k_nearest_indices = observed_neighbor_indices[:k]
            #print("for i= ", i, "k_nearest_indices: ", k_nearest_indices)
            """for idx, neighbors in enumerate(k_nearest_indices):
                if (T[neighbors] == T[i]):
                    k_nearest_indices = np.delete(k_nearest_indices,
                                                           np.argwhere(k_nearest_indices == neighbors))"""
            user_filter.append(list(k_nearest_indices))  # list(candidate_neighbor_indices)
            k_nearest_indices_filter[str(i)] = user_filter.copy()
            weights = row_weights[k_nearest_indices]
            weight_sum = weights.sum()
            if weight_sum > 0:
                column = X_column_major[:, j]
                values = column[k_nearest_indices]
                X_row_major[i, j] = dot(values, weights) / weight_sum
    #print(k_nearest_indices_filter)
    print("save save save ")
    with open("Flixster/With_Fancy_KNN/NN_FX_All_Before_2370_allUsers_KNN_fancy_imputation.json", "w") as fp:
        json.dump(k_nearest_indices_filter, fp, cls=NpEncoder)

    return X_row_major

#TrainingSet
#test_Confidence_Score_Items_Selection
# in test/ we generated different version of data by limiting "k"
