import numpy as np
import pandas as pd
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
import MovieLensData as MD
import FlixsterData as FD
import FlixsterDataSub as FDS
import LastFMData as LFM
import json
import matplotlib.pyplot as plt



# X = MD.load_user_item_matrix_1m_trainingSet()
# X = FDS.load_user_item_matrix_FX_TrainingSet()
X = LFM.load_user_item_matrix_lfm_Train()
#X= X[:100, :100]
#X = MD.load_user_item_matrix_100k()
X [X == 0] = np.nan
X_filled_knn_1m = KNN(k=30).fit_transform(X)
print (X_filled_knn_1m)
X_filled_knn = np.rint(X_filled_knn_1m)

output_file = "lastFM/" # "Flixster/With_Fancy_KNN/" FX
with open(output_file + "aTrainingSet_users_KNN_fancy_imputation_LFM_k_30.dat", # TestSet_2370_allUsers_KNN_fancy_imputation_FX_k_30
                  'w') as f:
        for index_user, user in enumerate(X_filled_knn):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(
                        str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")


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


#with open(
        #'ml-1m/user_based_imputation/With_Fancy_KNN/test/NN_TrainingSet_Before_Opposite_Gender_KNN_fancy_imputation_1m_k_30.json') as json_file:
    #data = json.load(json_file)
    
with open(
        'ml-1m/user_based_imputation/With_Fancy_KNN/test_Confidence_Score_Items_Selection/NN_TrainingSet_Before_Opposite_Gender_KNN_fancy_imputation_1m_k_30.json') as json_file:
    data = json.load(json_file)

T = MD.load_gender_vector_1m()
dict_new = {}
x_axis = []
y_axis = []
for index, users in data.items():
    # print(len(users))
    # print(users[0])
    temp = []
    for x in users:
        if len(x) > 0:
            for y in x:
                temp.append(y)
    temp = list(np.unique(temp))
    print(temp)
    for z in temp:
        #print(z)
        if (T [z] == T[int(index)]):
            #print("Yeees")
            temp.remove(z)
            print(temp)
    x_axis.append(index)
    y_axis.append(len(temp))

    dict_new[index] = (temp, len(temp))
#y_pos = np.arange (len(x_axis))
#plt.bar(y_pos, y_axis)
#plt.xticks(y_pos, x_axis)
#plt.show()


plt.rcParams.update({'font.size': 28})
#f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
interval_start, interval_end = 0, 50
plt.bar(range(interval_start,interval_end), y_axis[0:50])
#plt.set_title("(A)\nOriginal data")
plt.xlabel("user ID")
plt.ylabel("#Number of Neighbors")
#ax1.set_xticks(range(1,6), [1,2,3,4,5])
#print("Original Data:", sum(y_axis))

plt.show()
"""


