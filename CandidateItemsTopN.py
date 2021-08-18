import numpy as np
import pandas as pd
import json
import MovieLensData as MD
import FlixsterDataSub as FDS
import LastFMData as LFM

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

# ML1M

# X_training = MD.load_user_item_matrix_1m_trainingSet()
# X_test = MD.load_user_item_matrix_1m_testSet()
# X_ObfTraining= MD.load_user_item_matrix_1m_trainMasked(file_index= 7 )
# X_ObfTraining1= MD.load_user_item_matrix_1m_trainMasked(file_index= 8 )
# X_ObfTraining2= MD.load_user_item_matrix_1m_trainMasked(file_index= 9 )
# X_ObfTraining3= MD.load_user_item_matrix_1m_trainMasked(file_index= 10 )
# X_ObfTraining4= MD.load_user_item_matrix_1m_trainMasked(file_index= 11 )
# X_ObfTraining5= MD.load_user_item_matrix_1m_trainMasked(file_index= 12 )
# X_ObfTraining6= MD.load_user_item_matrix_1m_trainMasked(file_index= 13 )

# X_ObfTraining7= MD.load_user_item_matrix_1m_trainMasked(file_index= 7 )
# X_ObfTraining8= MD.load_user_item_matrix_1m_trainMasked(file_index= 14 )
# X_ObfTraining9= MD.load_user_item_matrix_1m_trainMasked(file_index= 15 )
# X_ObfTraining10= MD.load_user_item_matrix_1m_trainMasked(file_index= 16 )

"""
X_training = MD.load_user_item_matrix_100k_train()
X_test = MD.load_user_item_matrix_100k_testSet()
X_ObfTraining= MD.load_user_item_matrix_100k_trainMasked(file_index= 0 )
X_ObfTraining1= MD.load_user_item_matrix_100k_trainMasked(file_index= 1 )
X_ObfTraining2= MD.load_user_item_matrix_100k_trainMasked(file_index= 2 )
#X_ObfTraining3= MD.load_user_item_matrix_100k_trainMasked(file_index= 3 )
#X_ObfTraining4= MD.load_user_item_matrix_100k_trainMasked(file_index= 4 )
X_ObfTraining5= MD.load_user_item_matrix_100k_trainMasked(file_index= 3 )
X_ObfTraining6= MD.load_user_item_matrix_100k_trainMasked(file_index= 4 )
X_ObfTraining7= MD.load_user_item_matrix_100k_trainMasked(file_index= 5 )
X_ObfTraining8= MD.load_user_item_matrix_100k_trainMasked(file_index= 6 )
"""

# Flixster

X_training = FDS.load_user_item_matrix_FX_TrainingSet()
X_test = FDS.load_user_item_matrix_FX_Test()
#
# X_ObfTraining= FDS.load_user_item_matrix_FX_trainMasked(file_index= 0 )
# X_ObfTraining1= FDS.load_user_item_matrix_FX_trainMasked(file_index= 1 )
# X_ObfTraining2= FDS.load_user_item_matrix_FX_trainMasked(file_index= 2 )
# X_ObfTraining3= FDS.load_user_item_matrix_FX_trainMasked(file_index= 3 )
# X_ObfTraining4= FDS.load_user_item_matrix_FX_trainMasked(file_index= 4 )
# X_ObfTraining5= FDS.load_user_item_matrix_FX_trainMasked(file_index= 5 )
# X_ObfTraining6= FDS.load_user_item_matrix_FX_trainMasked(file_index= 6 )

# X_ObfTraining7= FDS.load_user_item_matrix_FX_trainMasked(file_index= 0 )
# X_ObfTraining8= FDS.load_user_item_matrix_FX_trainMasked(file_index= 7 )
# X_ObfTraining9= FDS.load_user_item_matrix_FX_trainMasked(file_index= 8 )
# X_ObfTraining10= FDS.load_user_item_matrix_FX_trainMasked(file_index= 9 )


# LastFM
X_training = LFM.load_user_item_matrix_lfm_Train()
X_test = LFM.load_user_item_matrix_lfm_Test()

# X_ObfTraining= LFM.load_user_item_matrix_lfm_masked_Train(file_index= 0 )
# X_ObfTraining1= LFM.load_user_item_matrix_lfm_masked_Train(file_index= 1 )
# X_ObfTraining2= LFM.load_user_item_matrix_lfm_masked_Train(file_index= 2 )
# X_ObfTraining3= LFM.load_user_item_matrix_lfm_masked_Train(file_index= 3 )
# X_ObfTraining4= LFM.load_user_item_matrix_lfm_masked_Train(file_index= 4 )
# X_ObfTraining5= LFM.load_user_item_matrix_lfm_masked_Train(file_index= 5 )
# X_ObfTraining6= LFM.load_user_item_matrix_lfm_masked_Train(file_index= 6 )

X_ObfTraining7= LFM.load_user_item_matrix_lfm_masked_Train(file_index= 7 )
# X_ObfTraining8= LFM.load_user_item_matrix_lfm_masked_Train(file_index= 10 )
X_ObfTraining9= LFM.load_user_item_matrix_lfm_masked_Train(file_index= 12 )
# X_ObfTraining10= LFM.load_user_item_matrix_lfm_masked_Train(file_index= 9 )

def intersectI(arr1, arr2, arr3, arr4):#, arr5):#, arr6 ):#, arr7, arr8, arr9): #, arr10, arr11
    users = {}

    N = len(arr1)

    for i in range(N):

        print("user ", i + 1)

        set1 = set(list(np.argwhere(arr1[i] == 0).T[0]))

        set2 = set(list(np.argwhere(arr2[i] == 0).T[0]))

        set3 = set(list(np.argwhere(arr3[i] == 0).T[0]))

        set4 = set(list(np.argwhere(arr4[i] == 0).T[0]))

        # set5 = set(list(np.argwhere(arr5[i] == 0).T[0]))
        #
        # set6 = set(list(np.argwhere(arr6[i] == 0).T[0]))

        # set7 = set(list(np.argwhere(arr7[i] == 0).T[0]))
        #
        # set8 = set(list(np.argwhere(arr8[i] == 0).T[0]))
        #
        # set9 = set(list(np.argwhere(arr9[i] == 0).T[0]))

        #set10 = set(list(np.argwhere(arr10[i] == 0).T[0]))

        #set11 = set(list(np.argwhere(arr11[i] == 0).T[0]))

        res = set1.intersection(set2).intersection(set3).intersection(set4)#.intersection(set5)#.intersection(set6)#.intersection(set10).intersection(set11)

        if len(res) > 0:
            users[i] = list(res)

    return users


# res = intersectI(X_training, X_ObfTraining, X_test, X_ObfTraining1, X_ObfTraining2, X_ObfTraining3, X_ObfTraining4, X_ObfTraining5, X_ObfTraining6)
res = intersectI(X_training, X_test, X_ObfTraining7, X_ObfTraining9)#, X_ObfTraining10)


# Flixster/CandidateItemsTopN/FX_Candidate_Items_UnratedIntersection_All_Except_Imputation.json
#ml-1m/CandidateItemsTopN/ML1M_Candidate_Items_UnratedIntersection_All_Except_Imputation.json
# Flixster/CandidateItemsTopN/FX_Candidate_Items_UnratedIntersection_All_Except_Imputation_top100.json
# ml1m/ML1M_Candidate_Items_UnratedIntersection_All_Except_Imputation_top150.json
# FX/FX_Candidate_Items_UnratedIntersection_All_Except_Imputation_top50.json
#lastFM/LFM_ExceptTestSet_Candidate_Items_UnratedIntersection_All_Except_Imputation_top50
with open("lastFM/LFM_NoRemoval_ExceptTestSet_Candidate_Items_UnratedIntersection_All_Except_Imputation_top500.json", "w") as fp:
    json.dump(res, fp, cls=NpEncoder)