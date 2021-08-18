import numpy as np
import pandas as pd
"""
df = pd.read_table("/Users/manel/Documents/RMSE2019/thesis/BlurMore/Flixster/flixster.event", sep="\t")
df_user = pd.read_table("/Users/manel/Documents/RMSE2019/thesis/BlurMore/Flixster/profile.csv", sep=",")

df.rating = np.round(df.rating)

userId = df.userid

df_new_user = df_user[df_user.userid.isin(userId)]

unique_id = df_new_user.userid.unique()
dictID = {}
i = 1
for k in unique_id:
    dictID[k] = i
    i += 1

df_new_user['IDtoUser'] = df_new_user.userid.map(dictID)

unique_uid = df.userid.unique()
dictIID = {}
i = 1
for k in unique_uid:
    dictIID[k] = i
    i += 1

df['IDtoUser'] = df.userid.map(dictIID)

df_user_subset = df_new_user[:3000]
userid_Users = df_user_subset.IDtoUser
df_subset = df[df.IDtoUser.isin(userid_Users)]
len(np.unique(df_subset.userid))
len(np.unique(df_user_subset.userid))

df_subset.drop('userid', axis=1)
df_subset = df_subset[['IDtoUser', 'movieid', 'rating']]

unique_iid = df_subset.movieid.unique ()
dictIID = {}
i = 1
for k in unique_iid:
    dictIID [k] = i
    i+=1

df_subset ['itemid'] = df_subset.movieid.map (dictIID)

df_subset.drop('movieid', axis =1)
df_subset = df_subset[['IDtoUser', 'itemid', 'rating']]

df_user_subset.drop('userid', axis=1)
df_user_subset = df_user_subset[['IDtoUser', 'gender', 'location', 'memberfor', 'lastlogin',
                                 'profileview', 'age']]

df_user_subset.to_csv("/Users/manel/Documents/RMSE2019/thesis/BlurMore/Flixster/subset_user_3000.csv", index=False)
df_subset.to_csv("/Users/manel/Documents/RMSE2019/thesis/BlurMore/Flixster/subset_3000.csv", index=False)
"""

def load_user_item_matrix_FX_All(max_user=2370, max_item=2835): #2370 2835 24676 11927
    """
    this function loads the user x items matrix from the  movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
    set contains only 943 users and 1330 items
    :return: user-item matrix
    """
    # Flixster/subset_FX_O.dat Flixster/subset_FX_O.csv FX/FX_subsubset_Users.dat
    df = np.zeros(shape=(max_user, max_item))
    with open("Flixster/subset_FX_O.dat", 'r') as f: #subset_FX_O All_2370_allUsers_KNN_fancy_imputation_FX_k_30

        for line in f.readlines():
            user_id, movie_id, rating, timestamp = line.split("::")
            user_id, movie_id, rating, timestamp = int(user_id), int(movie_id), float(rating), int (timestamp)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df

def load_user_item_matrix_FX_TrainingSet(max_user=2370, max_item=2835 ): #2370 2835 24676 11927 2835
    """
    this function loads the user x items matrix from the  movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
    set contains only 943 users and 1330 items
    :return: user-item matrix
    """
    # FX/FX_original_training.dat
    df = np.zeros(shape=(max_user, max_item))
    with open("Flixster/trainingSet_FX_1.dat", 'r') as f: # Flixster/trainingSet_FX_1.dat New_Flixster/FX_train.csv

        for line in f.readlines():
            user_id, movie_id, rating, timestamp = line.split("::")
            user_id, movie_id, rating, timestamp = int(user_id), int(movie_id), float(rating), int (timestamp)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df



def load_user_item_matrix_FX_Test(max_user=2370, max_item=2835): # 2370 2008
    """
    this function loads the user x items matrix from the  movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
    set contains only 943 users and 1330 items
    :return: user-item matrix
    """
    df = np.zeros(shape=(max_user, max_item))
    with open("Flixster/testSet_FX_1.dat", 'r') as f: #Flixster/testSet_FX_1.dat FX/FX_original_test.dat

        for line in f.readlines():
            user_id, movie_id, rating, timestamp = line.split("::")
            user_id, movie_id, rating, timestamp = int(user_id), int(movie_id), float(rating), int (timestamp)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df

def load_user_item_FX_Complet(max_user=2370, max_item=2835):# 2370
    df = np.zeros(shape=(max_user, max_item))
    # FX/TrainingSet_users_KNN_fancy_imputation_FX_k_30.dat
    with open(# Flixster/With_Fancy_KNN/TrainingSet_2370_allUsers_KNN_fancy_imputation_FX_k_30.dat
            "Flixster/With_Fancy_KNN/TrainingSet_2370_allUsers_KNN_fancy_imputation_FX_k_30.dat",
            'r') as f: #All_2370_allUsers_KNN_fancy_imputation_FX_k_30 // TrainingSet_2370_allUsers_KNN_fancy_imputation_FX_k_30
        for line in f.readlines():
            user_id, movie_id, rating, timestamp = line.split("::")
            user_id, movie_id, rating, timestamp = int(user_id), int(movie_id), float(rating), int(timestamp)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id - 1, movie_id - 1] = rating

    return df

def load_user_item_matrix_FX_limited_ratings(limit=20):
    user_item = load_user_item_matrix_FX_All()
    user_item_limited = np.zeros(shape=user_item.shape)
    for user_index, user in enumerate(user_item):
        # filter rating indices
        rating_index = np.argwhere(user > 0).reshape(1, -1)[0]
        # shuffle them
        np.random.shuffle(rating_index)
        for i in rating_index[:limit]:
            user_item_limited[user_index, i] = user[i]
    #print(np.sum(user_item_limited, axis=1))
    return user_item_limited

def load_user_item_matrix_FX_trainMasked(max_user=2370, max_item=2835, file_index=-1):
    df = np.zeros(shape=(max_user, max_item))
    masked_files = [
        "Flixster/TrainingSet_FX_excludeTestSet_blurme_obfuscated_0.02_greedy_avg_top-1.dat",#0
        "Flixster/BlurMore/RandomRem/TrainingSet_threshold20_ExcludeTestSet_FX_blurmepp_obfuscated_greedy_0.02_2_random.dat",
        "Flixster/BlurMore/RandomRem/TrainingSet_threshold20_ExcludeTestSet_FX_blurmepp_obfuscated_greedy_0.02_2_strategic.dat",
        "Flixster/BlurSome/Top-50-ExcludeTest/TrainingSet_thresh20_blurSome_FX_obfuscated_Top50IndicativeItems_avg_greedy_0.02_2_random.dat",
        "Flixster/BlurSome/Top-50-ExcludeTest/TrainingSet_thresh20_blurSome_FX_obfuscated_Top50IndicativeItems_avg_greedy_0.02_2_strategic.dat",
        "Flixster/BlurSome/Top-50-ExcludeTest/TrainingSet_thresh20_blurSome_FX_obfuscated_Top50IndicativeItems_pred_greedy_0.02_2_random.dat",
        "Flixster/BlurSome/Top-50-ExcludeTest/TrainingSet_thresh20_blurSome_FX_obfuscated_Top50IndicativeItems_pred_greedy_0.02_2_strategic.dat",

        # No removal
        "Flixster/DoubleCount_BlurMe/TrainingSet_FX_DCount_excludeTestSet_blurme_obfuscated_0.05_greedy_avg_top-1.dat",#7
        "Flixster/BlurSome/Top-50-NoRemoval/TrainingSet_thresh20_NoRemoval_blurSome_FX_obfuscated_Top50IndicativeItems_avg_greedy_0.1_2.dat",
        "Flixster/BlurSome/Top-50-NoRemoval/TrainingSet_thresh20_NoRemoval_blurSome_FX_obfuscated_Top50IndicativeItems_pred_greedy_0.1_2.dat",#9
        # "Flixster/BlurMe/TrainingSet_FX_blurme_obfuscated_0.02_greedy_avg_top-1.dat",#0
        # "Flixster/BlurMore/Random_Removal/TrainingSet_FX_blurmepp_obfuscated_greedy_0.02_2.dat",
        # "Flixster/BlurMore/Greedy_Removal/TrainingSet_FX_blurmepp_obfuscated_greedy_0.02_2_strategic.dat", #2
        # "Flixster/BlurSome/Top-100/TrainingSet_FX_Thresh50_Top100_Average_blurSome_obfuscated_greedy_0.02_2.dat",
        # "Flixster/BlurSome/Top-100/TrainingSet_FX_Thresh50_Top100_Average_blurSome_obfuscated_greedy_0.02_2_strategic.dat",
        # "Flixster/BlurSome/Top-100/TrainingSet_FX_Thresh50_Top100_Predicted_blurSome_obfuscated_greedy_0.02_2.dat",
        # "Flixster/BlurSome/Top-100/TrainingSet_FX_Thresh50_Top100_Predicted_blurSome_obfuscated_greedy_0.02_2_strategic.dat",
        #
        #
        # "Flixster/BlurSome/Top-50/TrainingSet_FX_Thresh50_Top50_Average_blurSome_obfuscated_greedy_0.02_2.dat",
        # "Flixster/BlurSome/Top-50/TrainingSet_FX_Thresh50_Top50_Average_blurSome_obfuscated_greedy_0.02_2_strategic.dat",#4
        # "Flixster/BlurSome/Top-50/TrainingSet_FX_Thresh50_Top50_Predicted_blurSome_obfuscated_greedy_0.02_2.dat",
        # "Flixster/BlurSome/Top-50/TrainingSet_FX_Thresh50_Top50_Predicted_blurSome_obfuscated_greedy_0.02_2_strategic.dat",#6
        #
        #
        # "FX/TrainingSet_FX_blurme_obfuscated_0.02_greedy_avg_top-1.dat",#7
        # "FX/TrainingSet_FX_blurmepp_obfuscated_greedy_0.02_2_random.dat",
        # "FX/TrainingSet_FX_blurmepp_obfuscated_greedy_0.02_2_strategic.dat",
        # "FX/TrainingSet_FX_Thresh50_Top50_Average_blurSome_obfuscated_greedy_0.02_2_random.dat",#10
        # "FX/TrainingSet_FX_Thresh50_Top50_Average_blurSome_obfuscated_greedy_0.02_2_strategic.dat",
        # "FX/TrainingSet_FX_Thresh50_Top50_Predicted_blurSome_obfuscated_greedy_0.02_2_random.dat",
        # "FX/TrainingSet_FX_Thresh50_Top50_Predicted_blurSome_obfuscated_greedy_0.02_2_strategic.dat",#13

        "Flixster/",

    ]
    with open(masked_files[file_index], 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df




def load_gender_vector_FX(max_user=2370 ): #2370 2008
    """
        this function loads and returns the gender for all users with an id smaller than max_user
        :param max_user: the highest user id to be retrieved
        :return: the gender vector
        """
    gender_vec = []
    with open("Flixster/subset_FX_User_O.csv", 'r') as f: # FX/FX_subsubset_Users.dat
        for line in f.readlines()[:max_user]: #Flixster/subset_FX_User_Last.dat
            user_id, gender, _ = line.split(",") #, location, _, _, _ , _

            if gender == "M":
                gender_vec.append(0)
            else:
                gender_vec.append(1)

    return np.asarray(gender_vec)

def load_user_item_matrix_FX_masked(max_user=2370, max_item=2835, file_index=-1):
    files = [
        "Flixster/BlurMe/All_FX_blurme_obfuscated_0.01_greedy_avg_top-1.dat",#0
        "Flixster/BlurMe/All_FX_blurme_obfuscated_0.02_greedy_avg_top-1.dat",
        "Flixster/BlurMe/All_FX_blurme_obfuscated_0.05_greedy_avg_top-1.dat",
        "Flixster/BlurMe/All_FX_blurme_obfuscated_0.1_greedy_avg_top-1.dat",#3

        "Flixster/BlurMore/Random_Removal/All_FX_blurmepp_obfuscated_greedy_0.01_2.dat", #4
        "Flixster/BlurMore/Random_Removal/All_FX_blurmepp_obfuscated_greedy_0.02_2.dat",
        "Flixster/BlurMore/Random_Removal/All_FX_blurmepp_obfuscated_greedy_0.05_2.dat",
        "Flixster/BlurMore/Random_Removal/All_FX_blurmepp_obfuscated_greedy_0.1_2.dat",

        "Flixster/BlurMore/Greedy_Removal/All_FX_blurmepp_obfuscated_greedy_0.01_2_strategic.dat",#8
        "Flixster/BlurMore/Greedy_Removal/All_FX_blurmepp_obfuscated_greedy_0.02_2_strategic.dat",#9
        "Flixster/BlurMore/Greedy_Removal/All_FX_blurmepp_obfuscated_greedy_0.05_2_strategic.dat",
        "Flixster/BlurMore/Greedy_Removal/All_FX_blurmepp_obfuscated_greedy_0.1_2_strategic.dat",#11

        "Flixster/BlurSome/Top-50/All_FX_Thresh50_Top50_Predicted_blurSome_obfuscated_greedy_0.01_2_strategic.dat",#12
        "Flixster/BlurSome/Top-50/All_FX_Thresh50_Top50_Predicted_blurSome_obfuscated_greedy_0.02_2_strategic.dat",#13
        "Flixster/BlurSome/Top-50/All_FX_Thresh50_Top50_Predicted_blurSome_obfuscated_greedy_0.05_2_strategic.dat",#14
        "Flixster/BlurSome/Top-50/All_FX_Thresh50_Top50_Predicted_blurSome_obfuscated_greedy_0.1_2_strategic.dat",  #15

        "Flixster/BlurSome/Top-100/All_FX_Thresh50_Top100_Predicted_blurSome_obfuscated_greedy_0.01_2_strategic.dat",#16
        "Flixster/BlurSome/Top-100/All_FX_Thresh50_Top100_Predicted_blurSome_obfuscated_greedy_0.02_2_strategic.dat",#17
        "Flixster/BlurSome/Top-100/All_FX_Thresh50_Top100_Predicted_blurSome_obfuscated_greedy_0.05_2_strategic.dat",#18
        "Flixster/BlurSome/Top-100/All_FX_Thresh50_Top100_Predicted_blurSome_obfuscated_greedy_0.1_2_strategic.dat",  # 19

        "Flixster/BlurSome/Top-All/All_FX_Thresh50_TopAll_Predicted_blurSome_obfuscated_greedy_0.01_2_strategic.dat",#20
        "Flixster/BlurSome/Top-All/All_FX_Thresh50_TopAll_Predicted_blurSome_obfuscated_greedy_0.02_2_strategic.dat",#21
        "Flixster/BlurSome/Top-All/All_FX_Thresh50_TopAll_Predicted_blurSome_obfuscated_greedy_0.05_2_strategic.dat",#22
        "Flixster/BlurSome/Top-All/All_FX_Thresh50_TopAll_Predicted_blurSome_obfuscated_greedy_0.1_2_strategic.dat",#23

        # Average Rating
        "Flixster/BlurSome/Top-50/All_FX_Thresh50_Top50_Average_blurSome_obfuscated_greedy_0.01_2_strategic.dat",#24
        "Flixster/BlurSome/Top-50/All_FX_Thresh50_Top50_Average_blurSome_obfuscated_greedy_0.02_2_strategic.dat",#25
        "Flixster/BlurSome/Top-50/All_FX_Thresh50_Top50_Average_blurSome_obfuscated_greedy_0.05_2_strategic.dat",#26
        "Flixster/BlurSome/Top-50/All_FX_Thresh50_Top50_Average_blurSome_obfuscated_greedy_0.1_2_strategic.dat",  #27

        "Flixster/BlurSome/Top-100/All_thresh50_blurSome_AverageFX_obfuscated_Top100_IndicativeItems_greedy_0.01_2_strategic.dat",#28
        "Flixster/BlurSome/Top-100/All_thresh50_blurSome_AverageFX_obfuscated_Top100_IndicativeItems_greedy_0.02_2_strategic.dat",#29
        "Flixster/BlurSome/Top-100/All_thresh50_blurSome_AverageFX_obfuscated_Top100_IndicativeItems_greedy_0.05_2_strategic.dat",
        "Flixster/BlurSome/Top-100/All_thresh50_blurSome_AverageFX_obfuscated_Top100_IndicativeItems_greedy_0.1_2_strategic.dat",#31

        "Flixster/BlurSome/Top-All/All_thresh50_blurSome_AverageFX_obfuscated_TopAllIndicativeItems_greedy_0.01_2_strategic.dat",#32
        "Flixster/BlurSome/Top-All/All_thresh50_blurSome_AverageFX_obfuscated_TopAllIndicativeItems_greedy_0.02_2_strategic.dat",
        "Flixster/BlurSome/Top-All/All_thresh50_blurSome_AverageFX_obfuscated_TopAllIndicativeItems_greedy_0.05_2_strategic.dat",
        "Flixster/BlurSome/Top-All/All_thresh50_blurSome_AverageFX_obfuscated_TopAllIndicativeItems_greedy_0.1_2_strategic.dat",#35

        "Flixster/BlurMore/RandomRem/All_FX_blurmepp_obfuscated_greedy_0.01_2_random.dat",#36
        "Flixster/BlurMore/RandomRem/All_FX_blurmepp_obfuscated_greedy_0.02_2_random.dat",
        "Flixster/BlurMore/RandomRem/All_FX_blurmepp_obfuscated_greedy_0.05_2_random.dat",
        "Flixster/BlurMore/RandomRem/All_FX_blurmepp_obfuscated_greedy_0.1_2_random.dat",#39

        "Flixster/BlurMore/GreedyRem/All_FX_blurmepp_obfuscated_greedy_0.01_2_strategic.dat",#40
        "Flixster/BlurMore/GreedyRem/All_FX_blurmepp_obfuscated_greedy_0.02_2_strategic.dat",#41
        "Flixster/BlurMore/GreedyRem/All_FX_blurmepp_obfuscated_greedy_0.05_2_strategic.dat",
        "Flixster/BlurMore/GreedyRem/All_FX_blurmepp_obfuscated_greedy_0.1_2_strategic.dat",#43

        # # Part I : No removal needed only Adding % fake items VS personalization
        "Flixster/DoubleCount_BlurMe/All_FX_DCount_excludeTestSet_blurme_obfuscated_0.01_greedy_avg_top-1.dat",#44
        "Flixster/DoubleCount_BlurMe/All_FX_DCount_excludeTestSet_blurme_obfuscated_0.02_greedy_avg_top-1.dat",
        "Flixster/DoubleCount_BlurMe/All_FX_DCount_excludeTestSet_blurme_obfuscated_0.05_greedy_avg_top-1.dat",
        "Flixster/DoubleCount_BlurMe/All_FX_DCount_excludeTestSet_blurme_obfuscated_0.1_greedy_avg_top-1.dat",#47

        # PerBlur average
        "Flixster/BlurSome/Top-50-NoRemoval/All_thresh20_NoRemoval_blurSome_FX_obfuscated_Top50IndicativeItems_avg_greedy_0.01_2.dat",#48
        "Flixster/BlurSome/Top-50-NoRemoval/All_thresh20_NoRemoval_blurSome_FX_obfuscated_Top50IndicativeItems_avg_greedy_0.02_2.dat",
        "Flixster/BlurSome/Top-50-NoRemoval/All_thresh20_NoRemoval_blurSome_FX_obfuscated_Top50IndicativeItems_avg_greedy_0.05_2.dat",
        "Flixster/BlurSome/Top-50-NoRemoval/All_thresh20_NoRemoval_blurSome_FX_obfuscated_Top50IndicativeItems_avg_greedy_0.1_2.dat",#51

        # PerBlur Predicted
        "Flixster/BlurSome/Top-50-NoRemoval/All_thresh20_NoRemoval_blurSome_FX_obfuscated_Top50IndicativeItems_pred_greedy_0.01_2.dat",#52
        "Flixster/BlurSome/Top-50-NoRemoval/All_thresh20_NoRemoval_blurSome_FX_obfuscated_Top50IndicativeItems_pred_greedy_0.02_2.dat",
        "Flixster/BlurSome/Top-50-NoRemoval/All_thresh20_NoRemoval_blurSome_FX_obfuscated_Top50IndicativeItems_pred_greedy_0.05_2.dat",
        "Flixster/BlurSome/Top-50-NoRemoval/All_thresh20_NoRemoval_blurSome_FX_obfuscated_Top50IndicativeItems_pred_greedy_0.1_2.dat",#55

        "Flixster/BlurSome/Top-100-NoRemoval/All_thresh20_NoRemoval_blurSome_FX_obfuscated_Top100IndicativeItems_avg_greedy_0.01_2.dat",#56
        "Flixster/BlurSome/Top-100-NoRemoval/All_thresh20_NoRemoval_blurSome_FX_obfuscated_Top100IndicativeItems_avg_greedy_0.02_2.dat",
        "Flixster/BlurSome/Top-100-NoRemoval/All_thresh20_NoRemoval_blurSome_FX_obfuscated_Top100IndicativeItems_avg_greedy_0.05_2.dat",
        "Flixster/BlurSome/Top-100-NoRemoval/All_thresh20_NoRemoval_blurSome_FX_obfuscated_Top100IndicativeItems_avg_greedy_0.1_2.dat",

        "Flixster/All_testSafe`Count_threshold20_ExcludeTestSet_FX_blurmepp_obfuscated_greedy_0.01_2_strategic.dat",#60
        "Flixster/All_testSafe`Count_threshold20_ExcludeTestSet_FX_blurmepp_obfuscated_greedy_0.02_2_strategic.dat",
        "Flixster/All_testSafe`Count_threshold20_ExcludeTestSet_FX_blurmepp_obfuscated_greedy_0.05_2_strategic.dat",#62
        "Flixster/All_testSafe`Count_threshold20_ExcludeTestSet_FX_blurmepp_obfuscated_greedy_0.1_2_strategic.dat",
    ]
    df = np.zeros(shape=(max_user, max_item))

    with open(files[file_index], 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df

"""
import pandas as pd
import matplotlib.pyplot as plt
user_df = pd.read_table("/Users/manel/Documents/RMSE2019/thesis/BlurMore/Flixster/subset_FX_User_Gender.dat", sep="::")

# user_df.columns = ["userid", "gender"] #, "age", "occupation", "zipcode"]
# count the number of male and female raters
gender_counts = user_df.gender.value_counts()

# plot the counts
plt.figure(figsize=(12, 5))
plt.bar(x= gender_counts.index[0], height=gender_counts.values[0], color="lightpink")
plt.bar(x= gender_counts.index[1], height=gender_counts.values[1], color="lightskyblue")
plt.title("Number of Male and Female users for Flixster Data", fontsize=16, fontweight="bold")
plt.xlabel("Gender", fontsize=19)
plt.ylabel("Counts", fontsize=19)
plt.savefig("images/gender_dist_FX.pdf", bbox_inches='tight')
plt.show()

df = pd.read_table("/Users/manel/Documents/RMSE2019/thesis/BlurMore/Flixster/subset_FX_10000_To_2372_Items.dat", sep="::")
df_user = pd.read_table("/Users/manel/Documents/RMSE2019/thesis/BlurMore/Flixster/subset_FX_users_10000_To_2372_gender.dat", sep="::")

df.rating = np.round(df.rating)

userId = df.userid

df_new_user = df_user[df_user.userid.isin(userId)]

unique_id = df_new_user.userid.unique()
dictID = {}
i = 1
for k in unique_id:
    dictID[k] = i
    i += 1

df_new_user['IDtoUser'] = df_new_user.userid.map(dictID)
########

df ['IDtoUser'] = df.userid.map(dictID)
##############
unique_uid = df.userid.unique()
dictIID = {}
i = 1
for k in unique_uid:
    dictIID[k] = i
    i += 1

df['IDtoUser'] = df.userid.map(dictIID)

df_user_subset = df_new_user[:3000]
userid_Users = df_user_subset.IDtoUser
df_subset = df[df.IDtoUser.isin(userid_Users)]
len(np.unique(df_subset.userid))
len(np.unique(df_user_subset.userid))

df_subset.drop('userid', axis=1)
df_subset = df_subset[['IDtoUser', 'movieid', 'rating']]

unique_iid = df_subset.movieid.unique ()
dictIID = {}
i = 1
for k in unique_iid:
    dictIID [k] = i
    i+=1

df_subset ['itemid'] = df_subset.movieid.map (dictIID)

df_subset.drop('movieid', axis =1)
df_subset = df_subset[['IDtoUser', 'itemid', 'rating']]

df_user_subset.drop('userid', axis=1)
df_user_subset = df_user_subset[['IDtoUser', 'gender', 'location', 'memberfor', 'lastlogin',
                                 'profileview', 'age']]

df_user_subset.to_csv("/Users/manel/Documents/RMSE2019/thesis/BlurMore/Flixster/subset_user_3000.csv", index=False)
df_subset.to_csv("/Users/manel/Documents/RMSE2019/thesis/BlurMore/Flixster/subset_3000.csv", index=False)"""