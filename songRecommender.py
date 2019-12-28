"""
This project is based on Siraj Raval's recommendation system tutorial:
https://github.com/llSourcell/recommender_live

Data set comes from the following project:
Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. 
The Million Song Dataset. In Proceedings of the 12th International Society
for Music Information Retrieval Conference (ISMIR 2011), 2011.
"""
# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# truncate column width to fit on terminal output
pd.set_option('display.max_colwidth', 35)
# pd.set_option('display.max_rows', None) # view all rows

"""
class for popularity based Recommender System model
"""
class popularity_recommender_py():
    def __init__(self):
        self.train_data =   None
        self.user_id    =   None
        self.item_id    =   None
        self.popularity_recommendations = None

    # create method
    def create(self, train_data, user_id, item_id):
        self.train_data =   train_data
        self.user_id    =   user_id
        self.item_id    =   item_id

        # recommendation score = number of user_ids for each unique song
        train_data_grouped = train_data.groupby([self.item_id]).agg(
            {self.user_id:'count'}).reset_index()
        train_data_grouped.rename(columns = {'user_id': 'score'},
            inplace = True)

        # user recommendation score to sort songs
        train_data_sort = train_data_grouped.sort_values(
            ['score', self.item_id], ascending = [0, 1])

        # use score to generate 
        train_data_sort['rank'] = train_data_sort['score'].rank(
            ascending = 0, method = 'first')

        #determine top ten recommendataions
        self.popularity_recommendations = train_data_sort.head(10)

    # recommend method
    def recommend(self, user_id):
        user_recommendations = self.popularity_recommendations

        # add user_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user_id

        # bring user_id column to front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        return user_recommendations
"""
end class
"""

"""
class for item-similarity based Recommender System model
"""
class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None

    # get unique items (songs) corresponding to a given user
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        return user_items
        
    # get unique users for a given item (song)
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
        return item_users
        
    # get unique items (songs) in the training data
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())
        return all_items
        
    # construct cooccurence matrix
    def construct_cooccurence_matrix(self, user_songs, all_songs):

        # get users for all songs in user_songs
        user_songs_users = []        
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))

        # initialize the item cooccurence matrix of size 
        # len(user_songs) X len(songs)
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs),
            len(all_songs))), float)
           
        # calculate similarity between user songs and all unique songs
        # in the training data
        for i in range(0,len(all_songs)):

            # calculate unique listeners (users) of song (item) i
            songs_i_data = self.train_data[
                self.train_data[self.item_id] == all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())
            
            for j in range(0,len(user_songs)):       
                    
                # get unique listeners (users) of song (item) j
                users_j = user_songs_users[j]
                    
                # calculate intersection of listeners of songs i and j
                users_intersection = users_i.intersection(users_j)
                
                # calculate cooccurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:

                    # calculate union of listeners of songs i and j
                    users_union = users_i.union(users_j)
                    cooccurence_matrix[j,i] = float(
                        len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
        # return            
        return cooccurence_matrix

    #Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs,
        user_songs):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(
            cooccurence_matrix))

        
        # calculate a weighted average of the scores in cooccurence matrix
        # for all user songs
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(
            cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        # sort the indices of user_sim_scores based upon their value
        # also maintain the corresponding score
        sort_index = sorted(((e,i) for i,e in enumerate(list(
            user_sim_scores))), reverse=True)
    
        # create a dataframe from the following
        columns = ['user_id', 'song', 'score', 'rank']
        # index = np.arange(1) # array of numbers for the number of samples
        df = pd.DataFrame(columns=columns)
         
        # fill the dataframe with top 10 item based recommendations
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[
                sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],
                    sort_index[i][0],rank]
                rank = rank+1
        
        # handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no songs for training the item " +
            "similarity based recommendation model.")
            return -1
        else:
            return df

    # create the item similarity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    # use the item similarity based recommender system model to
    # make recommendations
    def recommend(self, user):
        
        # A. get all unique songs for this user
        user_songs = self.get_user_items(user)    
        print("No. of unique songs for the user: %d" % len(user_songs))
        
        #B. get all unique items (songs) in the training data
        all_songs = self.get_all_items_train_data()
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        # C. construct item cooccurence matrix of size 
        # len(user_songs) X len(songs)
        cooccurence_matrix = self.construct_cooccurence_matrix(
            user_songs, all_songs)
        
        # D. use the cooccurence matrix to make recommendations
        df_recommendations = self.generate_top_recommendations(
            user, cooccurence_matrix, all_songs, user_songs)

        # return    
        return df_recommendations
    
    # get similar items to given items
    def get_similar_items(self, item_list):
        user_songs = item_list
        
        #B. get all unique items (songs) in the training data
        all_songs = self.get_all_items_train_data()
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        #C. construct item cooccurence matrix of size 
        # len(user_songs) X len(songs)
        cooccurence_matrix = self.construct_cooccurence_matrix(
            user_songs, all_songs)
        
        #D. use the cooccurence matrix to make recommendations
        user = ""
        df_recommendations = self.generate_top_recommendations(
            user, cooccurence_matrix, all_songs, user_songs)
        
        # return
        return df_recommendations
"""
end class
"""

# open files with database information
# 10000.txt     =>  user id, song id, listen count
# song_data.csv =>  song id, title, release, artist, year
triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'

# read the three columns of triplet file as data frames
song_df_1 = pd.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

# combine triplet file with metadata file and drop duplicate columns
# song_df => user id, song id, listen count, title, release, artist, year
song_df_2 = pd.read_csv(songs_metadata_file)
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']),
                   on = "song_id", how = "left")

# visualize combined data set
print("Data Set:")
print("")
print(song_df)
print("")
print("")

# Transormation
# merge title and artist_name into one column
song_df['song'] = song_df["title"] + ' - ' + song_df["artist_name"]
song_grouped = song_df.groupby(
    ['song']).agg({'listen_count':'count'}).reset_index()

# calculate the percentage of a song's listen count compared to the total listen
# count of all songs
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage'] = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])

# print
print("Relative Percentages:")
print("")
print(song_grouped)

# determine the number of unique users and songs
users = song_df['user_id'].unique()
print("Unique Users: ", len(users))
songs = song_df['song'].unique()
print("Unique Songs: ", len(songs))
print("")
print("")

# split dataset into training and testing data using 20% testing size
train_data, test_data = train_test_split(song_df, test_size = 0.20,
    random_state = 0)

# use a popularity-based recommender class as a blackbox to train model
# create a recommender: input = user_id, output = list of recommended songs
# will be based on the popularity of each song
pm = popularity_recommender_py()
pm.create(train_data, 'user_id', 'song')

# make a prediction for a random user
print("Popularity-based Recommendations:")
print("")
user_id = users[5]
print(pm.recommend(user_id))
print("")
print("")

"""
utilize the item similarity based collaboritive filtering model
content based => recalls what a user liked in the past
collaborative based => predicts based on similar user likes
item-item filtering => co-occurrence matrix based on a song a user likes
"""

# declare item-similarity based recommender class and feed with training data
is_model = item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')

# generate top recommendations by calculating a weighted average of the scores
print("Similarity-based Recommendations:")
print("")
user_id = users[5]
user_items = is_model.get_user_items(user_id)
print("-----------------------------------------------------------------------")
print("training data for the user ID: %s" % user_id)
print("-----------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("-----------------------------------------------------------------------")
print("recommendation process details:")
print("-----------------------------------------------------------------------")

# recommend songs for the user using the personalized model
print(is_model.recommend(user_id))