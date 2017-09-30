import graphlab
#from graphlab.recommender.factorization_recommender import create
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from john_cleaning import *

"""ratings_contents = pd.read_table("data/ratings.dat")
sf = graphlab.SFrame(ratings_contents)
m1 = graphlab.recommender.factorization_recommender.create(sf, user_id = 'user_id', item_id = 'joke_id', target = 'rating', solver = 'als')"""



#predict one:
#one_datapoint_sf = graphlab.SFrame({'user_id': [49541, 39499], 'joke_id': [113, 37]})
#m1.predict(one_datapoint_sf)

#Read in test set:
test_set = pd.read_csv('data/sample_submission.csv')
testusers = test_set['user_id'].tolist()
testjokes = test_set['joke_id'].tolist()

#predictions:
test_predict_sf = graphlab.SFrame({'user_id': testusers, 'joke_id': testjokes})
#m1.predict(test_predict_sf)
predictions = m1.predict(test_predict_sf)

test_output_df = test_set.copy()
test_output_df['rating'] = list(predictions)
test_output_df.to_csv('output1.csv')

#This is from John's code:
filename = 'data/jokes.dat'
list_of_jokes = clean_raw_data(filename)
vectors, features = convert_to_tdidf(list_of_jokes)
tfidf_matrix = vectors.toarray()
tfidf_matrix.argsort()

#Convert to pandas dataframe:
"""itemdata_df = pd.DataFrame(tfidf_matrix)
itemdata_df.index += 1
itemdata_df['joke_id'] = itemdata_df.index


#Running the recommender...
ratings_contents = pd.read_table("data/ratings.dat")
sf = graphlab.SFrame(ratings_contents)
itemdata_sf = graphlab.SFrame(itemdata_df)
m1 = graphlab.recommender.factorization_recommender.create(sf, user_id = 'user_id', item_id = 'joke_id', target = 'rating', solver = 'als')"""
