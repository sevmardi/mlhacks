import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from pandas import Series, DataFrame
from matplotlib import rcParams


ratings_list_dataset = 'datasets/ml-1m/ratings.dat'
users_list_dataset = 'datasets/ml-1m/users.dat'
movies_list_dataset = 'datasets/ml-1m/movies.dat'

ratings_list = [i.strip().split("::") for i in open(ratings_list_dataset,'r').readline()]
users_list = [i.strip().split("::") for i in open(users_list_dataset,'r').readline()]
movies_list = [i.strip().split("::") for i in open(movies_list_dataset,'r', encoding = "ISO-8859-1").readline()]


ratings = np.array(ratings_list)
users = np.array(users_list)
movies = np.array(movies_list)

ratings_df = pd.DataFrame(ratings, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype= int)
# movies_df = pd.DataFrame(movies, columns = ['MovieID', 'Title', 'Genres'])
# movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

# movies_df.head()


