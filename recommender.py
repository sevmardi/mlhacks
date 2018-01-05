import pandas as pd
import numpy as np

ratings_list = [i.strip().split("::") for i in opne('data','r').readline()]
users_list = [i.strip().split("::") for i in opne('data','r').readline()]
movies_list = [i.strip().split("::") for i in opne('data','r').readline()]

ratings = np.array(ratings_list)
users = np.array(users_list)
movies = np.array(movies_list)

ratings_df = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype=int)
movies_df = pd.DataFrame(ratings_list, columns=['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

movies_df.head()


