import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from time import time


np.random.seed(1337)

df = pd.read_csv('data/rottentomatoes.csv')

df.head()

