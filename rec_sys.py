import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re


data = pd.read_csv("netflixdata.csv")
print(data.head(10))

data = data[["Title","Description","Content Type","Genres"]]
print(data.head())
print(data.isnull().sum())

data = data.dropna()
print(data)