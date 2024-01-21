import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import nltk #NLP library
import re  #regular expression
nltk.download('stopwords') #download the extension
stemmer = nltk.SnowballStemmer("english") #create stemmer method
from nltk.corpus import stopwords #load stopwords
import string
stopword = set(stopwords.words('english'))



data = pd.read_csv("netflixdata.csv")
print(data.head(10))

data = data[["Title","Description","Content Type","Genres"]]
print(data.head())
print(data.isnull().sum())

data = data.dropna()
print(data)

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub('http?s?://\S+|www.\.S+','',text)
    text = re.sub('<.*?>+','',text)
    text = re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w','',text)