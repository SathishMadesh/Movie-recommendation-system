import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import nltk #NLP library
import re  #regular expression
nltk.download('stopwords', quiet=True) #download the extension
stemmer = nltk.SnowballStemmer("english") #create stemmer method
from nltk.corpus import stopwords #load stopwords
import string
stopword = set(stopwords.words('english'))



data = pd.read_csv("netflixdata.csv")

data = data[["Title","Description","Content Type","Genres"]]
# print(data.head())
# print(data.isnull().sum())

data = data.dropna()

def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]','',text)
    text = re.sub(r'http?s?://\S+|www.\.S+','',text)
    text = re.sub(r'<.*?>+','',text)
    text = re.sub(r'[%s]'%re.escape(string.punctuation),'',text)
    text = re.sub(r'\n','',text)
    text = re.sub(r'\w*\d\w','',text)

    text = [word for word in text.split(' ') if word not in stopword]
    text =" ".join(text)

data["Title"] = data["Title"].apply(clean)

print(data.Title.sample(10))
print(data.Genres)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

feature = data['Genres'].tolist()

#create an instance of TF-IDF-Vectorizer
tfidf = TfidfVectorizer(stop_words="english")

#fit and transform the vectorizer on our corpus
tfidf_matrix = tfidf.fit_transform(feature)

#compuer the cosine similarity matrix
similarity = cosine_similarity(tfidf_matrix)

indices = pd.Series(data.index, index=data['Title']).drop_duplicates()


def movie_recommendation(title,similarity = similarity):
    index = indices[title]
    similarity_scorces = list(enumerate(similarity[index]))
    similarity_scorces = sorted(similarity_scorces,key=lambda x:x[1], reverse=True)
    similarity_scorces = similarity_scorces[0:10]
    movieindices = [i[0] for i in similarity_scorces]
    return data['Title'].iloc[movieindices]

movie_recommendation("#Alive")