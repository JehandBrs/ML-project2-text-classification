import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter  import PorterStemmer
from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud

import pandas as pd
import random, time
from babel.dates import format_date, format_datetime, format_time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


# Preprocessing functions


def import_data(path_dataset_neg, path_dataset_pos):
    """
    This function imports the data set, adds labels and returns a Pandas Dataframe, without duplicates. 
    Input : path of negative data set, path of postive dataset 
    Output: Pandas data frame with two columns : text and label
    """

    #Kaggle version
    train_neg = [tweet[:-1] for tweet in open(path_dataset_neg).readlines()]
    train_pos = [tweet[:-1] for tweet in open(path_dataset_pos).readlines()]
        
    X, y = train_neg + train_pos, [-1 for i in range(len(train_neg))]+[1 for i in range(len(train_pos))]
    df = pd.DataFrame(list(zip(y, X)), columns = ['label','text'], dtype = str)
    df.drop_duplicates(inplace = True)# Delete duplicate Tweets
    df['label'] = df['label'].astype(int)
    
    return df




def cleaning_data(df):
    """
    This function removes special characters, numbers, url links, single characters  
    Input : Pandas data frame with two columns : text and label 
    Output: Pandas data frame with two columns : text and label
    """
    
    # remove special characters from text column
    df.text = df.text.str.replace('[#,@,&]', '')
    
    #Replace special characters
    df.text = df.text.str.replace('(','')
    df.text = df.text.str.replace(')','')
    df.text = df.text.str.replace('=','')
    df.text = df.text.str.replace('!','')
    df.text = df.text.str.replace('?','')
    df.text = df.text.str.replace('"','')
    df.text = df.text.str.replace('_','')
    df.text = df.text.str.replace('-','')
    df.text = df.text.str.replace(',','')
    df.text = df.text.str.replace('.','')
    df.text = df.text.str.replace(';','')
    df.text = df.text.str.replace('+','')
    df.text = df.text.str.replace('<user>','')
    df.text = df.text.str.replace('<rt>','')
    df.text = df.text.str.replace(':','')
    df.text = df.text.str.replace('/','')
    df.text = df.text.str.replace('<','')
    df.text = df.text.str.replace('>','')
    df.text = df.text.str.replace('\'s','')
    
    # Remove digits
    df.text = df.text.str.replace('\d*','')
    
    #Remove www
    df.text = df.text.str.replace('w{3}','')
    # remove urls
    df.text = df.text.str.replace("http\S+", "")
    # remove multiple spaces with single space
    df.text = df.text.str.replace('\s+', ' ')
    #remove all single characters (except "i")
    df.text = df.text.str.replace(r'\s+[a-hA-H]\s+', '')
    df.text = df.text.str.replace(r'\s+[j-zJ-Z]\s+', '')
    df.text = df.text.str.replace(r'\s+[i-iI-I]\s+',' ')
    return df



def remove_stopwords(df):
    
    """
    This function stopwords, defined in the list in the function.
    We delete Twitter specific words, english stopwords, but we keep negative forms of verbs and negative adverbs
    Input : Pandas data frame with two columns : text and label 
    Output: Pandas data frame with two columns : text and label
    """
    
    stop_words = ['i', 'me', 'my', 'myself', 'we','url' 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain']
    stop_words.extend(['u', 'wa', 'ha','ho', 'would', 'com', 'user','<user>', '<rt>' 'url', 'rt', 'custom picture', 'i\'m', 'picture frame','<url>', 'positer frame', 'x','i\'ll'])
    stop_words.remove('not')
    stop_words.remove('no')
    stop_words.remove('nor')
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    return df


def Porter_stemmer(df):
    """
    This function applies Porter Stemmer methodology to reduces words to their stem
    Input : Pandas data frame with two columns : text and label 
    Output: Pandas data frame with two columns : text and label
    """   
    stemmer = PorterStemmer()
    df['text'] = df['text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    return df

def snow_ball_stemmer(df):
    """
    This function applies Snowball Stemmer methodology to reduces words to their stem
    Input : Pandas data frame with two columns : text and label 
    Output: Pandas data frame with two columns : text and label
    """   
    snow_stemmer = SnowballStemmer(language='english')
    df['text'] = df['text'].apply(lambda x: ' '.join([snow_stemmer.stem(word) for word in x.split()]))
    return df

def lemmatize_text(df):
    """
    This function applies World Net Lemmatizing methodology to reduces words to their stem
    Input : Pandas data frame with two columns : text and label 
    Output: Pandas data frame with two columns : text and label
    """   
    lemmatizer = nltk.stem.WordNetLemmatizer()
    df['text'] = df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    return df


def Basic_Vectorizer(df):
    """
    This function transforms text into a matrix mapping X using all words in text as vocabulary list 
    It also transform the labels to a numpy vector y
    Input : Pandas data frame with two columns : text and label 
    Output:  X vector of features, y vector of labels
    """   
    text = df['text']
    y = df['label'].to_numpy()
    
    basic_vectorizer = CountVectorizer(binary=True)
    basic_vectorizer.fit(text)
    X = basic_vectorizer.transform(text)
    
    return X, y



def N_Gram_Vectorizer(df, N):
    """
    This function transforms text into a matrix mapping X using all words in text as vocabulary list.
    It maps N-grams (series of N consecutive words)
    It also transform the labels to a numpy vector y
    Input : Pandas data frame with two columns : text and label, N the parameter for N-grams 
    Output:  X vector of features, y vector of labels
    """   
    text = df['text']
    y = df['label'].to_numpy()
    
    #adding two or three word sequences (bigrams or trigrams)
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, N))
    ngram_vectorizer.fit(text)
    X = ngram_vectorizer.transform(text)
    
    return X, y


def SVD_preprocessing(X, y, N):
    
    """
    This function applies SVD transformation to the features matrix X, keeping the N most significant drivers
    Input : Matrix of features X, vector of labels y, parameter N for number of drivers to keep
    Output:  X vector of features after SVD, y vector of labels
    """  
    clf = TruncatedSVD(100)
    X_SVD = clf.fit_transform(X)
    
    return X_SVD, y
