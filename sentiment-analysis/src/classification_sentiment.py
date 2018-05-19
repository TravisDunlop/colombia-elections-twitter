
import pandas as pd
import numpy as np
import pickle
import os
import preprocessor as tweet_p
import datetime
import nltk
import gc
from sklearn.feature_extraction.text import strip_accents_unicode
from sklearn.semi_supervised import LabelSpreading
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, VectorizerMixin
from sklearn.model_selection import train_test_split

import re

print("Initialized...")

data_path =  "/home/juan/Desktop/Text_Mining/Om_Project/Data"
tables_path = "/home/juan/Desktop/Text_Mining/Om_Project/colombia-elections-twitter/sentiment-analysis/tables"

with open(os.path.join(tables_path,"sentiment_labels"), 'rb') as fp:
    sentiment_label = pickle.load(fp)
fp.close()

with open(os.path.join(tables_path,"tweet_id"), 'rb') as fp:
    tweet_id = pickle.load(fp)
fp.close()

labels_ = pd.DataFrame({"sentiment_label":sentiment_label,"tweet_id":tweet_id})
labels_train, labels_test= train_test_split(labels_,random_state=42)


stopwords = nltk.corpus.stopwords.words(['spanish'])
stemmer = nltk.stem.snowball.SnowballStemmer('spanish')

my_list=['cual','pm','am','va','p m','a m','q','ver','hoy',
        'aca','aqui','da','m','p','tal','tan','haga',
        'v','u','como','ve','retweeted','fm','usted','hace',
        'responde','espere','tambien','dice','dicen','dijo',
        'segun','segun','cada','anos','aun','aunque','cree','ay',
        'creen','creer','creo','decir','demas','estan','retwit',
        'hace','hacen','hacer','hecha','hicieron' ,'hizo','cosa','d',
        'porque','demas','diga','digo','estan','etc','ir','llega','pa','ser',
        'hoy','puede','quiere','ser','sera','si','van','ir',
        'sr','tan','ud','va','van','vamos','voy','x','vez','sra',
        'ahi','ahora','vez','via','vea','mas','b','uds','ahi','alla',
        'dejen','dejar','cosas','asi','solo','rt','ps','petro',
        'ivanduque','petrogustavo','sergio_fajardo','DeLaCalleHum',
        'German_Vargas','duque','fajardo','vargas','lleras','colombia',
        'alvaro','uribe','colombiano','venezuela','candidato','voto','votar']
stopwords.extend(my_list)

def preprocessor_tweet(s):

    tweet_p.set_options(tweet_p.OPT.EMOJI,
                        tweet_p.OPT.URL,
                        tweet_p.OPT.RESERVED,
                        tweet_p.OPT.SMILEY,
                        tweet_p.OPT.MENTION)
    s = re.sub(r'@petrogustavo', 'petrogustavo', s)
    s = re.sub(r'@sergio_fajardo', 'sergio_fajardo', s)
    s = re.sub(r'@IvanDuque','IvanDuque',s)
    s = re.sub(r'@AlvaroUribeVel','AlvaroUribeVel',s)
    s = re.sub(r'@JuanManSantos','JuanManSantos',s)
    s = re.sub(r'@German_Vargas','German_Vargas',s)
    s = re.sub(r'@ClaudiaLopez','ClaudiaLopez',s)
    s = re.sub(r'@DeLaCalleHum','DeLaCalleHum',s)
    s = tweet_p.clean(s)
    s = re.sub(r'\b(?:a*(?:ja)+h?|(?:l+o+)+l+)\b', ' ', s)
    s = re.sub(r'[^\w]', ' ', s)
    s = strip_accents_unicode(s.lower())
    s = tweet_p.clean(s)

    return s

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

countvectorizer_ = CountVectorizer(tokenizer = tokenize_only,
                                    stop_words = stopwords,
                                    max_df=0.95,
                                    min_df=0.009,
                                    ngram_range=(1, 2))
###semisup
###
from sklearn.metrics import (precision_score, recall_score,f1_score,accuracy_score,roc_auc_score,roc_curve)

data = pd.read_csv( os.path.join( data_path,"db_tweets.csv" ) , sep = "|", lineterminator = '\n')
data_RF = data.merge(labels_,how='inner',left_on = "tweet_id",right_on = "tweet_id")
data_test = data.merge(labels_test,how='inner',left_on = "tweet_id",right_on = "tweet_id")
data = data.merge(labels_train,how='left',left_on = "tweet_id",right_on = "tweet_id")
data_labeled = data[data.sentiment_label>=0]
data = data[data.sentiment_label.isnull()].sample(10000)

data = pd.concat([data,data_labeled])

clean_tweets = data.text_tweet.apply (preprocessor_tweet)
clean_tweets_test = data_test.text_tweet.apply (preprocessor_tweet)
print("cleaning done!")

countvectorizer_matrix = countvectorizer_.fit_transform (clean_tweets)
countvectorizer_matrix_test = countvectorizer_.transform (clean_tweets_test)

labels_g = np.array(data.sentiment_label)
labels_g[np.where(np.isnan(labels_g))] = -1
labels_g[np.where(np.isin(labels_g,99))] = 2
print("fitting model knn = 3")
label_prop_model_500 = LabelSpreading(kernel = 'knn',n_jobs = 3,n_neighbors=7)
from scipy.sparse import csgraph

label_prop_model_500.fit(countvectorizer_matrix.toarray(),labels_g)

print("done!")

y_test = data_test.sentiment_label
y_pred = label_prop_model_500.predict(countvectorizer_matrix_test.toarray())

pres_semisup = precision_score(y_test, y_pred,average = 'weighted')
recall_semisup =  recall_score(y_test, y_pred,average = 'weighted')
f1_semisup =  f1_score(y_test, y_pred,average = 'weighted')
acc_semisup =  accuracy_score(y_test, y_pred)


#### RF
########
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt


X_train,X_test,y_train,y_test = train_test_split(data_RF.text_tweet,data_RF.sentiment_label,random_state=42)

y_test = y_test.reset_index(drop = True)
y_train = y_train.reset_index(drop = True)

y_test[np.where(np.isin(y_test,99))[0]] = 2
y_train[np.where(np.isin(y_train,99))[0]] = 2

clean_tweets_train = X_train.apply(preprocessor_tweet)
clean_tweets_test = X_test.apply(preprocessor_tweet)

countvectorizer_ = CountVectorizer(tokenizer = tokenize_and_stem,
                                    stop_words = stopwords,
                                    max_df=0.95,
                                    min_df=0.002,
                                    ngram_range=(1, 3))

countvectorizer_matrix = countvectorizer_.fit_transform (clean_tweets_train)
countvectorizer_matrix_test = countvectorizer_.transform(clean_tweets_test)

model_rf = RandomForestClassifier(**{'n_estimators': 400,
 'min_samples_split': 10,
 'min_samples_leaf': 4,
 'max_features': 'auto',
 'max_depth': 70,
 'bootstrap': True})
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
# Fit the random search model

model_rf.fit (countvectorizer_matrix, y_train)
y_pred = model_rf.predict (countvectorizer_matrix_test)

pres_sup = precision_score(y_test, y_pred,average = 'weighted')
recall_sup =  recall_score(y_test, y_pred,average = 'weighted')
f1_sup =  f1_score(y_test, y_pred,average = 'weighted')
acc_sup =  accuracy_score(y_test, y_pred)

resultados = pd.DataFrame({"f1_score":[f1_sup,f1_semisup],
                            "precision":[pres_sup,pres_semisup],
                            "recall":[recall_sup,recall_semisup],
                            "accuracy":[acc_sup,acc_semisup]},index=["Supervised","Semisupervised"])


round(resultados,3).to_csv(os.path.join(data_path,"results_classification.csv"))
