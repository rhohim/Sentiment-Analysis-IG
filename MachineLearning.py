import nltk
nltk.download('punkt')
import pandas as pd
import numpy as np
import json
import os
import sys
import re
import time
import nltk
from sklearn.model_selection import LeaveOneOut
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import requests
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import pickle

# Machine Learning Algorithm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def process(s):
    s = str(s)
    stopwords = StopWordRemoverFactory().get_stop_words()
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    s = s.lower()
    s = re.sub('\n', ' ', s)
    # Remove link web
    s = re.sub(r'http\S+', '', s)
    # Remove @username
    s = re.sub('@[^\s]+', '', s)
    # Remove #tagger
    s = re.sub(r'#([^\s]+)', '', s)
    # Remove angka termasuk angka yang berada dalam string
    # Remove non ASCII chars
    s = re.sub(r'[^\x00-\x7f]', r'', s)
    s = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', s)
    s = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", s)
    s = re.sub(r'\\u\w\w\w\w', '', s)
    # Remove simbol, angka dan karakter aneh
    s = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ", s)
    #s = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ", s)
    # s = re.sub(r'\d+', '', s)

    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    # print(len(tokens))
    if len(tokens) == 0 :
        # print("###########################")
        tokens = ['Netral']
    print(tokens)
    return tokens

def tokens_vector(tokens, label):
    k = np.zeros(len(word_index_map) + 1)
    for t in tokens:
        if t in word_index_map:
            i = word_index_map[t]
            k[i] += 1
    if k.sum() > 0:
        k = k / k.sum()
    k[-1] = label
    return k

show_data = pd.DataFrame(columns=['Teks','Prediction'])
df_sentiment = pd.read_csv('dataset1000.csv')

positif, netral, negatif = 0, 0, 0
tokenized, data = [], []
word_index_map = {}
cur_index = 0


for i in df_sentiment.index:
    lb = str(df_sentiment['category'][i])
    x = str(df_sentiment['comment'][i])
    if pd.notna(lb) and pd.notna(x):
        tokens = process(x)
        tokenized.append({'token':tokens,'label':lb})
        for token in tokens:
        #print(token)
            if token not in word_index_map:
                word_index_map[token] = cur_index
                cur_index += 1
                
for tokens in tokenized:
        y = tokens_vector(tokens["token"], tokens["label"])
        data.append(y.tolist())

# print("data before ", data)
data = np.array(data)
# print("shape ", data.shape)
# print("data after ", data)
X = data[:,:-1]
y = data[:,-1]
y[np.isnan(y)] = 1
# print("data X ", X)
# print(len(X))
# print(len(y))

loo = LeaveOneOut()
loo.get_n_splits(X)

# model = GaussianNB()  # 1000 0,922 
model = SVC() #1000 0,883
# model = KNeighborsClassifier() 

scores , y_pred, y_tes = [], [], []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # model.partial_fit(X_train, y_train, classes=[0,1,2]) #GaussianNB
    model.fit(X_train, y_train)
    #pickle.dump(model, open("model.pkl","wb"))
    y_predict= model.predict(X_test)
    #print(y_predict)
    y_pred.append(y_predict.tolist()[0])
    print(y_test[0])
    y_tes.append(y_test.tolist()[0])
    scores.append(accuracy_score(y_test,y_predict))

    #scores.append(model.score(X_test,y_test))
print(np.array(scores).sum()/len(scores))
#print(y_pred)
print(confusion_matrix(y_tes,y_pred))
#pickle.dump(model, open(model_file, "wb"))
# with open('model_svm.pkl', 'wb') as model_file:
#     pickle.dump(model, model_file)
    
#Predict

text = pd.read_csv('resultabc5dasarweek4.csv')
teks = text["text"]

for x in teks:
        print(x)
        data = []
        tokens = process(x)
        tokenized.append({'token':tokens,'label':1})
        for tokens in tokenized:
            y = tokens_vector(tokens["token"], tokens["label"])
            data.append(y.tolist())
        json_file_path = 'tokenized.json'
        with open(json_file_path, 'w') as json_file:
            json.dump(tokenized, json_file)

        # print(f"Data saved to {json_file_path}")
        # print("data before ", data)
        data = np.array(data)
        # print("data after ", data)
        X = data[:,:-1]
        y = data[:,-1]
        # print("data X ", X)
        loo.get_n_splits(X)
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            prediction = model.predict(X_test)
            pred = prediction.tolist()[0]

            print(pred)
            
        if pred == 0.0:
            label = 'NEGIATIF'
            negatif += 1
        elif pred == 1.0:
            label = 'NETRAL'
            netral +=1
        elif pred == 2.0:
            label = 'POSITIF'
            positif +=1
        print("negatif : " ,  negatif)
        print("netral : " ,  netral)
        print("positif : " ,  positif)
        # show_data = show_data.append({'Teks': x,
        #                          'Prediction': label
        #                          }, ignore_index=True)
        show_data = pd.concat([show_data, pd.DataFrame({'Teks': [x], 'Prediction': [label]})], ignore_index=True)
        counts_df = pd.DataFrame({
            'Sentiment': ['NEGATIF', 'NETRAL', 'POSITIF'],
            'Count': [negatif, netral, positif]
        })

        counts_df.to_csv('sentiment_countabc5dasarweek4.csv', index=False)
