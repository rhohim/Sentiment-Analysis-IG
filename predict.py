import pandas as pd
import numpy as np
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pickle
from sklearn.model_selection import LeaveOneOut
import json

loo = LeaveOneOut()

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


positif, netral, negatif = 0, 0, 0
show_data = pd.DataFrame(columns=['Teks', 'Prediction'])
tokenized , data = [],[]
word_index_map = {}
cur_index = 0
df_sentiment = pd.read_csv('dataset1000.csv')
file_path = 'tokenized.json'
# with open(file_path, 'r') as file:
#     tokenized = json.load(file)
# # print(tokenized)
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


with open('model_svm.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Predict
text = pd.read_csv('result copy.csv')
teks = text["text"]

# Initialize counters for sentiment categories
positif, netral, negatif = 0, 0, 0

# Create a DataFrame to store prediction results
show_data = pd.DataFrame(columns=['Teks', 'Prediction'])

for x in teks:
    data = []
    if x != None:
        print(x)
        tokens = process(x)
        print("tokens ", tokens)
        tokenized.append({'token':tokens,'label':1})
    for tokens in tokenized:
        y = tokens_vector(tokens["token"], tokens["label"])
        data.append(y.tolist())
    print("data before ", data)
    data = np.array(data)
    print("shape ", data.shape)
    print("data after ", data)
    X = data[:, :-1]
    y = data[:, -1]
    # y[np.isnan(y)] = 1
    print("data X ", X)
    loo.get_n_splits(X)
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
#     # Make predictions using the trained model
        prediction = model.predict(X)
        pred = prediction.tolist()[0]

        print(pred)

        # Interpret the prediction and update counters
        if pred == 0.0:
            label = 'NEGATIF'
            negatif += 1
        elif pred == 1.0:
            label = 'NETRAL'
            netral += 1
        elif pred == 2.0:
            label = 'POSITIF'
            positif += 1

    # Append the prediction to the show_data DataFrame
        # show_data = show_data.append({'Teks': x, 'Prediction': label}, ignore_index=True)



# Display and save the prediction results
print("NEGATIF:", negatif)
print("NETRAL:", netral)
print("POSITIF:", positif)

# # Save the prediction results to a CSV file
# show_data.to_csv('prediction_results.csv', index=False)