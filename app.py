# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 18:55:10 2020

@author: Chintan Munjani
"""


from flask import Flask, render_template, request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    # df = pd.read_csv('spam.csv', encoding='latin-1')
    # df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    # df.rename(columns={'v1':'label', 'v2':'message'}, inplace=True)
    # df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # X = df['message']
    # y = df['label']
    
    # cv = CountVectorizer()
    # X = cv.fit_transform(X)
    
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    
    # clf = MultinomialNB()
    # clf.fit(X_train, y_train)
    # clf.score(X_test, y_test)
    
    # with open('transform', 'wb') as f:
    #     pickle.dump(cv, f)
    # with open('nlp_model', 'wb') as file:
    #     pickle.dump(clf, file)
    
    with open('transform', 'rb') as f:
        cv1 = pickle.load(f)
    with open('nlp_model', 'rb') as file:
        clf1 = pickle.load(file)
        
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv1.transform(data).toarray()
        my_prediction = clf1.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=False)