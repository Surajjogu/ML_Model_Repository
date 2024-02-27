# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:14:05 2024

@author: kalpavruksh_sjo
"""

from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    output = prediction[0]
    
    return render_template('index.html', prediction_text='Loan Eligibility = {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
