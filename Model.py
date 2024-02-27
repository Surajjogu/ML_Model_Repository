# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:00:36 2023

@author: kalpavruksh_sjo
"""

#import libraries related to data transformation and analysis
import pandas as pd

#import data
dataset = pd.read_csv("https://raw.githubusercontent.com/Surajjogu/ML_Model_Repository/main/loan_sanction_train.csv")

#View top 10 rows from dataset
dataset.head()

#total rows and column in dataset
dataset.shape

#delete null values
dataset.dropna(inplace = True)

# ----- Data Preparation -----------
#creating dummy variables for categorical variables 
GN_dummies = pd.get_dummies(dataset['Gender'],prefix="Gender",drop_first=True)
dataset = dataset.drop('Gender', axis = 1)

#include gender dummy variable in the dataset
dataset = dataset.join(GN_dummies)


#create dummy variables for other categorical variables
PA_dummies = pd.get_dummies(dataset['Property_Area'],prefix="PA",drop_first=True)

#drop property area column from dataset
dataset = dataset.drop('Property_Area', axis = 1)

#include Property area dummy variables in the dataset
dataset = dataset.join(PA_dummies)

#Scaling other categorical variables to match the magnitude of data with other numerical variables

dataset['Married'] = dataset['Married'].map({'Yes':1, 'No':2})
dataset['Self_Employed'] = dataset['Self_Employed'].map({'Yes':1, 'No':2})
dataset['Education'] = dataset['Education'].map({'Graduate':1, 'Not Graduate':2})
dataset['Loan_Status'] = dataset['Loan_Status'].map({'Y':1, 'N':0})
dataset['Dependents'] = dataset['Dependents'].map({'0': 0, '1':1, '2':2, '3+':3})

#delete the Loan_ID column
dataset = dataset.drop(['Loan_ID'], axis = 1)

#prepare the target vairable and other feature variables and assign them to variables
X = dataset.drop('Loan_Status',axis = 1)
y = dataset['Loan_Status']

#Split data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#import libraries for Random forest classifier model and other validation related libraries
from sklearn.ensemble import RandomForestClassifier
#From sklearn.metrics import confusion_matrix,accuracy_score,f1_score

#fit the data into the model

rfc = RandomForestClassifier(n_estimators=250, random_state=250)
rfc.fit(X_train,y_train)

#predict the result
y_pred = rfc.predict(X_test)


#dumping the model object into the Git repository
import pickle
import os

# Define the path to save the model within the Git repository
model_path = os.path.join("ML_Model_Repository", "model.pkl")

# Save the model to the defined path
pickle.dump(rfc, open(model_path, 'wb'))

print("Model saved to:", model_path)

# Uncomment the following line if you want to reload the model immediately
model = pickle.load(open(model_path, 'rb'))

#print(rfc.predict([['LP008000','Male','Yes',1,'Graduate','Yes',4000,0,100,180,1,'Rural']]))
print(model.predict([[1,1,1,1,4000,0,100,180,1,1,0,0]]))
