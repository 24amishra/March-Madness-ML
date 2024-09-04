## Decision Tree for MM 
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score







x_training_data = pd.read_excel('/Users/agastyamishra/Downloads/projetcs/Book2.xlsx', sheet_name= 'Sheet1', header = 0,
names = [	'Pace',	'ORtg',	'FTr',	'3PAr',	'TS%',	'TRB%',	'AST%',	'STL%',	'BLK%',	'eFG%',	'TOV%',	'ORB%','FT/FGA','Pace2',	'ORtg2',	'FTr2','3PAr2',	'TS%2'	,'TRB%2',	'AST%2',	'STL%2',	'BLK%2',	'eFG%2',	'TOV%2',	'ORB%2',	'FT/FGA2','SRS','Drat'])
#x_data = x_data.drop(columns = ['Srs'])
y_training_data = pd.read_excel('/Users/agastyamishra/Downloads/projetcs/Book1.xlsx', sheet_name= 'Sheet1', header = 0,
names = ['Final Four'])
x_training_data= x_training_data.dropna(axis = 1)

x= x_training_data.values
y = y_training_data.values[1:150] 



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=7 )
scaler = MinMaxScaler(feature_range=(0,1))
# # # #print(y_test)
escaled_x_train = scaler.fit_transform(x_train)
#rescaled_x_train[:5]

x_data = pd.read_excel('/Users/agastyamishra/Downloads/projetcs/2024Data.xlsx', sheet_name= 'Sheet1', header = 0,
names = [	'Pace',	'ORtg',	'FTr',	'3PAr',	'TS%',	'TRB%',	'AST%',	'STL%',	'BLK%',	'eFG%',	'TOV%',	'ORB%','FT/FGA','Pace2',	'ORtg2',	'FTr2','3PAr2',	'TS%2'	,'TRB%2',	'AST%2',	'STL%2',	'BLK%2',	'eFG%2',	'TOV%2',	'ORB%2',	'FT/FGA2', 'SRS','Drat'])

#y_data = pd.read_excel('/Users/agastyamishra/Downloads/projetcs/2024Data.xlsx', sheet_name= 'Sheet2', header = 0,
#names = ['Final Four'])
x_test = x_data.values

clf = RandomForestClassifier(n_estimators = 100,max_features='sqrt', max_depth=None, min_samples_split= 2, random_state= 0)
clf.fit(x_train,y_train.ravel())
prob = clf.predict_proba(x_test)
pred = clf.predict(x_test)
print(prob,pred)


#scores = cross_val_score(clf, x_test, y_test, cv=10)
#print(scores.mean(),scores.std())




