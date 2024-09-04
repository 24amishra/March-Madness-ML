
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import svm
x_data = pd.read_excel('/Users/agastyamishra/projetcs/Book2.xlsx', sheet_name= 'Sheet1', header = 0,
names = ['Srs','SRS','SOS',	'Pace',	'ORtg',	'FTr',	'3PAr',	'TS%',	'TRB%',	'AST%',	'STL%',	'BLK%',	'eFG%',	'TOV%',	'ORB%','FT/FGA','Pace2',	'ORtg2',	'FTr2','3PAr2',	'TS%2'	,'TRB%2',	'AST%2',	'STL%2',	'BLK%2',	'eFG%2',	'TOV%2',	'ORB%2',	'FT/FGA2', 'DRtg','NRtg'])
x_data = x_data.drop(columns = ['Srs'])
y_data = pd.read_excel('/Users/agastyamishra/projetcs/Book1.xlsx', sheet_name= 'Sheet1', header = 0,
names = ['Final Four'])
x_data = x_data.dropna(axis = 1)



x = x_data.values
y = y_data.values[1:89] 
clf = svm.SVC(decision_function_shape='ovr') 
clf.fit(x,y.ravel())
#pred = clf.decision_function([[13.98,6.84,68.9,114.3,0.307,0.342,0.579,52,50.6,10.5,8.4,0.543,14,31.8,0.241,68.9,103.9,0.24,0.385,0.539,48,54.4,8.5,8.6,0.512,15.8,28.8,0.178,98.47,19.94]])
pred = clf.decision_function(x)
print(pred)


