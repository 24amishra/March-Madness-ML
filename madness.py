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

x_data = pd.read_excel('/Users/agastyamishra/projetcs/Book2.xlsx', sheet_name= 'use', header = 0,
names = ['Srs','SRS','SOS',	'Pace',	'ORtg',	'FTr',	'3PAr',	'TS%',	'TRB%',	'AST%',	'STL%',	'BLK%',	'eFG%',	'TOV%',	'ORB%','FT/FGA','Pace2',	'ORtg2',	'FTr2','3PAr2',	'TS%2'	,'TRB%2',	'AST%2',	'STL%2',	'BLK%2',	'eFG%2',	'TOV%2',	'ORB%2',	'FT/FGA2', 'DRtg','NRtg'])
x_data = x_data.drop(columns = ['Srs'])
y_data = pd.read_excel('/Users/agastyamishra/projetcs/Book1.xlsx', sheet_name= 'Sheet1', header = 0,
names = ['Final Four'])
x_data = x_data.dropna(axis = 1)



x = x_data.values
y = y_data.values[1:89] 



correlation = x_data.corr()
# #print(y.dtype)

plt.figure(dpi=130)
sns.heatmap(x_data.corr(), annot=True, fmt= '.2f')

plt.show()


# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42 )

# scaler = MinMaxScaler(feature_range=(0,1))
# # # #print(y_test)
# rescaled_x_train = scaler.fit_transform(x_train)
# rescaled_x_train[:5]

# model = LogisticRegression().fit(x_train,y_train.ravel())
# pred = model.predict(x_test)


# acc = f1_score(y_test,pred)
# print(acc)





           
