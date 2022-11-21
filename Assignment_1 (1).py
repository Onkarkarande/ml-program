# uber data prediction

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import  mean_squared_error

import warnings 
warnings.filterwarnings("ignore")
#%% ...
df=pd.read_csv(r"C:\Users\rpsv1\Desktop\uber.csv")
print(df.head())
#%%   DATA PREPROCESSING 
# 1. REMOVE NULL VALUE 
print(df.isnull().sum())
#drop the null values
df.dropna()
print(df.isnull().sum())

#%%
#2. CHECK DUPLICATE and REMOVE
print(df.duplicated().sum())

#print(df.drop_duplicates())
#%%
#EDA 
print(df.info())
print(df.describe())    
print(df.columns)

#%%

df.pickup_datetime=pd.to_datetime(df.pickup_datetime)

df['year'] = df.pickup_datetime.dt.year
df['month'] = df.pickup_datetime.dt.month
df['weekday'] = df.pickup_datetime.dt.weekday
df['hour'] = df.pickup_datetime.dt.hour
print(df.head())
#%% DROP UNNESSESARY COLUMNS

df = df.drop(['Unnamed: 0','key'], axis='columns')
df = df.drop(['pickup_datetime','month', 'hour',], axis='columns')
#%%
sns.heatmap(df.corr(),annot=True)
#%%
print(df.columns)
#%%
target = 'fare_amount'
features = ['pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'year',
       'weekday']

#%%
##OUTERLIERS DETECTION 
sns.distplot(df['fare_amount'])
#%%

#inter quartile range (iqr) 

df1=df.copy()

for i in features:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    df = df[df[i] <= (Q3+(1.5*IQR))]
    df = df[df[i] >= (Q1-(1.5*IQR))]
   
print(df.head())
print('\nBefore removal of outliers, The dataset had {} samples.'.format(df1.shape[0]))
print('After removal of outliers, The dataset now has {} samples.'.format(df.shape[0]))
#%%
#corr r2
#creating a correlation matrix

corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()
#%% 

#Spli the data into Training and testing set 

X = df[features]
Y = df[target]
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y, train_size=0.8, random_state=42)
#%%
regr = LinearRegression()
regr.fit(Train_X,Train_Y)
print("Linear Regression score R^2 Score ")

#Root Mean Square Error
print(regr.score(Test_X, Test_Y))
y_pred_lr = regr.predict(Test_X)
lr_mse = np.sqrt(mean_squared_error(y_pred_lr, Test_Y))
print("RMSE value for Linear regression is:", lr_mse) #root means square deviation

#%%
rfr = RandomForestRegressor(n_estimators = 20, random_state = 101)
rfr.fit(Train_X,Train_Y)
print("RandomForestRegressor Regression score R^2 Score ")
rfr.score(Test_X, Test_Y)
y_pred_rfr = rfr.predict(Test_X)
rfr_mse = np.sqrt(mean_squared_error(y_pred_rfr, Test_Y))
print("RMSE Value for Random Forest Regression is:", rfr_mse)
#%%

