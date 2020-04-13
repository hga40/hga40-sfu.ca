#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 22:18:02 2020

@author: humayun_gazi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
#Loading the Dataset
dataset = pd.read_csv("diet.csv")

#Dropping features with nul values
dataset = dataset.drop(['SEQN','DRQSDT1', 'DRQSDT2', 'DRQSDT3','DRQSDT4','DRQSDT5','DRQSDT6',
      'DRQSDT7','DRQSDT8','DRQSDT9','DRQSDT10','DRQSDT11','DRQSDT12',
       'DRQSDT91','DRD370TQ', 'DRD370UQ','DRD370SQ','DRD370RQ','DRD370QQ',
       'DR1SKY','DR1TATOA','DR1TB12A','DR1TTHEO','DR1TALCO','DR1TP184',
       'DR1.320Z','DR1.330Z','DR1BWATZ','DRD350A','DRD350AQ','DRD350B',
       'DRD350BQ','DRD350F','DRD350FQ','DRD350G','DRD350GQ','DRD350H',
       'DRD350HQ','DRD350I','DRD350IQ','DRD350J','DRD350JQ','DRD350K',
       'DRD370A','DRD370AQ','DRD370B','DRD370BQ','DRD370C','DRD370CQ',
       'DRD370D','DRD370DQ','DRD370E','DRD370EQ','DRD370F','DRD370FQ',
       'DRD370G','DRD370GQ','DRD370H','DRD370HQ','DRD370I','DRD370IQ',
       'DRD370J','DRD370JQ','DRD370K','DRD370JQ','DRD370L','DRD370LQ',
       'DRD370M','DRD370MQ','DRD370N','DRD370NQ','DRD370O','DRD370OQ',
       'DRD370P','DRD370PQ','DRD370Q','DRD370R','DRD370S','DRD370T',
       'DRD370U','DRD370V','DRD370KQ','DRD350EQ','DRD350E','DRD350DQ',
       'DRD350D','DRD350CQ','DRD350C','DBD100','DRD340','DRD360','WTDR2D'], axis = 1)


#Cleaning the data
dataset.sort_values(by ='WTDRD1', ascending=True, inplace=True)
dataset.reset_index(drop=True, inplace=True)
dataset.drop(dataset.index[0:1152], inplace=True)
dataset.reset_index(drop=True, inplace=True)
dataset.interpolate(method='linear', limit_direction='both', axis = 1, inplace=True)
dataset.sample(frac=1)

#Checking missing data
missing_data = dataset.isnull().sum()


#Scalling the Dataset
normalized_df=(dataset-dataset.min())/(dataset.max()-dataset.min())
print(normalized_df)

#Looking for linearity between the dependant and independant features
plot1 = plt.figure(1)
corr = normalized_df.corr()
heatmap = sns.heatmap(corr)
plt.savefig('heatmap.png', dpi=800)

#Looking at the lINEAR realtionships between x and y
plot2 = plt.figure(2)
plt.scatter(dataset['DR1TKCAL'], dataset['WTDRD1'], color='orange')
plt.title('DR1TKCAL vs WTDRD1', fontsize=14)
plt.xlabel('DR1TKCAL', fontsize=14)
plt.ylabel('WTDRD1', fontsize=14)
plt.grid(True)
plt.savefig('plot_2.png', dpi=400)

plot3 = plt.figure(3)
plt.scatter(dataset['DBQ095Z'], dataset['WTDRD1'], color='red')
plt.title('DBQ095Z vs WTDRD1', fontsize=14)
plt.xlabel('DBQ095Z', fontsize=14)
plt.ylabel('WTDRD1', fontsize=14)
plt.grid(True)
plt.savefig('plot_3.png', dpi=400)

plot4 = plt.figure(4)
plt.scatter(dataset['DR1TCARB'], dataset['WTDRD1'], color='yellow')
plt.title('DR1TCARB vs WTDRD1', fontsize=14)
plt.xlabel('DR1TCARB', fontsize=14)
plt.ylabel('WTDRD1', fontsize=14)
plt.grid(True)
plt.savefig('plot_4.png', dpi=400)

plot5 = plt.figure(5)
plt.scatter(dataset['DRQSDIET'], dataset['WTDRD1'], color='blue')
plt.title('DRQSDIET vs WTDRD1', fontsize=14)
plt.xlabel('DRQSDIET', fontsize=14)
plt.ylabel('WTDRD1', fontsize=14)
plt.grid(True)
plt.savefig('plot_5.png', dpi=400)


#Setting Independant and Dependant variables
X = normalized_df.drop(['WTDRD1'], axis=1)
y = np.ravel(normalized_df[['WTDRD1']])


#Splitting of Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#Random Forrest Regression
model_rf = RandomForestRegressor(n_estimators=10, random_state=0)
model_rf.fit(X_train, y_train)

#model_rf.show(view='Tree', tree_id=0)
RF_predictions = model_rf.predict(X_test)
print("Random Forest Predictions:\n%s" % model_rf.predict(X_test))
#print("Random Forest Results:\n%s" % model_rf.evaluate(X_test))
print("Random Forest Score:\n%s" % model_rf.score(X_train, y_train))

#RMSE of RF Model
rms4 = np.sqrt(mean_squared_error(y_test, RF_predictions))
print('RF_RMSE: \n', rms4)

#R2 Score
rf_r_squared = r2_score(y_test, RF_predictions)
print("RF R-Squared Score:\n%s" % rf_r_squared)

#Saving the RF Model
joblib.dump(model_rf,'Random_Forest_model')

#Loading the RF Model
New_RF_Model = joblib.load('Random_Forest_model')
New_RF_Model.predict(X_test)

#Making our Prediction using Linear Regression
regr_ols = linear_model.LinearRegression()
regr_sgd = linear_model.SGDRegressor(max_iter=1000, power_t = 0.01, eta0 = 0.01, tol = 1e-3)
MLP_regr = MLPRegressor(max_iter=10000,
                             hidden_layer_sizes=(100,), 
                             activation='relu',
                             solver='adam',)

#Fitting the model
model_lr_ols = regr_ols.fit(X_train, y_train)
model_lr_sgd = regr_sgd.fit(X_train, y_train)
MLP_lr = MLP_regr.fit(X_train, y_train)

#Predicting the test set results
predictions_ols = regr_ols.predict(X_test)
predictions_sgd = regr_sgd.predict(X_test)
MLP_predictions = MLP_lr.predict(X_test)

print('Linear Regression Predictions_ols: \n', predictions_ols)
print('Linear Regression Predictions_sgd: \n', predictions_sgd)
print('Linear Regression Predictions with MLP: \n', MLP_predictions)

print('Intercept_ols: \n', regr_ols.intercept_)
print('Intercept_sgd: \n', regr_sgd.intercept_)

print('Coefficients_ols: \n', regr_ols.coef_)
print('Coefficients_sgd: \n', regr_sgd.coef_)

modlist = [model_lr_ols, model_lr_sgd, MLP_lr]

#Plotting the Linear Regression Model
plot6 = plt.figure(6)
plt.scatter(y_test, predictions_sgd)
plt.title('Plotting the Model', fontsize=14)
plt.xlabel('True Values', fontsize=10)
plt.ylabel('Predictions', fontsize=10)
plt.savefig('Model_Plot.png', dpi=400)

#RMSE Values of Linear Regression Models
rms = np.sqrt(mean_squared_error(y_test, predictions_ols))
print('RMSE_ols: \n', rms)
rms2 = np.sqrt(mean_squared_error(y_test, predictions_sgd))
print('RMSE_sgd: \n', rms2)
rms3 = np.sqrt(mean_squared_error(y_test, MLP_predictions))
print('MLP_RMSE: \n', rms3)

#R Squared Score of Linear Regression
r_squared = r2_score(y_test, predictions_ols)
print("OLS R Squared Score:\n%s" % r_squared)
r_squared_2 = r2_score(y_test, predictions_sgd)
print("SGD R Squared Score:\n%s" % r_squared)
r_squared_3 = r2_score(y_test, MLP_predictions)
print("MLP R Squared Score:\n%s" % r_squared_3)

#Cross Validation
print("Cross_Val_Predictions_OLS:\n%s" % cross_val_predict(regr_ols, X, y, cv=5))
print("Cross_Val_Predictions_SGD:\n%s" % cross_val_predict(regr_sgd, X, y, cv=5))
print("Cross_Val_Predictions_MLP:\n%s" % cross_val_predict(MLP_regr, X, y, cv=5))

print("Cross_Val_score_OLS:\n%s" % cross_val_score(regr_ols, X, y, cv = 5))
print("Cross_Val_score_SGD:\n%s" % cross_val_score(regr_sgd, X, y, cv = 5))
print("Cross_Val_score_SGD:\n%s" % cross_val_score(MLP_regr, X, y, cv = 5))

#Saving the Scikit-Learn Models
joblib.dump(modlist, 'Scikit-Learn_models')

modlist_loaded = joblib.load('Scikit-Learn_models')

model_1 = modlist_loaded[0]
model_2 = modlist_loaded[1]
model_3 = modlist_loaded[2]

print(model_1.predict(X_test))
print(model_2.predict(X_test))
print(model_3.predict(X_test))

#Keras Regression Model
keras_model = Sequential([
    Dense(77, input_shape=(76,), activation='relu'),
    Dense(39, activation='relu'),
    Dense(1, activation=None)
])

keras_model.summary()

keras_model.compile(Adam(lr=0.0001), loss='mse', metrics=['mse'])

history = keras_model.fit(X_train, y_train, epochs=150, batch_size=100, verbose=1, validation_split=0.1, shuffle=True)

print(history.history.keys())
# "Loss"
plot7 = plt.figure(7)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('Loss.png', dpi=400)

Keras_Predictions = keras_model.predict(X_test, batch_size = 100, verbose=0)

for i in Keras_Predictions:
    print (i)
    
#Keras R-Squared Score
r_squared_4 = r2_score(y_test, Keras_Predictions)
print(r_squared_4)

#Keras RMSE
rms5 = np.sqrt(mean_squared_error(y_test, Keras_Predictions))
print('Keras_RMSE: \n', rms5)

#Saving the model
keras_model.save("keras_model.h5")
print("Saved model to disk")

from keras.models import load_model
new_keras_model = load_model("keras_model.h5")

new_keras_model.summary()

#Generating Keras Model Weights
new_keras_model.get_weights()

new_keras_model.optimizer

