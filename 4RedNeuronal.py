# -*- coding: utf-8 -*-
"""
Created on Wed May 26 19:09:37 2021

@author: Alexander Humberto Nina P. 5950236
"""

import pandas as pd
data=pd.read_csv('c:/Users/aaale/Desktop/audit_risk.csv')

#remover id's
mat_feat=data.iloc[:,:-1].values
mat_class=data.iloc[:,-1].values

#hay nans en el data así que se imputa
from sklearn.impute import SimpleImputer
import numpy as np
imp=SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(mat_feat[:,:])
mat_feat[:,:]=imp.transform(mat_feat[:,:])

#separa para entrenar y para prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(mat_feat, mat_class, test_size=0.2, random_state=0)

#escalado
from sklearn.preprocessing import StandardScaler
escala=StandardScaler()
X_train=escala.fit_transform(X_train)
X_test=escala.fit_transform(X_test)

#instanciar ann
import tensorflow as tf
ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#segunda capa oculta
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#capa de salida
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#entrena la red neuronal
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y_train, batch_size=32, epochs=100)

#pruebas
y_pred=ann.predict(X_test)
y_pred=(y_pred>0.5)
#np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1))1))

#matriz de confusión
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))