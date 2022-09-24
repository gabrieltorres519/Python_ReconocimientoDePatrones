# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:35:29 2022

@author: Inspiron 7568
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  sklearn.impute import SimpleImputer # Rellenar datos faltantes con la media
from sklearn.preprocessing import LabelEncoder # Convertir variables categóricas en números
from sklearn.compose import ColumnTransformer
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandartScaler

dataset = pd.read_csv("Salary_Data.csv");

#imputer = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None, verbose=0, copy=True, add_indicator=False)# Rellenar datos faltantes con la media


X = dataset.iloc[:,:-1].values; # Años de experiencia


Y = dataset.iloc[:,-1]; # Salario

#labelenconder_X = LabelEncoder(); #Creando codificador de datos 

#X[:,0] = labelenconder_X.fit_transform(X[:,0]); # Convertir variables categóricas en números

#imputer.fit(X[:, 1:3]);

#X[:, 1:3] = imputer.transform(X[:, 1:3]);

#Country Column
#OneHotEncoder = sklearn.preprocessing.OneHotEncoder;
#transformers = [('Country', OneHotEncoder(), [0])];
#ct = ColumnTransformer(transformers, remainder='passthrough');
#X= ct.fit_transform(X);

#Dividiendo conjunto de entrenamiento y conjunto de testing
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

#Dado que no hace falta escalar las variables para la regresión lineal se han comentado las líneas que lo hacían

#Escalado de variables (primero convertimos las variables categoricas en la columna 'Purchased' a valores)

#labelenconder_X2 = LabelEncoder(); #Creando codificador de datos 

#X_train[:,5] = labelenconder_X2.fit_transform(X_train[:,5]); # Convertir variables categóricas en números

#X_test[:,5] = labelenconder_X2.fit_transform(X_test[:,5]); # Convertir variables categóricas en números

#sc_X=sklearn.preprocessing.StandardScaler()
#X_train=sc_X.fit_transform(X_train)
#X_test=sc_X.transform(X_test)


### Regresión Lineal Simple (Mínimos cuadrados)
# Entrenamiento de regresión lineal con el set de entrenamiento
from sklearn.linear_model import LinearRegression
regresion = LinearRegression(); #Ya escala los datos
regresion.fit(X_train,y_train);

#Resultado con el set de prueba
y_pred= regresion.predict(X_test)

#Visualizando los resultados en el entrenamiento
plt.scatter(X_train, y_train, color='red') #En rojo los datos de entrenamiento
plt.plot(X_train, regresion.predict(X_train), color='blue') #Azul la salida de la regresión
plt.title('Salario vs Experiencia (Set de entrenamiento)')
plt.xlabel('Años de experiencia')
plt.ylabel('Salario')
plt.show()

#Visualizando los resultados 
plt.scatter(X_test, y_test, color='red') #En rojo los datos de entrenamiento
plt.plot(X_train, regresion.predict(X_train), color='blue') #Azul la salida de la regresión
plt.title('Salario vs Experiencia (Set de prueba)')
plt.xlabel('Años de experiencia')
plt.ylabel('Salario')
plt.show()

