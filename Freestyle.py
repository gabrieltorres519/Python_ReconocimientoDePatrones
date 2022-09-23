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

dataset = pd.read_csv("Data.csv");

imputer = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None, verbose=0, copy=True, add_indicator=False)# Rellenar datos faltantes con la media

# Llenando una matriz con los datos del archivo dividiéndolos
X = dataset.values;

# Obteniéndo la última columna del set de datos en un array 
Y = dataset.iloc[:,3];

labelenconder_X = LabelEncoder(); #Creando codificador de datos 

X[:,0] = labelenconder_X.fit_transform(X[:,0]); # Convertir variables categóricas en números

imputer.fit(X[:, 1:3]);

X[:, 1:3] = imputer.transform(X[:, 1:3]);

#Country Column
OneHotEncoder = sklearn.preprocessing.OneHotEncoder;
transformers = [('Country', OneHotEncoder(), [0])];
ct = ColumnTransformer(transformers, remainder='passthrough');
X= ct.fit_transform(X);

#Dividiendo conjunto de entrenamiento y conjunto de testing
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)



