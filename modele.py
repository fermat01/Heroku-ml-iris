
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
df = pd.read_csv('D:\Documents\Document2\Projets\quadra\streamlitAppli\iris.csv',sep=',')
features=df[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']].values
labels = df['class'].values
X_train,X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)
# Arbre de decision 
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)



#Support Vector Machine
svm=SVC()
svm.fit(X_train, y_train) 
##############  Modele avec pickle ############################

import pickle
from joblib import dump, load
modele1=dump(svm, 'modelSvm.pkl')
modele2=dump(dtc, 'modelDtc.pkl')
