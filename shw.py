import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from joblib import dump, load
#import plotly.graph_objects as gost.title('Iris')

# Add a title
st.title('Construction de classifieur à partir de données iris ')
df = pd.read_csv('D:\Documents\Document2\Projets\quadra\streamlitAppli\iris.csv',sep=',')
if st.checkbox('Show dataframe'):
    st.write(df)
    st.subheader('Scatter plot')
species = st.multiselect('Show iris per variety?', df['class'].unique())
col1 = st.selectbox('Quel feature dans  x?', df.columns[0:4])
col2 = st.selectbox('Quel feature dans y?', df.columns[0:4])
new_df = df[(df['class'].isin(species))]
st.write(new_df)
# create figure using plotly express
fig = px.scatter(new_df, x =col1,y=col2,color='class')
# Plot!
st.plotly_chart(fig)
st.subheader('Histogramme')
feature = st.selectbox('Quel feature?', df.columns[0:4])
# Filter dataframe
new_df2 = df[(df['class'].isin(species))][feature]
fig2 = px.histogram(new_df, x=feature, marginal="rug")
st.plotly_chart(fig2)
##############################################################
# Modèle de machine learning                                 #
##############################################################
st.subheader('Modèles de machine Learning ')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
jobSvm=load('D:\Documents\Document2\Projets\quadra\streamlitAppli\modelSvm.pkl')
jobDtc=load('D:\Documents\Document2\Projets\quadra\streamlitAppli\modelDtc.pkl')
features=df[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']].values
labels = df['class'].values
X_train,X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)
algo =['Decision Tree', 'Support Vector Machine']
#print(jobSvm.predict(X_test))   #controle de la sortie 
classifier = st.selectbox('Quel algorithme?', algo)
if classifier=='Decision Tree':
    acc = jobDtc.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_dtc = jobDtc.predict(X_test)
    cm_dtc=confusion_matrix(y_test,pred_dtc)
    st.write('Matrice de confusion: ', cm_dtc)
elif classifier == 'Support Vector Machine':
    acc1 = jobSvm.score(X_test, y_test)
    st.write('Accuracy: ', acc1)
    pred_svm = jobSvm.predict(X_test)
    cm=confusion_matrix(y_test,pred_svm)
    st.write('Matrice de confusion: ', cm)

