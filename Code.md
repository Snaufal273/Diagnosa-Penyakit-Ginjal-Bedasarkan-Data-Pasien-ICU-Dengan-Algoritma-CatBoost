# Diagnosa-Penyakit-Ginjal-Dengan-Algoritma-CatBoost

# ImportLibraries
!pip install catboost
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score,precision_score,f1_score
from sklearn.metrics import roc_curve,roc_auc_score
from catboost import CatBoostClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics

# Data Extraction
df=pd.read_csv("../data/data01.csv")
m = df.shape[0]
n = df.shape[1]

print("Number of rows: " + str(m))
print("Number of columns: " + str(n))

# Visualisasi Data
sns.countplot(df['Renal failure'])

df["Renal failure"].value_counts()

# Data Split
def train_test(df):
    Y_train = df['Renal failure']
    # pembagian data 90/10
    X_train, X_val, Y_train, Y_val = train_test_split(df, Y_train, test_size=0.1, stratify=Y_train, random_state=42) 
    X_train = X_train.drop(["Renal failure"], axis=1)
    X_val =X_val.drop(["Renal failure"], axis=1)
    print(X_train.shape,X_val.shape)
    return X_train, X_val, Y_train, Y_val
		
# Data Cleaning
def drop_table(df, table):
    df = df.drop(table, axis=1)
    return df
		
df = drop_table(df, ['ID']) 

# Mendetect numerik dan kategori
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']

# Fill Missing Value
def impute_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)
		
for col in num_cols:
    impute_mode(col)

# Hasil Data Split
X_train, X_val, Y_train, Y_val = train_test(df)

# Algoritma Decision Tree
def DTC(X_train, X_val, Y_train):  
    clf = DecisionTreeClassifier()
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_val)
    y_pred_proba = clf.predict_proba(X_val)[:,1]
    
    return y_pred, y_pred_proba
		
dtc_pred, dtc_pred_proba = DTC(X_train, X_val, Y_train)

print("Accuracy:",accuracy_score(Y_val, dtc_pred))
print("Precision:",precision_score(Y_val, dtc_pred))
print("Recall:",recall_score(Y_val, dtc_pred))
print("F1 score:",f1_score(Y_val, dtc_pred))
print("AUC score:",roc_auc_score(Y_val, dtc_pred_proba))

# Algoritma Random Forest
def RF(X_train, X_val, Y_train):  
    clf = RandomForestClassifier()
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_val)
    y_pred_proba = clf.predict_proba(X_val)[:,1]
    
    return y_pred, y_pred_proba
	
rf_pred, rf_pred_proba = RF(X_train, X_val, Y_train)

print("Accuracy:",accuracy_score(Y_val, rf_pred))
print("Precision:",precision_score(Y_val, rf_pred))
print("Recall:",recall_score(Y_val, rf_pred))
print("F1 score:",f1_score(Y_val, rf_pred))
print("AUC score:",roc_auc_score(Y_val, rf_pred_proba))
	
# Algoritma CatBoost
def catboost(X_train, X_val, Y_train):
    clf = CatBoostClassifier()
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_val)
    y_pred_proba = clf.predict_proba(X_val)[:,1]

    return y_pred, y_pred_proba
	
cat_pred, cat_pred_proba = catboost(X_train, X_val, Y_train)

print("Accuracy:",accuracy_score(Y_val, cat_pred))
print("Precision:",precision_score(Y_val, cat_pred))
print("Recall:",recall_score(Y_val, cat_pred))
print("F1 score:",f1_score(Y_val, cat_pred))
print("AUC score:",roc_auc_score(Y_val, cat_pred_proba))
