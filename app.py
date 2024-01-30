#!/usr/bin/env python
# coding: utf-8

# In[443]:


import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns


# In[444]:


df = pd.read_csv('healthcare-dataset-stroke-data.csv')


# In[445]:


df.head()


# In[446]:


df.describe()


# In[447]:


df.drop("id", axis=1, inplace=True)
df.info()


# In[448]:


df["ever_married"].value_counts()


# In[449]:


df["work_type"].value_counts()


# In[450]:


df["Residence_type"].value_counts()


# In[451]:


df["smoking_status"].value_counts()


# In[452]:


df.groupby("stroke").mean(numeric_only=True)


# ## 處理缺失數據

# In[453]:


df.isnull().sum().sort_values(ascending=False)


# df.groupby("gender")["bmi"].transform("mean")

# In[454]:


df["bmi"].fillna(df.groupby("gender")["bmi"].transform("mean"), inplace=True)
df.isnull().sum()


# ## 類別資料的處理

# In[455]:


df = pd.get_dummies(data=df, dtype=int, columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status" ])
df


# In[456]:


df.info()


# In[457]:


df.drop(["ever_married_No","Residence_type_Rural"], axis=1, inplace=True)
df.corr()


# ## 特徵縮放 normalization 

# In[458]:


from sklearn.preprocessing import MinMaxScaler
scal = MinMaxScaler()
for col_name in df.columns:
    if df[col_name].nunique() > 5: 
        df[col_name] = scal.fit_transform(df[[col_name]])
df


# from sklearn.preprocessing import LabelEncoder
# le=LabelEncoder()
# for col in df.columns:
#     if df[col].dtype=='object':
#         df[col]=le.fit_transform(df[col]) 
# df

# In[459]:


X = df.drop("stroke", axis=1)
y = df['stroke']


# In[460]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # 1. Logistic Regression

# In[461]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=700)
lr.fit(X_train, y_train)


# In[462]:


predictions = lr.predict(X_test)
predictions


# In[463]:


from sklearn.metrics import accuracy_score
accuracy_using_decision_tree = round(accuracy_score(y_test, predictions)*100, 2)
print("Model accuracy using Decision Tree: ", accuracy_using_decision_tree, "%")


# In[464]:


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[465]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # 2. Decision Tree Classifier

# In[466]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# In[467]:


X_test


# In[468]:


predictions = dt.predict(X_test)
predictions


# In[469]:


print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[470]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # 3. Averaged Perceptron

# In[471]:


from sklearn.linear_model import Perceptron 
pt = Perceptron(max_iter=100, eta0=0.1, random_state=42)
pt.fit(X_train, y_train)


# In[472]:


predictions = pt.predict(X_test)
predictions


# In[473]:


print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[474]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # 4. Random Forest Classification

# In[475]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)


# In[476]:


predictions = rf.predict(X_test)
predictions


# In[477]:


print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[478]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # 5. Support Vector Machine

# In[487]:


from sklearn import svm
svm = svm.SVC(random_state=42)
svm.fit(X_train, y_train)


# In[488]:


predictions = svm.predict(X_test)
predictions


# In[489]:


print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[490]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # 6. Neural Networks

# In[495]:


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', 
                    alpha=1e-5,
                    hidden_layer_sizes=(6,), 
                    random_state=42,
                    max_iter=800)

clf.fit(X_train, y_train)   


# In[496]:


predictions = clf.predict(X_test)
predictions


# In[497]:


print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[498]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # Model Export

# In[500]:


import joblib
joblib.dump(lr, "Stroke-LR.pkl", compress=3)
joblib.dump(dt, "Stroke-DT.pkl", compress=3)
joblib.dump(pt, "Stroke-PT.pkl", compress=3)
joblib.dump(rf, "Stroke-RF.pkl", compress=3)
joblib.dump(svm, "Stroke-SVM.pkl", compress=3)
joblib.dump(clf, "Stroke-CLF.pkl", compress=3)


# In[ ]:




