#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('healthcare-dataset-stroke-data.csv')


# ## 資料探索

# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.drop("id", axis=1, inplace=True)
df.info()


# In[6]:


df["stroke"].value_counts()


# In[7]:


df["ever_married"].value_counts()


# In[8]:


df["work_type"].value_counts()


# In[9]:


df["Residence_type"].value_counts()


# In[10]:


df["smoking_status"].value_counts()


# In[11]:


df.groupby("stroke").mean(numeric_only=True)


# ## 處理缺失數據

# In[12]:


df.isnull().sum().sort_values(ascending=False)


# df.groupby("gender")["bmi"].transform("mean")

# In[13]:


df["bmi"].fillna(df.groupby("gender")["bmi"].transform("mean"), inplace=True)
df.isnull().sum()


# ## 類別資料的處理

# In[14]:


df = pd.get_dummies(data=df, dtype=int, columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status" ])
df


# In[15]:


df.info()


# In[16]:


df.corr()


# In[17]:


df.drop(["gender_Female","work_type_Never_worked","work_type_children","ever_married_No","Residence_type_Rural","smoking_status_Unknown", "smoking_status_never smoked"], axis=1, inplace=True)
df


# ## 特徵縮放 normalization 

# In[18]:


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

# In[19]:


X = df.drop("stroke", axis=1)
y = df['stroke']


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # 1. Logistic Regression

# In[21]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=600)
lr.fit(X_train, y_train)


# In[22]:


predictions = lr.predict(X_test)
predictions


# from sklearn.metrics import accuracy_score
# accuracy_using_decision_tree = round(accuracy_score(y_test, predictions)*100, 2)
# print("Model accuracy: ", accuracy_using_decision_tree, "%")

# In[23]:


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[24]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # 2. Decision Tree Classifier

# In[25]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# In[26]:


X_test


# In[27]:


predictions = dt.predict(X_test)
predictions


# In[28]:


print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[29]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # 3. Averaged Perceptron

# In[30]:


from sklearn.linear_model import Perceptron 
pt = Perceptron(max_iter=100, eta0=0.1, random_state=42)
pt.fit(X_train, y_train)


# In[31]:


predictions = pt.predict(X_test)
predictions


# In[32]:


print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[33]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # 4. Random Forest Classification

# In[34]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)


# In[35]:


predictions = rf.predict(X_test)
predictions


# In[36]:


print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[37]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # 5. Support Vector Machine

# In[38]:


from sklearn import svm
svm = svm.SVC(random_state=42)
svm.fit(X_train, y_train)


# In[39]:


predictions = svm.predict(X_test)
predictions


# In[40]:


print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[41]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # 6. Neural Networks

# In[42]:


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', 
                    alpha=1e-5,
                    hidden_layer_sizes=(6,), 
                    random_state=42,
                    max_iter=900)

clf.fit(X_train, y_train)   


# In[43]:


predictions = clf.predict(X_test)
predictions


# In[44]:


print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[45]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # Model Export

# In[47]:


import joblib
joblib.dump(lr, "Stroke-LR.pkl", compress=3)
joblib.dump(dt, "Stroke-DT.pkl", compress=3)
joblib.dump(pt, "Stroke-PT.pkl", compress=3)
joblib.dump(rf, "Stroke-RF.pkl", compress=3)
joblib.dump(svm, "Stroke-SVM.pkl", compress=3)
joblib.dump(clf, "Stroke-CLF.pkl", compress=3)


# In[ ]:




