#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[81]:


df = pd.read_csv('healthcare-dataset-stroke-data.csv')


# In[82]:


df.head()


# In[83]:


df.describe()


# In[84]:


df.drop("id", axis=1, inplace=True)
df.info()


# In[85]:


df["ever_married"].value_counts()


# In[86]:


df["work_type"].value_counts()


# In[87]:


df["Residence_type"].value_counts()


# In[88]:


df["smoking_status"].value_counts()


# In[89]:


df.groupby("stroke").mean(numeric_only=True)


# ## 處理缺失數據

# In[90]:


df.isnull().sum().sort_values(ascending=False)


# df.groupby("gender")["bmi"].transform("mean")

# In[91]:


df["bmi"].fillna(df.groupby("gender")["bmi"].transform("mean"), inplace=True)
df.isnull().sum()


# ## 類別資料的處理

# In[92]:


df = pd.get_dummies(data=df, dtype=int, columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status" ])
df


# In[93]:


df.info()


# In[94]:


df.drop(["ever_married_No","Residence_type_Rural"], axis=1, inplace=True)
df


# In[95]:


df.corr()


# ## 特徵縮放 normalization 

# In[96]:


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

# In[17]:


X = df.drop("stroke", axis=1)
y = df['stroke']


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # 1. Logistic Regression

# In[19]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=700)
lr.fit(X_train, y_train)


# In[20]:


predictions = lr.predict(X_test)
predictions


# In[21]:


from sklearn.metrics import accuracy_score
accuracy_using_decision_tree = round(accuracy_score(y_test, predictions)*100, 2)
print("Model accuracy using Decision Tree: ", accuracy_using_decision_tree, "%")


# In[22]:


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[23]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # 2. Decision Tree Classifier

# In[24]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# In[25]:


X_test


# In[26]:


predictions = dt.predict(X_test)
predictions


# In[27]:


print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[28]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # 3. Averaged Perceptron

# In[29]:


from sklearn.linear_model import Perceptron 
pt = Perceptron(max_iter=100, eta0=0.1, random_state=42)
pt.fit(X_train, y_train)


# In[30]:


predictions = pt.predict(X_test)
predictions


# In[31]:


print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[32]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # 4. Random Forest Classification

# In[33]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)


# In[34]:


predictions = rf.predict(X_test)
predictions


# In[35]:


print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[36]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # 5. Support Vector Machine

# In[37]:


from sklearn import svm
svm = svm.SVC(random_state=42)
svm.fit(X_train, y_train)


# In[38]:


predictions = svm.predict(X_test)
predictions


# In[39]:


print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[40]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # 6. Neural Networks

# In[41]:


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', 
                    alpha=1e-5,
                    hidden_layer_sizes=(6,), 
                    random_state=42,
                    max_iter=800)

clf.fit(X_train, y_train)   


# In[42]:


predictions = clf.predict(X_test)
predictions


# In[43]:


print("Accuracy:", round(accuracy_score(y_test, predictions)*100, 2),"%")
print("Recall:", recall_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))


# In[44]:


pd.DataFrame(confusion_matrix(y_test, predictions),
            columns = ['Predicted Not Stroke', 'Predicted Stroke'],
             index = ['True not Stroke', 'True Stroke']
            )


# # Model Export

# In[45]:


import joblib
joblib.dump(lr, "Stroke-LR.pkl", compress=3)
joblib.dump(dt, "Stroke-DT.pkl", compress=3)
joblib.dump(pt, "Stroke-PT.pkl", compress=3)
joblib.dump(rf, "Stroke-RF.pkl", compress=3)
joblib.dump(svm, "Stroke-SVM.pkl", compress=3)
joblib.dump(clf, "Stroke-CLF.pkl", compress=3)


# In[ ]:




