#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import pandas as pd


# In[53]:


lr_pretrained = joblib.load("Stroke-LR.pkl")
dt_pretrained = joblib.load("Stroke-DT.pkl")
pt_pretrained = joblib.load("Stroke-PT.pkl")
rf_pretrained = joblib.load("Stroke-RF.pkl")
svm_pretrained = joblib.load("Stroke-SVM.pkl")
clf_pretrained = joblib.load("Stroke-CLF.pkl")

data = [[10434,"Female",69,0,0,"No","Private","Urban",94.39,22.8,"never smoked"]]
df_test = pd.DataFrame(data, columns=["id","gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level",'bmi',"smoking_status"])

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.drop(["id","stroke"], axis=1, inplace=True)
df_test.drop("id", axis=1, inplace=True)
df_test["bmi"].fillna(df_test.groupby("gender")["bmi"].transform("mean"), inplace=True)
df_test = pd.get_dummies(data=df_test, dtype=int, columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status" ])

df_c = pd.get_dummies(data=df, dtype=int, columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status" ]).columns
df_t = pd.DataFrame(columns=df_c)
df_f = pd.DataFrame()

for c in df_t:
    if(c in df_test):
        df_f.loc[0, c] = df_test.loc[0, c]
    else:
       df_f.loc[0, c] = 0
   
# print(df_f)
df_f.drop(["ever_married_No","Residence_type_Rural"], axis=1, inplace=True)


predictions1 = lr_pretrained.predict(df_f)
predictions2 = dt_pretrained.predict(df_f)
predictions3 = pt_pretrained.predict(df_f)
predictions4 = rf_pretrained.predict(df_f)
predictions5 = svm_pretrained.predict(df_f)
predictions6 = clf_pretrained.predict(df_f)

print(predictions1, predictions2, predictions3, predictions4, predictions5, predictions6)
# %%
