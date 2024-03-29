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




def stroke_predict(gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status):
    data = [[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]]
    df_test = pd.DataFrame(data, columns=["gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","smoking_status"])

    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df.drop(["id","stroke"], axis=1, inplace=True)
    df_test["bmi"].fillna(df.groupby("gender")["bmi"].transform("mean"), inplace=True)
    df_test = pd.get_dummies(data=df_test, dtype=int, columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status" ])

    df_c = pd.get_dummies(data=df, dtype=int, columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status" ]).columns
    df_f = pd.DataFrame()

    for c in df_c:
        if(c in df_test):
            df_f.loc[0, c] = df_test.loc[0, c]
        else:
            df_f.loc[0, c] = 0
    
    # print(df_f)
    df_f.drop(["gender_Female","work_type_Never_worked","work_type_children","ever_married_No","Residence_type_Rural","smoking_status_Unknown", "smoking_status_never smoked"], axis=1, inplace=True)

    predictions1 = lr_pretrained.predict(df_f)
    predictions2 = dt_pretrained.predict(df_f)
    predictions3 = pt_pretrained.predict(df_f)
    predictions4 = rf_pretrained.predict(df_f)
    predictions5 = svm_pretrained.predict(df_f)
    predictions6 = clf_pretrained.predict(df_f)

    print("Logistic Regression: ", predictions1[0])
    print("Decision Tree Classifier: ", predictions2[0])
    print("Averaged Perceptron: ", predictions3[0])
    print("Random Forest Classification: ", predictions4[0])
    print("Support Vector Machine: ", predictions5[0])
    print("Neural Networks: ", predictions6[0])
    return [predictions1[0], predictions2[0], predictions3[0], predictions4[0], predictions5[0], predictions6[0]]
# %%
#stroke_predict("Female",1.24,0,0,"No","children",'Urban',84.2,19.2,"never smoked")