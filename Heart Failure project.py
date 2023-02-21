#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# ## 1. Importing and Exploring the dataset

# In[2]:


data=pd.read_csv("heart_failure_clinical_records_dataset.csv")


# In[3]:


data.head(10)


# In[4]:


data.shape


# In[5]:


type(data)


# In[6]:


categorical_variables=data[["anaemia","diabetes","high_blood_pressure","sex","smoking"]]
continuous_variables=data[["age","creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time"]]


# In[7]:


pd.set_option('display.max_rows',300)
data


# In[8]:


data.isna().sum()


# #### No null values

# In[9]:


data.isnull()


# In[10]:


continuous_variables.describe()


# In[11]:


data.groupby("DEATH_EVENT").count()


# In[12]:


print(data.duplicated().value_counts())


# In[13]:


plt.figure(figsize=(13,10))
for i,con in enumerate (continuous_variables):
    plt.subplot(3,3,i+1)
    sns.boxplot(data[con])
plt.show()


# ## 2. Visualization

# In[14]:


age=data[['age']]
platelets=data[["platelets"]]


# In[15]:


plt.figure(figsize=(13,7))
plt.scatter(platelets,age,c=data['DEATH_EVENT'],s=100,alpha=0.8)
plt.xlabel("Platelets",fontsize=20)
plt.ylabel("Age",fontsize=20)
plt.title("Visualizing the unbalanced data",fontsize=22)
plt.show()


# In[16]:


plt.figure(figsize=(13,7))
sns.heatmap(data.corr(),vmax=1,vmin=-1,annot=True)
plt.title("Relationship",fontsize=22)
plt.show()


# In[17]:


data.corr()


# In[18]:


plt.figure(figsize=(13,10))
for i,cat in enumerate(categorical_variables):
    plt.subplot(2,3,i+1)
    sns.countplot(data=data,hue="DEATH_EVENT",x=cat)
plt.show()


# In[19]:


plt.figure(figsize=(13,10))
plt.subplot(2,2,1)
sns.countplot(data=data,x='anaemia',hue="DEATH_EVENT")
plt.subplot(2,2,4)
sns.countplot(data=data,x='diabetes',hue="DEATH_EVENT")


# In[20]:


plt.figure(figsize=(13,10))
for j,con in enumerate(continuous_variables):
    plt.subplot(3,3,j+1)
    sns.histplot(data=data,hue="DEATH_EVENT",x=con,multiple='stack')
plt.show()


# In[21]:


plt.figure(figsize=(8,8))
sns.boxplot(data=data,x="sex",y="age",hue="DEATH_EVENT")
plt.title("The impact of Sex and Age on Death Event",fontsize=18)


# In[22]:


smokers=data[data["smoking"]==1]
non_smokers=data[data["smoking"]==0]

non_survived_smokers=smokers[smokers["DEATH_EVENT"]==1]
survived_smokers=smokers[smokers["DEATH_EVENT"]==0]
non_survived_non_smokers=non_smokers[non_smokers["DEATH_EVENT"]==1]
survived_non_smokers=non_smokers[non_smokers["DEATH_EVENT"]==0]

smoking_data=[len(non_survived_smokers),len(survived_non_smokers),len(non_survived_non_smokers),len(survived_smokers)]
smoking_labels=["non_survived_smokers","survived_non_smokers","non_survived_non_smokers","survived_smokers"]

plt.figure(figsize=(9,9))
plt.pie(smoking_data,labels=smoking_labels,autopct='%.1f%%',startangle=90)
circle=plt.Circle((0,0),0.7,color="white")
p=plt.gcf()
p.gca().add_artist(circle)
plt.title("Survival Status on Smoking",fontsize=18)
plt.show()


# In[23]:


male=data[data["sex"]==1]
female=data[data["sex"]==0]

non_survived_male=male[male["DEATH_EVENT"]==1]
survived_male=male[male["DEATH_EVENT"]==0]
non_survived_female=female[female["DEATH_EVENT"]==1]
survived_female=female[female["DEATH_EVENT"]==0]

sex_data=[len(non_survived_male),len(survived_male),len(non_survived_female),len(survived_female)]
sex_labels=["non_survived_male","survived_male","non_survived_female","survived_female"]

plt.figure(figsize=(9,9))
plt.pie(sex_data,labels=sex_labels,autopct='%.1f%%',startangle=90)
circle=plt.Circle((0,0),0.7,color="white")
p=plt.gcf()
p.gca().add_artist(circle)
plt.title("Survival Status on Sex",fontsize=18)
plt.show()


# ## 3. Data Modelling and Prediction 

# In[24]:


x=data[["age","creatinine_phosphokinase","ejection_fraction","serum_creatinine","time","serum_sodium"]]
y=data["DEATH_EVENT"]


# In[25]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)


# In[26]:


scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)


# In[27]:


accuracy_list=[]


# ### 3.1 : Logistic Regression

# In[28]:


lr_model=LogisticRegression()
lr_model.fit(x_train_scaled,y_train)
lr_prediction=lr_model.predict(x_test_scaled)
lr_accuracy=(round(accuracy_score(lr_prediction,y_test),4)*100)
accuracy_list.append(lr_accuracy)


# ### 3.2: Support Vector Classifier

# In[29]:


svc_model=SVC()
svc_model.fit(x_train_scaled,y_train)
svc_prediction=svc_model.predict(x_test_scaled)
svc_accuracy=(round(accuracy_score(svc_prediction,y_test),4)*100)
accuracy_list.append(svc_accuracy)


# ### 3.3: KNN Algorithm

# In[30]:


knn_list=[]
for k in range(1,50):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(x_train_scaled,y_train)
    knn_prediction=knn_model.predict(x_test_scaled)
    knn_accuracy=(round(accuracy_score(knn_prediction,y_test),4)*100)
    knn_list.append(knn_accuracy)
k = np.arange(1,50)
plt.plot(k,knn_list)


# In[31]:


knn_model = KNeighborsClassifier(n_neighbors=6)
knn_model.fit(x_train_scaled,y_train)
knn_prediction=knn_model.predict(x_test_scaled)
knn_accuracy=(round(accuracy_score(knn_prediction,y_test),4)*100)
accuracy_list.append(knn_accuracy)


# ### 3.4: Decision Tree

# In[32]:


dt_model = DecisionTreeClassifier(criterion="entropy",max_depth=2)
dt_model.fit(x_train_scaled,y_train)
dt_prediction=dt_model.predict(x_test_scaled)
dt_accuracy=(round(accuracy_score(dt_prediction,y_test),4)*100)
accuracy_list.append(dt_accuracy)


# ### 3.5: Naive Bayes

# In[33]:


nb_model = GaussianNB()
nb_model.fit(x_train_scaled,y_train)
nb_prediction=nb_model.predict(x_test_scaled)
nb_accuracy=(round(accuracy_score(nb_prediction,y_test),4)*100)
accuracy_list.append(nb_accuracy)


# ### 3.6: Random Forest Classifier

# In[34]:


rf_model = RandomForestClassifier()
rf_model.fit(x_train_scaled,y_train)
rf_prediction=rf_model.predict(x_test_scaled)
rf_accuracy=(round(accuracy_score(rf_prediction,y_test),4)*100)
accuracy_list.append(rf_accuracy)


# In[35]:


accuracy_list


# In[36]:


models=["Logistic Regression","SVC","KNearestNeighbors","Decision Tree","Naive Bayes","Random Forest"]


# In[37]:


plt.figure(figsize=(13,7))
ax=sns.barplot(x=models,y=accuracy_list)
plt.xlabel("Classifiers",fontsize=18)
plt.ylabel("Accuracy(%)",fontsize=18)
for p in ax.patches:
    width=p.get_width()
    height=p.get_height()
    x=p.get_x()
    y=p.get_y()
    ax.annotate(f"{height} %",(x+width/2,y+height*1.01),ha='center')
plt.show()


# ### Highest accuracy is attained by Decision Tree
