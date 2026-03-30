#!/usr/bin/env python
# coding: utf-8

# In[143]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[144]:


df = pd.read_csv("loan_approval_data.csv")


# In[145]:


df.head()
df.info()
df.isnull().sum()
df.describe()


# # Handle Missing Values

# In[146]:


categorical_cols = df.select_dtypes(include=["object"]).columns
#numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
numerical_cols = df.select_dtypes(include=["number"]).columns


# In[147]:


categorical_cols.size + numerical_cols.size


# In[148]:


numerical_cols


# In[149]:


from sklearn.impute import SimpleImputer
num_imp= SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])


# In[150]:


df.head()


# In[151]:


cat_imp= SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])


# In[152]:


df.head()


# In[153]:


df.isnull().sum()


# # EDA - exploratory data analysis

# In[154]:


# how balanced our classes are?

classes_count = df["Loan_Approved"].value_counts()

plt.pie(classes_count, labels=["No", "Yes"], autopct="%1.1f%%")
plt.title("Is Loan approved or not?")


# In[155]:


# analyze categories
gender_cnt = df["Gender"].value_counts()
ax=sns.barplot(gender_cnt)
ax.bar_label(ax.containers[0])



# In[156]:


edu_cnt = df["Education_Level"].value_counts()
ax=sns.barplot(edu_cnt)
ax.bar_label(ax.containers[0])


# In[157]:


# analyze income

sns.histplot(
    data = df,
    x="Applicant_Income",
    bins=20
)


# In[158]:


sns.histplot(
    data = df,
    x="Coapplicant_Income",
    bins=20
)


# In[159]:


# outliers - box plots
sns.boxplot(
    data = df,
    x = "Loan_Approved",
    y = "Applicant_Income"
)


# In[160]:


fig,axes = plt.subplots(2,2)
sns.boxplot(ax = axes[0,0],data = df, x = "Loan_Approved", y = "Applicant_Income")
sns.boxplot(ax = axes[0,1],data = df, x = "Loan_Approved", y = "Credit_Score")
sns.boxplot(ax = axes[1,0],data = df, x = "Loan_Approved", y = "DTI_Ratio")
sns.boxplot(ax = axes[1,1],data = df, x = "Loan_Approved", y = "Savings")

plt.tight_layout()


# In[161]:


# Credit Score with Loan Approved
sns.histplot(
    data = df,
    x="Applicant_Income",
    hue="Loan_Approved",
    bins=20,
    multiple = "dodge"
)


# In[162]:


# Remove Applicant Id
df=df.drop("Applicant_ID", axis=1)
df.head()


# # Encoding

# In[163]:


# df.head()
df.columns
df.info()


# In[168]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])


# In[169]:


df.head()


# cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
# 
# ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
# 
# encoded = ohe.fit_transform(df[cols]) # it give 2d array
# 
# # now convert into dataframe
# encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)
# #append this dataframme to original
# df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)

# In[173]:


df.head()
df.describe()
df.info()


# # Correlation Heatmap

# In[174]:


nums_cols = df.select_dtypes(include="number")
corr_matrix = nums_cols.corr()

plt.figure(figsize=(15,8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)


# In[175]:


nums_cols.corr()["Loan_Approved"].sort_values(ascending=False)


# # Train_Test_Split + Feature Scaling

# In[176]:


X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]


# In[177]:


X.head()


# In[178]:


y.head()


# In[179]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# In[183]:


X_train.head()


# In[184]:


X_test.head()


# In[185]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[187]:


X_train_scaled


# In[188]:


X_test_scaled


# # Train & Evaluate Models

# In[192]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
log_model = LogisticRegression()
log_model.fit(X_train_scaled,y_train)

y_pred = log_model.predict(X_test_scaled)

#Evaluation
print("Logistic Regression Model")
print("Precision: ",precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ",f1_score(y_test, y_pred))
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("CM: ",confusion_matrix(y_test, y_pred))


# In[195]:


# KNN 
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled,y_train)

y_pred = knn_model.predict(X_test_scaled)

#Evaluation
print("KNN Model")
print("Precision: ",precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ",f1_score(y_test, y_pred))
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("CM: ",confusion_matrix(y_test, y_pred))


# In[196]:


# Naive Bayes 

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train_scaled,y_train)

y_pred = nb_model.predict(X_test_scaled)

#Evaluation
print("Naive Bayes Model")
print("Precision: ",precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ",f1_score(y_test, y_pred))
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("CM: ",confusion_matrix(y_test, y_pred))


# # Best Model on the basis of Precision = Naive Bayes

# # Feature Enginnering

# In[202]:


#Add or Transform features


df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2

# df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

X=df.drop(columns =["Loan_Approved", "Credit_Score","DTI_Ratio" ])
y=df["Loan_Approved"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[203]:


X_train.head()


# In[204]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
log_model = LogisticRegression()
log_model.fit(X_train_scaled,y_train)

y_pred = log_model.predict(X_test_scaled)

#Evaluation
print("Logistic Regression Model")
print("Precision: ",precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ",f1_score(y_test, y_pred))
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("CM: ",confusion_matrix(y_test, y_pred))


# In[205]:


# KNN 
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled,y_train)

y_pred = knn_model.predict(X_test_scaled)

#Evaluation
print("KNN Model")
print("Precision: ",precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ",f1_score(y_test, y_pred))
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("CM: ",confusion_matrix(y_test, y_pred))


# In[206]:


# Naive Bayes 

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train_scaled,y_train)

y_pred = nb_model.predict(X_test_scaled)

#Evaluation
print("Naive Bayes Model")
print("Precision: ",precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ",f1_score(y_test, y_pred))
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("CM: ",confusion_matrix(y_test, y_pred))


# In[ ]:




