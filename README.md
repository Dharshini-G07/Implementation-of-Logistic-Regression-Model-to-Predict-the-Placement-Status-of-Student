# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## AlgorithmLoad the Dataset
1.Load the Dataset

2.Create a Copy of the Original Data

3.Drop Irrelevant Columns (sl_no, salary)

4.Check for Missing Values

5.Check for Duplicate Rows

6.Encode Categorical Features using Label Encoding

7.Split Data into Features (X) and Target (y)

8.Split Data into Training and Testing Sets

9.Initialize and Train Logistic Regression Model

10.Make Predictions on Test Set

11.Evaluate Model using Accuracy Score

12.Generate and Display Confusion Matrix

13.Generate and Display Classification Report

14.Make Prediction on a New Sample Input

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Priyadharshini G
RegisterNumber:  212224230209
*/
```

```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn .preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver= "liblinear")
lr.fit(x_train,y_train)
y_pred= lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report = classification_report(y_test,y_pred)
print(classification_report)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
## data.head()
![image](https://github.com/user-attachments/assets/9a72d67e-d57a-43c9-8f4e-eec25d1f4e6d)
## data1.head()
![image](https://github.com/user-attachments/assets/8eb5505d-4158-4400-88b4-5477a2309f5d)
## isnull()
![image](https://github.com/user-attachments/assets/0e1c4c75-1883-4133-adc8-5ca8944e06c1)
## duplicated()
![image](https://github.com/user-attachments/assets/570169e2-03f8-4113-a3f1-4c7089dfc07e)
## data1
![image](https://github.com/user-attachments/assets/567238f7-1fde-47ad-8c79-cea8f857fd8c)
## X
![image](https://github.com/user-attachments/assets/fea8cee6-4e28-4a18-b8e9-c2a99005d6db)
## y
![image](https://github.com/user-attachments/assets/7fc199f9-cae5-405e-b6e4-3c259c8d4d5e)
## y_pred
![image](https://github.com/user-attachments/assets/c0bced6d-257a-4810-aeb7-798844d93bad)
## confusion matrix
![image](https://github.com/user-attachments/assets/4cf95e7a-9a46-446a-8e56-1626808b25dd)
## classification report
![image](https://github.com/user-attachments/assets/628c6875-c1c0-40f8-8fe8-2f10eee9e490)
## prediction
![image](https://github.com/user-attachments/assets/72803270-8e5f-4e5b-a17f-682fb66075d3)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
