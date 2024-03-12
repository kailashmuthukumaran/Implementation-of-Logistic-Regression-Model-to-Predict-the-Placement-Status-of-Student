# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and
.duplicated() function respectively
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required
modules from sklearn.
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: kailash s m 
RegisterNumber: 212222040068 
*/
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy Score:",accuracy)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:\n",confusion)


from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("\nClassification Report:\n",classification_report1)

from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
### DATA:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123893976/bcd595b2-4f5e-445b-baa0-1c65a7ae321c)

### ENCODED DATA:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123893976/5a16a10f-0d45-465b-866d-b7ad0117b9c6)

### NULL FUNCTION:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123893976/2c69320e-1691-46b8-afed-28ef073c3a6d)


### DATA DUPLICATE:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123893976/d82d125c-ec4d-4c02-afe0-81841c90410d)


### ACCURACY:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123893976/6172dd96-54eb-4b4f-9662-bf5d4f92352c)

### CONFUSION MATRIX:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123893976/645a23ca-6485-4d24-a0ab-9c06da269c4b)

### CLASSIFICATION REPORT:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123893976/35b27845-7d83-4ba3-8e07-692c50a0ea0e)

### PREDICATED VALUE:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123893976/cd69c3cc-1fe7-48bf-9548-fa8bbf3f2db6)

### GRAPH:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123893976/63e2f889-2cba-4d65-9684-96ec6db1fd9b)







## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
