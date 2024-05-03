# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Prasanth U
RegisterNumber: 212222220031
*/
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
## Encoding:
![image](https://github.com/Prasanth9025/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118343686/b07d5c98-3648-4966-b29c-0840fad512be)
## Head():
![image](https://github.com/Prasanth9025/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118343686/f127e34e-4277-437b-81d2-7441de12a5ca)
## Info()
![image](https://github.com/Prasanth9025/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118343686/fc0e3858-3389-4a54-9dfd-f5a5ba32f0f7)
## snull().sum():
![image](https://github.com/Prasanth9025/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118343686/da49edc4-26d7-46b2-ab5a-c991636aa2c2)
## Prediction of y
![image](https://github.com/Prasanth9025/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118343686/39b133a8-6dda-4b41-8bc4-02e8aadd0e24)
## Accuracy
![image](https://github.com/Prasanth9025/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118343686/c1c56524-33e3-4d5c-9520-0de92e9cf0f5)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
