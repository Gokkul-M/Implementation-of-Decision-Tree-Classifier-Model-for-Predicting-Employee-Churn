# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:Gokkul M
RegisterNumber:212223240039
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
data = pd.read_csv("Employee.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
data["left"].value_counts
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
print(data.head())
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
print(x.head())
y=data["left"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
dt = DecisionTreeClassifier(criterion="entropy")
print(dt.fit(x_train,y_train))
y_pred = dt.predict(x_test)
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)
print(dt.predict([[0.5,0.8,9,260,6,0,1,2]]))
```
## Output:
![image](https://github.com/Gokkul-M/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870543/e9c00b16-db29-4f6d-94f1-de2ac1cde2e2)
![image](https://github.com/Gokkul-M/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870543/467cdd10-327c-4907-a4bc-fe8bae5ca51b)
![image](https://github.com/Gokkul-M/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870543/354b2746-e579-4410-8383-780f765258f3)
## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
