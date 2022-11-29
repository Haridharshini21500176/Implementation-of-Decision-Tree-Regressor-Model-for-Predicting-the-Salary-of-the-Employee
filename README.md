# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.

2.Read the data set.

3.Apply label encoder to the non-numerical column inoreder to convert into numerical values.

4.Determine training and test data set.

5.Apply decision tree regression on to the dataframe and get the values of Mean square error, r2 and data prediction
 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Haridharshini.S 
RegisterNumber: 212221230033
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
![204432404-16937098-5ce6-4864-b4d6-ed05b835ad91](https://user-images.githubusercontent.com/94168395/204471755-ac5bca67-5c16-431d-847a-ad45a19a5903.png)

![204432434-bbf6de13-d48f-4d3b-b704-210825e98c20](https://user-images.githubusercontent.com/94168395/204471797-9d2d407f-eed9-4794-b502-115e71020773.png)

![204432461-9a2c3d06-9805-406d-a432-3ea11111aa81](https://user-images.githubusercontent.com/94168395/204471824-dfb08e0d-8b77-4cfb-b482-6b8085bec50a.png)

![204432492-fb4861fc-07c0-49d1-9135-99d8ae75ff56](https://user-images.githubusercontent.com/94168395/204471884-db64f37d-d402-464e-a205-5d8b14c79447.png)

![204432559-b6aec00a-5e72-432a-9993-c71367372fe7](https://user-images.githubusercontent.com/94168395/204471892-194a2f2a-8ba1-4314-a774-f95899929e87.png)

![204432577-4204b824-98ac-4be4-b24f-d302be918012](https://user-images.githubusercontent.com/94168395/204472029-617d00e3-de54-4d86-ba26-6d97cbdbd821.png)

### mse:
![204432597-774447c1-0b45-48e9-870a-4e12b18cb443](https://user-images.githubusercontent.com/94168395/204472152-360f992c-6387-41b3-85fe-13fbe798a90c.png)

### r2:
![204432619-28a99e94-6e42-4196-845b-df0474eefdad](https://user-images.githubusercontent.com/94168395/204472201-9a8d1f92-50f8-4e36-a388-ebc917a03b10.png)

### prediction
![Uploading 204432632-e1775300-59c1-4054-8a8f-fccce9f77b8b.png…]()


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
