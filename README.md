# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn
4.Assign the points for representing in the graph
5.Predict the regression for marks by using the representation of the graph
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Niranjani.C
RegisterNumber:212223220069
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv('student_scores.csv')

df.head()

df.tail()

x = df.iloc[:,:-1].values
x

y = df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

y_pred

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE= ",rmse)

## Output:
![Screenshot 2024-08-25 194855](https://github.com/user-attachments/assets/fa46aab9-6db1-4a5c-a9f0-8b9b4d8acfd8)

![Screenshot 2024-08-25 194945](https://github.com/user-attachments/assets/3ed7082d-eed4-4cc8-915f-fb2524386df7)

![Screenshot 2024-08-25 195005](https://github.com/user-attachments/assets/cae1062c-2d3e-4104-8566-398811e1643f)

![Screenshot 2024-08-25 195017](https://github.com/user-attachments/assets/27ff0a59-ab67-4b9a-bace-46286039a9a8)

![Screenshot 2024-08-25 195026](https://github.com/user-attachments/assets/f6d6b60d-bf83-4f0d-aed0-2f673e81d271)

![6](https://github.com/user-attachments/assets/17fb2a19-8ed2-471b-926b-93ac7243a558)

![Screenshot 2024-08-25 195034](https://github.com/user-attachments/assets/c9c12976-3092-4b3c-a972-9d28dfa40913)

![Screenshot 2024-08-25 195100](https://github.com/user-attachments/assets/b22519b1-6890-45f6-8f1c-75b3347972a9)

![Screenshot 2024-08-25 195114](https://github.com/user-attachments/assets/6e6f026b-6a38-4307-a0aa-5c023a35e388)

![Screenshot 2024-08-25 195120](https://github.com/user-attachments/assets/f6487dca-bec5-4799-b4e8-25e3bf09cd08)

![Screenshot 2024-08-25 195124](https://github.com/user-attachments/assets/0119a020-dd1d-4929-bc5e-c67d035b5027)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
