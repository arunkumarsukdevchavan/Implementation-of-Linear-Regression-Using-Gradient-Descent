# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. add a column to x for the intercept,initialize the theta
2. perform graadient descent
3. read the csv file
4. assuming the last column is ur target variable 'y' and the preceeding column
5. learn model parameters
6. predict target value for a new data point

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: ARUN KUMAR SUKDEV CHAVAN
RegisterNumber:  212222230013
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())

X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta=linear_regression(X1_Scaled,Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
*/
```


## Output:

### Data:
![image](https://github.com/Anandanaruvi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120443233/a12d07f8-5d30-4121-8a97-ca6b4a997153)


### X values:

![image](https://github.com/Anandanaruvi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120443233/8fb412ef-196a-4e19-9cd1-f29f2016afa0)

### Y values:

![image](https://github.com/Anandanaruvi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120443233/2d41fc4c-74bc-49bd-8c08-1994fc0638a0)

### X scaled:
![image](https://github.com/Anandanaruvi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120443233/333a162c-af3d-4899-9e50-6f33671101b4)

### Y scaled:

![image](https://github.com/Anandanaruvi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120443233/d983846a-f309-4b6b-99a9-728d39906ea1)

### Predicted value:

![image](https://github.com/Anandanaruvi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120443233/36d93dbf-088c-4f8e-9dd3-90d244ffb96b)

## Result:

Thus the program to implement the linear regression using gradient descent is written and verified using python programming.


