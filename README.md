# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required library and read the dataframe.

2. Write a function computeCost to generate the cost function.

3. Perform iterations og gradient steps with learning rate.

4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: ARUN KUMAR SUKDEV CHAVAN
RegisterNumber:  212222230013
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt",header = None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X,y,theta and generate the cost function of using the in a linear regression model
  """
  m=len(y) # length of the training data
  h=X.dot(theta) #hypothesis
  square_err=(h-y)**2

  return 1/(2*m) * np.sum(square_err) #returning J

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) #Call the function

from matplotlib.container import ErrorbarContainer
from IPython.core.interactiveshell import error
def gradientDescent(X,y,theta,alpha,num_iters):
    """
    Take the numpy array X,y,theta and update theta by taking the num_tiers gradient with learning rate of alpha

    return theta and the list of the cost of theta during each iteration
    """

    m=len(y)
    J_history=[]

    for i in range(num_iters):
      predictions=X.dot(theta)
      error=np.dot(X.transpose(),(predictions -y))
      descent=alpha *1/m*error
      theta-=descent
      J_history.append(computeCost(X,y,theta))

    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

#Testing the implementation
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")


plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000")
plt.title("Profit Prediction")

def predict(x,theta):
  """
  Tkes in numpy array of x and theta and return the predicted value of y base
  """

  predictions=np.dot(theta.transpose(),x)

  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population =35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))


```

# Output:
## Profit Prediction Graph

![image](https://github.com/Nethraa24/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121215786/32ae3100-ec28-4589-9798-67f5608e14ef)


![image](https://github.com/Nethraa24/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121215786/18ea73ac-6ba8-44a3-aaa4-ee1786fafe91)

## Compute Cost Value

![image](https://github.com/Nethraa24/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121215786/865b66ee-b809-4225-9938-b5c637b8fe5a)

## h(x) Value

![image](https://github.com/Nethraa24/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121215786/ee1234a4-ac6f-4a73-932e-c6882b2ad40a)

## Cost function using Gradient Descent Graph

![image](https://github.com/Nethraa24/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121215786/1b398cc0-d18e-4b72-b18b-33f4f07f7874)

## Profit for the Population 35,000

![image](https://github.com/Nethraa24/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121215786/ebf46437-a362-459e-87bf-4af581b60cd2)

## Profit for Population 70,000

![image](https://github.com/Nethraa24/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121215786/d693817e-0e05-479d-993f-b8d05c28538c)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
