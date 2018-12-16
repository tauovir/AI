# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, [1]].values
y = dataset.iloc[:, 2].values   #dataset.iloc[:, [2]].values, it return 2D

#Fitting Linear Regression to Dataset
from sklearn.linear_model import LinearRegression
linearModel = LinearRegression()
linearModel.fit(X,y)

#Fitting Polynomial Regression to Dataset
from sklearn.preprocessing import PolynomialFeatures
poli_reg = PolynomialFeatures(degree=4) # Chnage Degree like 2,3,4 and check Graph
X_poli = poli_reg.fit_transform(X)  # It will make X data in polinomial form, like y = b0 + b1x1 + b2x^2
poli_reg.fit(X_poli, y)
#Make Another Linear Regression model
linearModel2 = LinearRegression()
linearModel2.fit(X_poli, y)
#Visualize Data
fig = plt.figure(figsize = (13,6))
#Visualizing Linear Regression result
subfig1 = fig.add_subplot(2,2,1)
subfig1.scatter(X, y, color = 'red')
subfig1.plot(X, linearModel.predict(X), color = 'blue')
subfig1.title.set_text("Linear Regression")
subfig1.set_xlabel('Position Lavel')
subfig1.set_ylabel('Salary')

#Now Plot Polinomial Regression
subfig2 = fig.add_subplot(2,2,2)
subfig2.scatter(X, y, color = 'red')
subfig2.plot(X, linearModel2.predict(X_poli), color = 'blue')
subfig2.title.set_text("Polinomial Regression")
subfig2.set_xlabel('Position Lavel')
subfig2.set_ylabel('Salary')

#Now Predict new Value
print(linearModel.predict([[6]]))
#Npw Predict with Polinomials
print("Polinomial Prediction")
print(linearModel2.predict(poli_reg.fit_transform([[6]])))
plt.show()
print("==================")
print(X_poli)