# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, [1]].values
y = dataset.iloc[:, [2]].values
print(y)
print(y.ravel())
#Feature Scaling because Svr not included Fetaure scaling in its Library
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler()
y_std = StandardScaler()
X = X_std.fit_transform(X)
y = y_std.fit_transform(y)

#Fitting Dataset in Model Svr
from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(X, y.ravel())    #ravel :  returns contiguous flattened array(1D array with all the input-array elements and with the same type as it)

#Predicting New Value
#y_pred = regression.predict(X_std.transform(np.array([[6.5]])) ) 
# But We need actual value of predicted value so let's inverse transformation
y_pred = y_std.inverse_transform(regression.predict(X_std.transform(np.array([[6.5]])) ) ) # Now Getting Same Result
print("================")
print(y_pred)
plt.scatter(X,y,color="red")
plt.plot(X,regression.predict(X), color = "blue")
plt.title("Support Vector Regressor")
plt.xlabel("Position Lavel")
plt.ylabel("Salary")
plt.show()
