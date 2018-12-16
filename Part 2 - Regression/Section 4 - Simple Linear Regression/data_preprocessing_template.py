# Data Preprocessing Template

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataSet = pd.read_csv("Salary_Data.csv")
X = dataSet.iloc[:,0:1].values  # Strain Data in 2d araay
y = dataSet.iloc[:,1].values    # 1D array

#Split Data for xtrain and  y train
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) # For getting same result we have to put random_state = 0 else it randomly generate
#print(X_train)
#print(X_train.reshape(1,-1))
#Fitting Simple Linear Regresion To training set
from sklearn.linear_model import LinearRegression
linerRegresiionModel  = LinearRegression()
linerRegresiionModel.fit(X_train, y_train)  
#=========Now Predict test set==============
predictVal = linerRegresiionModel.predict(X_test)
print("===========Predicted Value===================")
print(predictVal)

#Now Plot Scatter
fobj = plt.figure(figsize = (6,6), facecolor = (1,0,1))
fobj.canvas.set_window_title('Plot Diagram')
#Train X and Y vaue graph
spobj1=fobj.add_subplot(221)
spobj1.scatter(X_train,y_train)
#Xtest and predict value graph
spobj2=fobj.add_subplot(223)
spobj2.scatter(X_test,predictVal)


plt.show()
