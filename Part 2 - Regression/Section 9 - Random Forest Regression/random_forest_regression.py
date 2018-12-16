# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting RandomForestRegressor to the dataset
from sklearn.ensemble import RandomForestRegressor
randomModel = RandomForestRegressor(n_estimators=200, random_state=0)
randomModel.fit(X, y)



# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, randomModel.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
print("============Predict value===========")
print(randomModel.predict([[6.5]]))