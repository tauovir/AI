# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
#preprocessiing data or filling na to mean
dataset['Age'] = dataset['Age'].fillna(np.mean(dataset['Age']))
dataset['Salary'] = dataset['Salary'].fillna(np.mean(dataset['Salary']))

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Encode Categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoderX = LabelEncoder()
X[:,0] = labelEncoderX.fit_transform(X[:,0])

oneHotEncoderX = OneHotEncoder(categorical_features=[0]) # Here we specify which column we need oneHot Encoding
X = oneHotEncoderX.fit_transform(X).toarray()
# Now encode Y value
labelEncoderY = LabelEncoder()
y = labelEncoderY.fit_transform(y)

#======Spliting Data into traing set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)
print(X_train)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
X_train = scX.fit_transform(X_train)
X_test = scX.transform(X_test)
print("======================")
print(X_train)
