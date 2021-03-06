# Data Preprocessing Template

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("50_Startups.csv",skiprows=2)
#print(dataset)
X = dataset.iloc[:,:-1:].values    #Get data except last one column
y = dataset.iloc[:,4].values

#Now Encode Categorical state name into numerical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lebelEncode = LabelEncoder()
X[:,3] = lebelEncode.fit_transform(X[:,3])
#======Now Use Onte hot encoder
oneHotEnc = OneHotEncoder(categorical_features=[3]) # TAke index three as categorical feature
X = oneHotEnc.fit_transform(X).toarray()
#Now Avoid Dummy Variable Trap
X = X[:,1:] # We have Remove One dummy varibale which is generated by onhote encoder

#Split Dataset into training and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fit Model
from sklearn.linear_model import LinearRegression
linerModel = LinearRegression()
linerModel.fit(X_train, y_train)
predictVal = linerModel.predict(X_test)
#print("============Predicted Value==============")
#print(predictVal)

#Build Optimal Model using BAckward Elimination
import statsmodels.formula.api as sm
#We  y = b0 = x1b1 + x2b2 + ... xnbn,So we are making y = b0x0 + b1x1 + b2x2 + ..... bnxn
#X = np.append(X, np.ones((48,1)).astype(int), axis=1) # It will append in last position
#We need to add One clomuns on ones in first position that why we appending
X = np.append(np.ones((48,1)).astype(int),X, axis=1)
xOpt = X[:,[0,1,2,3,4,5]]
regressorOls = sm.OLS(endog = y, exog = xOpt).fit()
# See Summary if P > 0.05 then eleminate which have higher value of p 
print(regressorOls.summary())
#Eleminate x it has 0.959
xOpt = X[:,[0,1,3,4,5]]
regressorOls = sm.OLS(endog = y, exog = xOpt).fit()
print(regressorOls.summary())
# Now Remove X2 because it has 0.897 value
xOpt = X[:,[0,3,4,5]]
regressorOls = sm.OLS(endog = y, exog = xOpt).fit()
print(regressorOls.summary())
#BAckward Elimination process
xOpt = X[:,[0,3,5]]
regressorOls = sm.OLS(endog = y, exog = xOpt).fit()
print(regressorOls.summary())
#BAckward Elimination process


"""
1 > Select Significant level to stay in he model(ex SL = 0.05)
2 > fit full model with all posible predictor
3> Consider the predictor with highest P-value. if  P> S, go to step 4 , else got o fn
4 >Remove Prictor
5> fit model without this variable

"""


