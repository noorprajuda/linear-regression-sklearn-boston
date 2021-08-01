echo "# linear-regression-sklearn-boston" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/noorprajuda/linear-regression-sklearn-boston.git
git push -u origin main

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import  mean_squared_error, mean_absolute_error, r2_score

## Data preparation ##

#Loading data
X, y = datasets.load_boston(return_X_y = True)

#One feature usage
X = X[:, np.newaxis, 2]

#Determining the train size
train_size = int(0.85 * y.shape[0])

#Feature train-test sets split
X_train = X[:train_size]
X_test = X[train_size:]

#Target train-test sets split
y_train = y[:train_size]
y_test = y[train_size:]

## Data Modelling ##

#Creating linear regression object
lr = linear_model.LinearRegression()

#Train the model using training sets
lr.fit(X_train, y_train)

#Model predicting using test sets
y_pred = lr.predict(X_test)

## Showing statistical data and plot ##

#Showing the coefficient & intercept
print('Coefficient : ', lr.coef_)
print('Intercept : ', lr.intercept_)

#Showing the mean squared error, mean absolute error, and r-squared values
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error : ', mse)

mae = mean_absolute_error(y_test, y_pred)
print('Mean absolute error : ', mae)

rs = r2_score(y_test, y_pred)
print('R squared : ', rs)

#Showing the plot
plt.scatter(X_test, y_test, color = 'grey')
plt.plot(X_test, y_pred, color = 'red', linewidth = 3)
plt.show()
