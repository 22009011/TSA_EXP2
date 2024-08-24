
# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
### Developed by: THANJIYAPPAN K
### Register No: 212222240108
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
1. Import necessary libraries (NumPy, Matplotlib)
2. Load the dataset
3. Calculate the linear trend values using least square method
4. Calculate the polynomial trend values using least square method
5. End the program
### PROGRAM:
#### A - LINEAR TREND ESTIMATION
```
Name :THAJIYAPPAN K
Register No: 212222240108
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
data = pd.read_csv('EV.csv')

data['year'] = pd.to_datetime(data['year'], format='%Y')
data = data.sort_values('year')
data['year_ordinal'] = data['year'].apply(lambda x: x.toordinal())
X = data['year_ordinal'].values.reshape(-1, 1)
y = data['value'].values
linear_model = LinearRegression()
linear_model.fit(X, y)
data['Linear_Trend'] = linear_model.predict(X)

# Plotting the Linear Trend
plt.figure(figsize=(10, 6))
plt.plot(data['year'], data['value'], label='Original Data')
plt.plot(data['year'], data['Linear_Trend'], color='yellow', label='Linear Trend')
plt.title('Linear Trend Estimation')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# B - Polynomial Trend Estimation
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
data['Polynomial_Trend'] = poly_model.predict(X_poly)

# Plotting the Polynomial Trend
plt.figure(figsize=(10, 6))
plt.bar(data['year'], data['value'], label='Original Data', alpha=0.6)
plt.plot(data['year'], data['Polynomial_Trend'], color='green', label='Polynomial Trend (Degree 2)')
plt.title('Polynomial Trend Estimation')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
```
### Dataset:
![image](https://github.com/user-attachments/assets/826e3eaf-8931-4a41-b04a-56de3a0fc4ae)

### OUTPUT
A - LINEAR TREND ESTIMATION
![image](https://github.com/user-attachments/assets/3b9065e3-a8fe-4120-ba87-0ac683f5a7f3)


B- POLYNOMIAL TREND ESTIMATION
![image](https://github.com/user-attachments/assets/0e694bdd-33b7-41a1-9dfb-98c31a27a5b0)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
