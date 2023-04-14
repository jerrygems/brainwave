import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def polynom():

    data = pd.read_csv("C:/Users/shubh/Downloads/speed.csv")

    data2 = pd.DataFrame()
    data2['workers'], data2['growth'], data2['distance'] = data['workers'], data['growth'], data['rank']

    degree = 4 # Change this to any degree you want

    X_train, X_test, y_train, y_test = train_test_split(data2[['workers', 'distance']], data2['growth'])
    X_train_df, X_test_df = pd.DataFrame(X_train), pd.DataFrame(X_test)

    poly = PolynomialFeatures(degree=degree)
    X_train_poly, X_test_poly = poly.fit_transform(X_train_df), poly.fit_transform(X_test_df)

    model = LinearRegression()
    model = model.fit(X_train_poly, y_train)

    cf = model.coef_
    intercept = model.intercept_

    plt.clf()

    x_axis = np.arange(0, 80000, 0.1)
    res = intercept + cf[0]*x_axis + cf[1]*x_axis**2 + cf[2]*x_axis[:,np.newaxis]**3 + cf[3]*x_axis[:,np.newaxis]**4 + cf[4]*x_axis[:,np.newaxis]*X_train_df['distance'].mean() + cf[5]*x_axis[:,np.newaxis]**2*X_train_df['distance'].mean()

    plt.scatter(data2[['workers', 'rank']], data2['growth'])
    plt.plot(x_axis, res, color='r')
    plt.show()

polynom()