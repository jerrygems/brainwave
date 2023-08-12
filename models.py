import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression

class Models:
    def __init__(self,data):
        self.data=data

    def linearReg(self):
        print("Note: <hey there are two kind of linear regression >")
        print("Which one you want")
        anss = int(input(
            "1. simple linear regression\n2. multiple linear regression\nchoose (1 or 2) : "))

        if anss == 1:
            print("<========simple linear regression========>")
            feature = input("Enter the feature to use in prediction : ")
            target = input("enter the target value aka 'y' : ")

            print("#################################################")
            print(f"you entered this feature <{feature}>")
            print(f"you entered this target <{target}>")
            print("#################################################")

            X = np.array(self.data[feature]).reshape(-1, 1)
            y = np.array(self.data[target]).reshape(-1, 1)
            reg = LinearRegression()
            reg.fit(X, y)
            print("<========data fitted successfully========>")

            print(
                f"So you coeficient is {reg.coef_} and intercept is {reg.intercept_}")
            print(f"Model performance {reg.score()}")

            while True:
                predict_val = input(
                    "enter the value for respective {feature} you entered before : ")
                print(
                    f"predicted value is <{reg.predict(np.array(predict_val).reshape(-1,1).astype(float))}>")

                ans = input("wanna predict again? (y/n) : ")
                if ans == "y":
                    continue
                else:
                    break
        else:
            print("<========multiple linear regression========>")
            features_dummy = input(
                "Enter the features to use in prediction (use \";\" for separation): ")
            features = features_dummy.split(';')

            target = input("enter the target value aka 'y' : ")

            print("\t<========================================>")
            print(f"\tyou entered these features <{features}>")
            print(f"\tyou entered this target <{target}>")
            print("\t<========================================>")

            X = data[features]
            print(f"number of features entered <{X.shape[1]}>")
            y = data[target]
            reg = LinearRegression()
            reg.fit(X, y)
            print("<========data fitted successfully========>")

            print(
                f"So you coeficients are {reg.coef_} and intercept is {reg.intercept_}")

            while True:
                predict_val_dummy = input(
                    "enter the values for respective {features} you entered before : ")
                predict_val = [int(i) for i in predict_val_dummy.split(";")]
                print(f"predicted value is <{reg.predict([predict_val])}>")

                ans = input("wanna predict again? (y/n) : ")
                if ans == "y":
                    continue
                else:
                    break


    def polynomReg(data):
        degree = int(input("enter the the degree of the polynomial : "))
        features = input("enter the name of features separated by (';') : ")
        features = [name.strip() for name in features.split(";")]
        target = input("enter the name of target variable : ")

        X = data[features].values
        y = data[target].values

        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        print(f"Coefficients: {model.coef_}")
        print(f"Intercept: {model.intercept_}")
        print(f"R^2 score: {model.score(X_poly, y)}")

        while True:
            values = input(
                f"Enter the values of {features} (separated by semicolons): ")
            values = [float(value.strip()) for value in values.split(";")]
            X_predict = np.array(values).reshape(-1, 1)
            X_predict_poly = poly.transform(X_predict)

            y_pred = model.predict(X_predict_poly)
            print(f"Predicted values {y_pred[0]}")

            ans = input("Do you want to predict another value? (y/n): ")
            if ans.lower() == 'y':
                continue
            else:
                break


    def RidgeReg(data):
        features = input("enter the name of features separated by (';') : ")
        features = [name.strip() for name in features.split(";")]
        target = input("enter the name of target variable : ")

        X = data[features]
        y = data[target]

        print("\t<========hey thanks for entering the values========>\n\tbut you know Lasso Regression requires some parameters\n\t check parameters")
        reg_ridge = Ridge(alpha=40, max_iter=1000, tol=0.1)
        reg_ridge = reg_ridge.fit(X, y)
        print("\t<========model fitted successfully========>")
        reg_ridge.score(X, y)

        while True:
            values = input(
                f"Enter the values of {features} (separated by semicolons): ")
            values = [float(value.strip()) for value in values.split(";")]
            values = np.array(values).reshape(1, -1)

            predictions = reg_ridge.predict(values)
            print(f"Predicted values {predictions[0]}")

            ans = input("Do you want to predict another value? (y/n): ")
            if ans.lower() == 'y':
                continue
            else:
                break


    def LassoReg(data):
        features = input("enter the name of features separated by (';') : ")
        features = [name.strip() for name in features.split(";")]
        target = input("enter the name of target variable : ")

        X = data[features]
        y = data[target]

        print("\t<========hey thanks for entering the values========>\n\tbut you know Lasso Regression requires some parameters\n\t check parameters")
        reg_lasso = Lasso(alpha=40, max_iter=1000, tol=0.1)
        reg_lasso = reg_lasso.fit(X, y)
        print("\t<========model fitted successfully========>")
        reg_lasso.score(X, y)

        while True:
            values = input(
                f"Enter the values of {features} (separated by semicolons): ")
            values = [float(value.strip()) for value in values.split(";")]
            values = np.array(values).reshape(1, -1)

            predictions = reg_lasso.predict(values)
            print(f"Predicted values {predictions[0]}")

            ans = input("Do you want to predict another value? (y/n): ")
            if ans.lower() == 'y':
                continue
            else:
                break


    def ElasticNetReg(data):
        features = input("enter the name of features separated by (';') : ")
        features = [name.strip() for name in features.split(";")]
        target = input("enter the name of target variable : ")

        X = data[features]
        y = data[target]

        elastic_reg = ElasticNet(random_state=0)
        elastic_reg.fit(X, y)
        print("\t<========model fitted successfully========>")
        elastic_reg.score(X, y)

        while True:
            values = input(
                f"Enter the values of {features} (separated by semicolons): ")
            values = [float(value.strip()) for value in values.split(";")]
            values = np.array(values).reshape(1, -1)
            predictions = elastic_reg.predict(values)
            print(f"Predicted values {predictions[0]}")

            ans = input("Do you want to predict another value? (y/n): ")
            if ans.lower() == 'y':
                continue
            else:
                break


    def LogisticReg(data):
        features = input("enter the name of features separated by (';') : ")
        features = [name.strip() for name in features.split(";")]
        target = input("enter the name of target variable : ")


    def PoissonReg(data):
        print("hell")


    def CoxReg(data):
        print("hell")


    def SupportVectReg(data):
        print("hell")


    def DecisionTreeReg(data):
        print("hell")


    def tutor(self):
        print('hey for this feature you\'ll require a file from jerrygems github repo')


    