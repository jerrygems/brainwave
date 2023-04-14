import numpy as np
import pandas as pd
import os, time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("""
        options : following operations may performed by this function
        1. describe
        2. head
        3. tail
        4. describe
        5. plotter
    """)

def mean(data):
    columns = input("enter the columns (use \";\" as separator) : ").split(";")
    means = {}
    for col in columns:
        if col not in data.columns:
            print(f"There is no column named {col}.")
        else:
            means[col] = data[col].mean()
    print(means)

def median(data):
    columns = input("enter the columns (use \";\" as separator) : ").split(";")
    medians = {}
    for col in columns:
        if col not in data.columns:
            print(f"There is no column named {col}.")
        else:
            medians[col] = data[col].median()
    print(medians)

def dataInfo(data):
    print(data.info())

def showHead(data):
    print(data.head())

def showTail(data):
    print(data.tail())

def describeCsv(data):
    print(data.describe())

def showColumns(data):
    print(data.columns)

def getPrompt():
    prompt = "𝖇𝖗𝖆𝖎𝖓𝖜𝖆𝖛𝖊🧠>> \b\b\b\b\b\b\b\b\b\b"
    return prompt

def plotter(mode="simple"):
    if mode=="simple":
        print("hello")
    else:
        print("bye")

#apply prediction stuff here 
###################################
##        predictions            ##
###################################

def linearReg(data):
    
    print("Note: <hey there are two kind of linear regression >")
    print("Which one you want")
    anss = int(input("1. simple linear regression\n2. multiple linear regression\nchoose (1 or 2) : "))

    if anss == 1:
        print("<========simple linear regression========>")
        feature = input("Enter the feature to use in prediction : ")
        target = input("enter the target value aka 'y' : ")

        print("#################################################")
        print(f"you entered this feature <{feature}>" )
        print(f"you entered this target <{target}>")
        print("#################################################")

        X = np.array(data[feature]).reshape(-1,1)
        y = np.array(data[target]).reshape(-1,1)
        reg = LinearRegression()
        reg.fit(X,y)
        print("<========data fitted successfully========>")

        print(f"So you coeficient is {reg.coef_} and intercept is {reg.intercept_}")
        print(f"Model performance {reg.score()}")

        while True:
            predict_val = input("enter the value for respective {feature} you entered before : ")
            print(f"predicted value is <{reg.predict(np.array(predict_val).reshape(-1,1).astype(float))}>")

            ans = input("wanna predict again? (y/n) : ")
            if ans == "y":
                continue
            else:
                break
    else:
        print("<========multiple linear regression========>")
        features_dummy = input("Enter the features to use in prediction (use \";\" for separation): ")
        features = features_dummy.split(';')

        target = input("enter the target value aka 'y' : ")

        print("\t#################################################")
        print(f"\tyou entered these features <{features}>" )
        print(f"\tyou entered this target <{target}>")
        print("\t#################################################")

        X = data[features]
        print(f"number of features entered <{X.shape[1]}>")
        y = data[target]
        reg = LinearRegression()
        reg.fit(X,y)
        print("<========data fitted successfully========>")

        print(f"So you coeficients are {reg.coef_} and intercept is {reg.intercept_}")

        while True:
            predict_val_dummy = input("enter the values for respective {features} you entered before : ")
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
    features = [name.strip() for name in features.split(";") ]
    target = input("enter the name of target variable : ")
    
    X = data[features].values
    y = data[target].values

    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly,y)

    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"R^2 score: {model.score(X_poly, y)}")

    while True:
        values = input(f"Enter the values of {features} (separated by semicolons): ")
        values = [float(value.strip()) for value in values.split(";")]
        X_predict = np.array(values).reshape(1,-1)
        X_predict_poly = poly.transform(X_predict)

        y_pred = model.predict(X_predict_poly)
        print(f"Predicted values {y_pred[0]}")

        ans = input("Do you want to predict another value? (y/n): ")
        if ans.lower() == 'y':
            continue
        else:
            break

def RidgeReg(data):
    print("hell")

def LassoReg(data):
    print("hell")

def ElasticNetReg(data):
    print("hell")

def LogisticReg(data):
    print("hell")

def PoissonReg(data):
    print("hell")

def CoxReg(data):
    print("hell")

def SupportVectReg(data):
    print("hell")

def DecisionTreeReg(data):
    print("hell")



def models(data):
    print("""
        following models you can use directly
        1. Linear Regression
        2. Polynomial Regression
        3. Ridge Regression
        4. Lasso Regression
        5. ElasticNet Regression
        6. Logistic Regression
        7. Poisson Regression
        8. Cox Regression
        9. Support Vector Regression
        10. Decision Tree Regression
        11. Gradient Boosting Regression
        12. Neural Network Regression
        13. Bayesian Regression
        14. K-Nearest Neighbour
        15. Quantile Regression
        16. Gaussian Regression
        17. Robust Regression
        18. Multi Task Regression
        19. Multi-output Regression
        20. Time-series Regression
        21. Ordinal Regression
        22. Quantative Regression
    """)
    choice = input("select the model by their assined int numbers : ")
    try:
        if choice == '1':
            linearReg(data)
        elif choice == '2':
            polynomReg(data)
        else:
            print('not valid')
    except Exception as err:
        print(f"hey i didn't get what you want : {err}")
        


###################################
##        /predictions           ##
###################################


# Print the ASCII art of the brain
# with open('assets/brain.txt', 'r') as file:
#     print(file.read())

csv_path = input("Enter the path of the CSV file: ").replace("\"","")
data = pd.read_csv(csv_path)

while True:
    try:
        inp = input(getPrompt())
        if inp == 'mean':
            mean(data)
        elif inp == 'head':
            showHead(data)
        elif inp == 'tail':
            showTail(data)
        elif inp == 'describe':
            describeCsv(data)
        elif inp == 'show columns':
            showColumns(data)
        elif inp == 'models':
            models(data)
        elif inp == 'median':
            median(data)
        elif inp == 'info':
            dataInfo(data)
        elif inp == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
        elif inp == '':
            print("huh how many times i told you that you must have to enter commands here don't you just understand")
        elif inp == 'exit':
            print('hey i see yah man next time i\'ll be more cool than now i am')
            break
        else:
            continue
    except Exception as err:
        print(f"Sorry, the following error occurred: {err}")
        continue
