import numpy as np
import pandas as pd
import os, time, datetime
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso,Ridge,ElasticNet,LinearRegression

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

def showNull(data):
    print(data.isna().sum())

def showHead(data):
    print(data.head())

def showTail(data):
    print(data.tail())

def describeCsv(data):
    print(data.describe())

def showColumns(data):
    print(data.columns)

def getPrompt():
    prompt = "𝖇𝖗𝖆𝖎𝖓𝖜𝖆𝖛𝖊🧠"
    return prompt

def plotter(mode="simple"):
    if mode=="simple":
        print("hello")
    else:
        print("bye")

def grafty(data):
    
    print("""

        You're in grafty mode

         _______  _______  _______  _______ _________         
        (  ____ \(  ____ )(  ___  )(  ____ \\\\__   __/|\     /|
        | (    \/| (    )|| (   ) || (    \/   ) (   ( \   / )
        | |      | (____)|| (___) || (__       | |    \ (_) / 
        | | ____ |     __)|  ___  ||  __)      | |     \   /  
        | | \_  )| (\ (   | (   ) || (         | |      ) (   
        | (___) || ) \ \__| )   ( || )         | |      | |   
        (_______)|/   \__/|/     \||/          )_(      \_/   
        
        Note:  you may need to hit exit to go back on previous mode
                                                            
    """)
    ##########################
    ##    definations       ##
    ##########################
    def getInput():
        colx = input("Enter X-axis column: ")
        coly = input("Enter Y-axis column: ")
        return colx,coly

    def plotGraph(data):
        colx,coly = getInput()
        plt.ion()
        plt.plot(data[colx], data[coly])
        plt.show(block=False)

    def barGraph(data):
        colx,coly = getInput()
        plt.ion()
        plt.bar(range(len(data[colx])), data[coly])
        plt.xticks(range(len(data[colx])), data[colx])
        plt.xlabel(colx)
        plt.ylabel(coly)
        plt.show(block=False)

    def pieGraph(data):
        col = input("enter the column: ")
        plt.ion()
        plt.pie(data[col])
        plt.show(block=False)

    def boxGraph(data):
        col = input("enter the column: ")
        plt.ion()
        plt.boxplot(data[col])
        plt.show(block=False)

    def histoGraph(data):
        col = input("enter the column: ")
        plt.ion()
        plt.hist(data[col])
        plt.show(block=False)

    def scatterGraph(data):
        colx,coly = getInput()
        plt.ion()
        plt.scatter(data[colx], data[coly])
        plt.show(block=False)
    ##########################
    ##    /definations      ##
    ##########################
    while True:
        cmd = input(getPrompt() + " grafty>>\b\b\b\b\b\b")
        if cmd == 'plot':
            plotGraph(data)
        elif cmd == 'bar':
            barGraph(data)
        elif cmd == 'pie':
            pieGraph(data)
        elif cmd == 'box':
            boxGraph(data)
        elif cmd == 'histogram':
            histoGraph(data)
        elif cmd == 'scatter':
            scatterGraph(data)
        elif cmd == 'clean':
            plt.clf()
        elif cmd == 'clear':
            os.system("cls" if os.name == 'nt' else "clear")
        elif cmd == 'help me' or cmd == 'help':
            print("""\tFollowing commands are available for GRAFTY\n1. plot\n2. bar\n3. pie\n4. box\n5. histogram\n6. scatter\n7. help\n8. exit """)
        elif cmd == 'leave' or cmd == 'exit':
            break

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

        print("\t<========================================>")
        print(f"\tyou entered these features <{features}>" )
        print(f"\tyou entered this target <{target}>")
        print("\t<========================================>")

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
        X_predict = np.array(values).reshape(-1,1)
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
    features = [name.strip() for name in features.split(";") ]
    target = input("enter the name of target variable : ")


    X = data[features]
    y = data[target]

    print("\t<========hey thanks for entering the values========>\n\tbut you know Lasso Regression requires some parameters\n\t check parameters")
    reg_ridge = Ridge(alpha=40, max_iter=1000, tol=0.1)
    reg_ridge = reg_ridge.fit(X,y)
    print("\t<========model fitted successfully========>")
    reg_ridge.score(X,y)

    while True:
        values = input(f"Enter the values of {features} (separated by semicolons): ")
        values = [float(value.strip()) for value in values.split(";")]
        values = np.array(values).reshape(1,-1)

        predictions = reg_ridge.predict(values)
        print(f"Predicted values {predictions[0]}")

        ans = input("Do you want to predict another value? (y/n): ")
        if ans.lower() == 'y':
            continue
        else:
            break

def LassoReg(data):
    features = input("enter the name of features separated by (';') : ")
    features = [name.strip() for name in features.split(";") ]
    target = input("enter the name of target variable : ")


    X = data[features]
    y = data[target]

    print("\t<========hey thanks for entering the values========>\n\tbut you know Lasso Regression requires some parameters\n\t check parameters")
    reg_lasso = Lasso(alpha=40, max_iter=1000, tol=0.1)
    reg_lasso = reg_lasso.fit(X,y)
    print("\t<========model fitted successfully========>")
    reg_lasso.score(X,y)

    while True:
        values = input(f"Enter the values of {features} (separated by semicolons): ")
        values = [float(value.strip()) for value in values.split(";")]
        values = np.array(values).reshape(1,-1)

        predictions = reg_lasso.predict(values)
        print(f"Predicted values {predictions[0]}")

        ans = input("Do you want to predict another value? (y/n): ")
        if ans.lower() == 'y':
            continue
        else:
            break



def ElasticNetReg(data):
    features = input("enter the name of features separated by (';') : ")
    features = [name.strip() for name in features.split(";") ]
    target = input("enter the name of target variable : ")

    X = data[features]
    y = data[target]

    elastic_reg = ElasticNet(random_state=0)
    elastic_reg.fit(X,y)
    print("\t<========model fitted successfully========>")
    elastic_reg.score(X,y)

    while True:
        values = input(f"Enter the values of {features} (separated by semicolons): ")
        values = [float(value.strip()) for value in values.split(";")]
        values = np.array(values).reshape(1,-1)
        predictions = elastic_reg.predict(values)
        print(f"Predicted values {predictions[0]}")

        ans = input("Do you want to predict another value? (y/n): ")
        if ans.lower() == 'y':
            continue
        else:
            break


def LogisticReg(data):
    features = input("enter the name of features separated by (';') : ")
    features = [name.strip() for name in features.split(";") ]
    target = input("enter the name of target variable : ")

def PoissonReg(data):
    print("hell")

def CoxReg(data):
    print("hell")

def SupportVectReg(data):
    print("hell")

def DecisionTreeReg(data):
    print("hell")

def tutor():
    print('hey for this feature you\'ll require a file from jerrygems github repo')

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
        elif choice == '3':
            RidgeReg(data)
        elif choice == '4':
            LassoReg(data)
        elif choice == '5':
            ElasticNetReg(data)
        elif choice == '6':
            LogisticReg()
        else:
            print('not valid')
    except Exception as err:
        print(f"hey i didn't get what you want : {err}")
        


###################################
##        /predictions           ##
###################################

###################################
##        classification        ##
###################################
def classify():
    print("")


###################################
##       /classification         ##
###################################


# Print the ASCII art of the brain
# with open('assets/brain.txt', 'r') as file:
#     print(file.read())

def getFile():
    csv_path = input("Enter the path of the CSV file: ").replace("\"","")
    data = pd.read_csv(csv_path)
    return data

data = getFile()

while True:
    try:
        inp = input(getPrompt()+" >>\b\b\b\b\b\b")
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
        elif inp == 'linearReg':
            linearReg(data)
        elif inp == 'polynomReg':
            polynomReg(data)
        elif inp == 'ridgeReg':
            RidgeReg(data)
        elif inp == 'lassoReg':
            LassoReg(data)
        elif inp == 'logisticReg':
            LogisticReg(data)
        elif inp == 'learn':
            tutor()
        elif inp == 'grafty':
            grafty(data)
        elif inp == 'changeFile':
            data = getFile()
        elif inp == 'classification':
            classify()
        elif inp == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
        elif inp == 'ls' or inp == 'dir':
            os.system('dir' if os.name == 'nt' else 'ls -al')
        elif inp == 'brainwave' or inp == 'brain':
            fl = open('./assets/brain.txt','r').read()
            print(fl) 
        elif inp == 'showna' or inp == 'show na' or inp == 'showNA':
            showNull(data)
        elif inp == 'help' or inp == 'help me' or inp == 'please help' or inp == 'Help':
            print("""
                Usage: following commands can be executed from this script
                \n\t1. mean\n\t2. head\n\t3. tail\n\t4. describe\n\t5. show na\n\t6. show columns\n\t7. models\n\t8. median\n\t9. info\n\t10. learn
        11. linearReg\n\t12. polynomReg\n\t13. ridgeReg\n\t14. lassoReg\n\t15. logisticReg\n\t16. grafty\n\t17. changeFile
            """)
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

# "C:\Users\shubh\Downloads\speed.csv"