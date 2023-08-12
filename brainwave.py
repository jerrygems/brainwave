"""
BrAiNwAvE is a tool that allows us to perform usual operations related to data analysis on the terminal
Author : jerrgems
Social-Links : 
"""
#!/bin/python3
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from operations import Opr
from models import Models
import warnings
warnings.filterwarnings('ignore')

print("""
        hey  welcome here
    """)



def getPrompt():
    prompt = "BrAiNwAvE🧠"
    return prompt


def plotter(mode="simple"):
    if mode == "simple":
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
        return colx, coly

    def plotGraph(data):
        colx, coly = getInput()
        plt.ion()
        plt.plot(data[colx], data[coly])
        plt.show(block=False)

    def barGraph(data):
        colx, coly = getInput()
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
        colx, coly = getInput()
        plt.ion()
        plt.scatter(data[colx], data[coly])
        plt.show(block=False)
    ##########################
    ##    /definations      ##
    ##########################
    while True:
        cmd = input(getPrompt() + " grafty>>")
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

# apply prediction stuff here
###################################
##        predictions            ##
###################################



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
                mods.linearReg()
            elif choice == '2':
                mods.polynomReg()
            elif choice == '3':
                mods.RidgeReg()
            elif choice == '4':
                mods.LassoReg()
            elif choice == '5':
                mods.ElasticNetReg()
            elif choice == '6':
                mods.LogisticReg()
            else:
                print('not valid')
        except Exception as errr:
            print(f"hey i didn't get what you want : {errr}")



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
    csv_path = input("Enter the path of the CSV file: ").replace("\"", "")
    data = pd.read_csv(csv_path)
    return data


data = getFile()
opr = Opr(data)
mods = Models(data)

while True:
    try:
        inp = input(getPrompt()+" >>")
        if inp == 'mean':
            opr.mean()
        elif inp in ['head','top']:
            opr.showHead()
        elif inp in ['tail','last']:
            opr.showTail()
        elif inp in ['describe','des']:
            opr.describeCsv()
        elif inp == 'show columns':
            opr.showColumns()
        elif inp == 'models':
            models(data)
        elif inp == 'median':
            opr.median()
        elif inp == 'info':
            opr.dataInfo()
        elif inp == 'linearReg':
            mods.linearReg()
        elif inp == 'polynomReg':
            mods.polynomReg()
        elif inp == 'ridgeReg':
            mods.RidgeReg(data)
        elif inp == 'lassoReg':
            mods.LassoReg(data)
        elif inp == 'logisticReg':
            mods.LogisticReg(data)
        elif inp == 'learn':
            mods.tutor()
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
            fl = open('./assets/brain.txt', 'r').read()
            print(fl)
        elif inp in ['showna','show na','showNA']:
            opr.showNull()
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
            print(f"No such command found: {inp}")
            continue
    except Exception as err:
        print(f"Sorry, the following error occurred: {err}")
        continue

# "C:\Users\shubh\Downloads\speed.csv"
