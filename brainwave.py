import numpy as np
import pandas as pd
import os, time

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

def showHead(data):
    print(data.head())

def showTail(data):
    print(data.tail())

def describeCsv(data):
    print(data.describe())

def showColumns(data):
    print(data.columns)

def getPrompt():
    prompt = "𝖇𝖗𝖆𝖎𝖓𝖜𝖆𝖛𝖊>> \b\b\b\b\b\b\b\b\b\b"
    return prompt

#apply prediction stuff here 
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
    if choice == '1':
        print('1')
    elif choice == '2':
        print('2')
    else:
        print('not valid')


###################################
##        predictions            ##
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
        elif inp == 'exit':
            print('hey i see yah man next time i\'ll be more cool than now i am')
            break
        else:
            continue
    except Exception as err:
        print(f"Sorry, the following error occurred: {err}")
        continue
