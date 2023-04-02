import numpy as numpy
import pandas as pd
import os,socket,time


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
    means={}
    for cols in columns:
        if cols not in data.columns:
            print(f"there is not column like this")
        else:
            means[cols] = data[cols].mean()
    print(means)

def showHead(data):
    print(data.head())

def showTail(data):
    print(data.tail())

def describeCsv(data):
    print(data.describe())



def getPrompt():
    prompt = "𝖇𝖗𝖆𝖎𝖓𝖜𝖆𝖛𝖊>> \b\b\b\b\b\b\b\b\b\b"
    return prompt

# here we start alright jerry
csv_path = input("enter the path of csv file : ").replace("\"","")
data = pd.read_csv(csv_path)


while True:
    try:

        inp = input(getPrompt())
        if inp == 'mean':
            mean(data)
        elif inp=='head':
            showHead(data)
        elif inp=='tail':
            showTail(data)
        elif inp=='describe':
            describeCsv(data)
        else:
            break
    except Exception as err:
        print("sorry following error occurred")



