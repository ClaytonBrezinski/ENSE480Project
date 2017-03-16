import sys
import tensorflow
import pandas

def __init__():
    trainingIterations = 100
    ignDataset = getPandasDataset("ign")
    ignDataset.shape()  #get the rows and columns of the dataset 
    return

def getPandasDataset(csvName):
    pandas.set_option('display.max_colWidth', 1000)
    return pandas.read_csv(csvName + '.csv')

