import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib.pyplot import *

def readFile():
    with open("spam.data.txt","r") as Myfile:
        Mydata = Myfile.read()
        parseData = Mydata.split("\n")[:-1]
        parseData = [line.split() for line in parseData]
        np.random.shuffle(parseData)

        test = parseData[:1000]
        train = parseData[1000:] #rest of data


        test_Labels = [float(line[-1]) for line in test]
        test_Data = [list(map(float,line[:-1])) for line in test]
        train_Data = [list(map(float,line[:-1])) for line in train]
        train_Lables = [float(line[-1]) for line in train]

    return test_Data, test_Labels, train_Data, train_Lables


def fitLogisticReg(test_Data, test_Labels, train_Data, train_Lables):

    logisticReg = LogisticRegression()
    logisticReg.fit(train_Data,train_Lables)
    probsArray = logisticReg.predict_proba(test_Data)
    sort = np.argsort(probsArray.T[1])[::-1]

    test_Labels = np.array(test_Labels)
    test_Labels = test_Labels[sort]
    tpr_vals = [0]
    fpr_vals = [0]
    np_positives = np.count_nonzero(test_Labels)
    nn_negatives = 1000 - np_positives
    labels = test_Labels.cumsum()

    for i in range(1,np_positives+1):
        Ni = np.where(labels == i)[0][0]+1
        tpr_vals.append(i/np_positives)
        fpr_vals.append((Ni-i)/nn_negatives)

    tpr_vals.append(1)
    fpr_vals.append(1)
    plot(fpr_vals,tpr_vals)
    title("tpr vs ftr")
    xlabel("FTR")
    ylabel("TPR")
    show()


x,y,w,z =readFile()
fitLogisticReg(x,y,w,z)






