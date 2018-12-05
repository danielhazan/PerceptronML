import numpy as np
from sklearn.svm import SVC
from matplotlib.pyplot import *

class Perceptron:
    def __init__(self,feature_num):
        self.weightVec = np.zeros(feature_num)
        self.features = feature_num



    def predict(self,x):
        inner_prod = np.dot(self.weightVec,x)
        return 1 if inner_prod>=0 else -1

    def fit(self,X,Y):
        d = np.zeros(self.features)
        for k in range(len(Y)):
            x = X[k]
            y= self.predict(x)
            if(Y[k]*y<=0):
                self.weightVec = self.weightVec + y*x

            #error = Y[k] - y
            #d[k] = error
            #self.weightVec = self.weightVec + d

        return self.weightVec


    def score(self,X_Test,Y_Test):
        count = 0

        for i in range(len(X_Test)):
            #check if classified correctly-->
            if(np.sign(np.matmul(X_Test,self.weightVec))[i]== Y_Test[i]):
                count+=1
        return count/len(X_Test)



def draw_rect():
    rect_num = np.random.randint(2,size=1)
    if(rect_num==1):
        x= np.random.uniform(-3,1)
        y= np.random.randint(1,3)
    else:
        rect_num = -1
        x = np.random.uniform(-1, 3)
        y = np.random.randint(-3, -1)
    return x,y,rect_num


def normal_dist(sizeOfSample):
    X = np.random.multivariate_normal(np.zeros(2),np.identity(2),sizeOfSample)
    w = [0.3,-0.5]
    classify = [np.sign(np.dot(x,w)) for x in X ]#??????
    return X,classify


def draw_D2_points(size):
    recs = [[-3, 1, 1, 3], [-1, 3, -3, -1]]
    points = list()
    y_vec = list()
    for i in range(size):
        y_label = np.random.choice([0, 1])
        rec = recs[y_label]
        x, y = np.random.uniform(rec[0], rec[1]), np.random.uniform(rec[2], rec[3])
        points.append([x, y])
        y_vec.append(-1 if y_label == 0 else y_label)
    return np.array(points), y_vec
"""def draw_D2_points(sizeOfSample):
    x_D2=np.empty((0,2), int)
    y_D2= np.empty((0,1), int)
    for point in range(sizeOfSample):
        x,y,rect_num = draw_rect()
        x_D2 = np.append(x_D2,np.array([[x,y]]),axis=0)
        y_D2 = np.append(y_D2,rect_num)
    return x_D2, y_D2
"""


svm_accuracy_D1 = 0
svm_accuracy_D2 = 0
svm_accuracy_D1_Array = []
svm_accuracy_D2_Array = []
perceptron_accuracy_D1 = 0
perceptron_accuracy_D2 = 0
perceptron_accuracy_D1_Array = []
perceptron_accuracy_D2_Array = []
svm_D1 = SVC(C=1e10, kernel='linear')
svm_D2 = SVC(C=1e10, kernel='linear')
# use fit, score
perceptron_D1 = Perceptron(2)
perceptron_D2 = Perceptron(2)
for m in [5,10,15,25,70]:
    print ("im here1")
    for iter in range(500):
        #D1
        X_train_D1,y_train_D1 = normal_dist(m)
        while(len(np.unique(y_train_D1))<2):
            X_train_D1, y_train_D1 = normal_dist(m)
        X_test_D1, y_test_D1 = normal_dist(10000)
        #D2
        X_train_D2, y_train_D2 = draw_D2_points(m)
        while(len(np.unique(y_train_D2))<2):
            X_train_D2, y_train_D2 = draw_D2_points(m)
        X_test_D2, y_test_D2 = draw_D2_points(10000)
        #train perceptron
        perceptron_D1.fit(X_train_D1,y_train_D1)
        perceptron_D2.fit(X_train_D2,y_train_D2)
        #train svm
        svm_D1.fit(X_train_D1,y_train_D1)
        svm_D2.fit(X_train_D2,y_train_D2)
        #accuracy

        svm_accuracy_D1 += svm_D1.score(X_test_D1,y_test_D1)
        svm_accuracy_D2 += svm_D2.score(X_test_D2,y_test_D2)

        perceptron_accuracy_D1 += perceptron_D1.score(X_test_D1,y_test_D1)
        perceptron_accuracy_D2 += perceptron_D2.score(X_test_D2,y_test_D2)



    svm_accuracy_D1_Array.append(svm_accuracy_D1/500)
    svm_accuracy_D2_Array.append(svm_accuracy_D2/500)
    perceptron_accuracy_D1_Array.append(svm_accuracy_D1/500)
    perceptron_accuracy_D2_Array.append(svm_accuracy_D2/500)

#print ("im here2")
plot([5,10,15,25,70],svm_accuracy_D1_Array,label="SVM")
plot([5,10,15,25,70],perceptron_accuracy_D1_Array,label="PERCEPTRON")

xlabel("Sample size")
ylabel("Accuracy")
title("Accuracy vs number of samples")
legend()
show()


plot([5,10,15,25,70],svm_accuracy_D2_Array,label="SVM")
plot([5,10,15,25,70],perceptron_accuracy_D2_Array,label="PERCEPTRON")

xlabel("Sample size")
ylabel("Accuracy")
title("Accuracy vs number of samples")
legend()
show()




"""
class Perceptron(object):

    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size+1)
        # add one for bias
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        #return (x >= 0).astype(np.float32)
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * x
"""