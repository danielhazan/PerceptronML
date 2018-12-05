import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt

class Knn():

    def __init__(self,k):
        self.k = k
        self.samples = None
        self.labels = None

    def fit(self, samples,labels):
        self.samples = samples
        self.labels = labels

    def predict(self,X  ):
        #calculate the distance between samples and the new point x
        distance_Arr = []
        for sample in self.samples:
            distance_Arr.append(distance.euclidean(X,sample))
        distance_Arr = np.array(distance_Arr)

        #getting the k nearest neighbors by argsort

        distance_min = np.argsort(distance_Arr)[:self.k]

        count_of_ones = 0
        count_of_zeroes = 0
        for i in distance_min:
            if self.labels[i] == 1:
                count_of_ones += 1
            else:
                count_of_zeroes += 1

        #the prediction is the greater value between the labels
        if count_of_ones> count_of_zeroes:
            return 1
        else:
            return 0


def read_data():

     with open("spam.data.txt", "r") as MyFile:
        MyData = MyFile.read()
        samples_Array = MyData.split("\n")[:-1]

        samples_Array = [line.split() for line in samples_Array]
        np.random.shuffle(samples_Array)

        train_d = samples_Array[1000:]
        test_d = samples_Array[:1000]

        train_sample = [list(map(float,line[:-1])) for line in train_d]
        train_labels = [float(line[-1]) for line in train_d]
        test_sample = [ list(map(float, line[:-1])) for line in test_d]
        test_labels = [float(line[-1]) for line in test_d]

        return [train_sample, train_labels, test_sample, test_labels]


k_values = [1,2,5,10,100]

error = []

for i in k_values:
    X_tr_sam, Y_tr_lab , X_tes_sam, Y_tes_lab = read_data()
    knn = Knn(i)
    knn.fit(X_tr_sam,Y_tr_lab)

    count = 0
    for j in range(len(X_tes_sam)):
        print(str(j) + "\n")

        pred = knn.predict(X_tes_sam[j])
        if pred != Y_tes_lab[j]:
            count +=1

    error.append(count/1000)


plt.plot(k_values,error, "purple")
plt.title("error of Knn on k=1,2,5,10,100")
plt.xlabel("k vals")
plt.ylabel("error of knn")
plt.show()
