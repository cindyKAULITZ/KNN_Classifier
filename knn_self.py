import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import collections

iris = datasets.load_iris()
X = iris.data[:,:]
y = iris.target
train_X,test_X,train_y,test_y = train_test_split(X , y , test_size=0.5)

def distance(train, test):
    d = 0
    for i in range(0, 4):
        d += np.power((train[i] - test[i]), 4)
    d = np.power(d, 0.25)
    return d

def predict(testSet, trainSet, expected_y, k_size):
    w, h = 2, len(expected_y)
    temp = [[0 for x in range(w)] for y in range(h)] 
    predict_y = [0 for i in range(len(testSet))]

    for i in range(0, len(testSet)):
        for j in range(0, len(trainSet)):
            temp[j][0] = expected_y[j]
            temp[j][1] = distance(trainSet[j], testSet[i])

        temp = sorted(temp, key = lambda temp : temp[1])
        distance_sorted = temp[:k_size]
        distance_sorted = np.array(distance_sorted)
        distance_sorted = distance_sorted[:,:1]
        distance_sorted = distance_sorted.ravel()
        # print(distance_sorted)
        # print(collections.Counter(distance_sorted).most_common(1))
        c = collections.Counter(distance_sorted).most_common(1)
        c = np.array(c)
        # print('predict = ', c[0][0])
        predict_y[i] = c[0][0]
    # print(predict_y)
    return predict_y

def calculateError(mydata, realdata , times):
    Error = [0 for i in range(1, 22)]

    for avg in range(0, times):
        for k in range(1, 21):
            result_y = predict(mydata, train_X, train_y, k)

            for i in range(0, len(train_y)):
                if result_y[i] != realdata[i]:
                    Error[k] +=1
            # trainError[k] = train_y[k]/len(train_y)
    for k in range(1, 21):
        Error[k] = Error[k]/times
        Error[k] = Error[k]/len(train_y)

    return Error


a = calculateError(test_X,test_y, 1)
b = calculateError(train_X,train_y, 1)

new_ticks = np.linspace(1, 20, 20)
plt.xticks(new_ticks)
new_ticks2 = np.linspace(0, 1, 100)
plt.yticks(new_ticks2)

plt.plot(a[1:],'-o', label = 'Test Error')
plt.plot(b[1:], '-o', label = 'Train Error')
plt.legend(loc='upper right')
plt.show()


# trainError = [0 for i in range(1, 22)]
# testError = [0 for i in range(1, 22)]
# for k in range(1, 21):
#     resultTest_y = predict(test_X,train_X,train_y, k)
#     resultTrain_y = predict(train_X,train_X,train_y, k)
#     print(resultTest_y)
#     print(resultTrain_y)
#     for i in range(0, len(train_y)):
#         if resultTest_y[i] != test_y[i]:
#             testError[k] +=1
#         if resultTrain_y[i] != train_y[i]:
#             trainError[k] +=1

#     testError[k] = 1 - testError[k]/len(test_y)
#     trainError[k] = 1 - train_y[k]/len(train_y)

# print(testError)
# print(trainError)

