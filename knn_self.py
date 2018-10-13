import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets,metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
X = iris.data[:,:]
y = iris.target

train_X,test_X,train_y,test_y = train_test_split(X , y , test_size=0.5)
def knnClf(X, y, k_size):
    pass
#show different k
k_range = range(1, 20)
k_scores = []



# def plot_decision_regions(X, y, classifier, 
# 	test_idx=None, resolution=0.02):
# # setup marker generator and color map
#     markers = ('^', 'x', 'o')
#     colors = ('salmon', 'royalblue', 'springgreen')
#     cmap = ListedColormap(colors[:len(np.unique(test_y))])
# 
# # plot the decision surface
#     x1_min, x1_max = test_X[:, 0].min() - 1, test_X[:, 0].max() + 1
#     x2_min, x2_max = test_X[:, 1].min() - 1, test_X[:, 1].max() + 1
# #創矩陣xx1是x值(第一個attribute),xx2是y值(第二個attribute)
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
#                            np.arange(x2_min, x2_max, resolution))
#    # print(np.arange(x1_min, x1_max, resolution))
#    # print(xx1.ravel())
#     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     Z = Z.reshape(xx1.shape)
# #根據predict出來的結果顯示不同顏色
# #繪製predict出來的結果(背景圖)
#     plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())
#     plt.title("KNN (k = %i, weights = '%s')\n accuracy = %3f"
#               % (30, 'uniform',metrics.accuracy_score(test_y, clf.predict(test_X))))

# # plot the training point 
#     for idx, cl in enumerate(np.unique(y)):
# #畫點狀圖
#         plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
#                     alpha=0.8, c=cmap(idx),
#                     marker=markers[idx], label=cl,edgecolors='black',s=26)             
#     plt.show()
# plot_decision_regions(X, y, clf)
