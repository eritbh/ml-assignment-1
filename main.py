# read our data
import pandas as pd
data = pd.read_csv('vertebral_column_data/column_3C_with_headers.dat', sep=' ')

# create a mapping from category ID to category name
lookup_category = dict(zip(data.category_id.unique(), data.category.unique()))
print(lookup_category)

from sklearn.neighbors import KNeighborsClassifier

# initialize k, p, and partition values used in practical calculations
KPRACTICE = 4
PPRACTICE = 2
PERPRACTICE = 0.25

# create a 3NN classifier for our data
knn = KNeighborsClassifier(n_neighbors=KPRACTICE, p=PPRACTICE)
X = data[[
    'pelvic_incidence',
    'pelvic_tilt',
    'lumbar_lordosis_angle',
    'sacral_slope',
    'pelvic_radius',
    'spondylolisthesis_grade',
]]
y = data['category_id']
knn.fit(X, y)

# make a prediction for another data point
unknown1 = pd.DataFrame([[
    50,
    15,
    40,
    35,
    110,
    2,
]], columns=[
    'pelvic_incidence',
    'pelvic_tilt',
    'lumbar_lordosis_angle',
    'sacral_slope',
    'pelvic_radius',
    'spondylolisthesis_grade',
])
prediction = knn.predict(unknown1)
print(prediction[0])
print(knn.predict_proba(unknown1))

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PERPRACTICE, random_state=42)

# Train the classifier (fit the estimator) using the training data
knn.fit(X_train, y_train)

# Estimate the accuracy of the classifier on future data, using the test data
knn.score(X_test, y_test)

# How sensitive is k-NN classification accuracy to the choice of the 'k' parameter?
k_range = range(1, 20)
kScores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, p=PPRACTICE)
    knn.fit(X_train, y_train)
    kScores.append(knn.score(X_test, y_test))

# What 'p' value (degree of minkowski distance) results in the highest accuracy?
p_range = range(1, 8)
pScores = []
for pVal in p_range:
    knn = KNeighborsClassifier(n_neighbors=KPRACTICE, p=pVal)
    knn.fit(X_train, y_train)
    pScores.append(knn.score(X_test, y_test))

# visualization
# plotting a scatter matrix
from matplotlib import cm
from pandas.plotting import scatter_matrix
cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X, c=y, marker='o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

# plotting a 3D scatter plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d   # must keep
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X['pelvic_incidence'], X['pelvic_tilt'], X['pelvic_radius'], c = y, marker = 'o', s=100)
ax.set_xlabel('pelvic_incidence')
ax.set_ylabel('pelvic_tilt')
ax.set_zlabel('pelvic_radius')

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, kScores)
plt.xticks([0, 5, 10, 15, 20])

plt.figure()
plt.xlabel('p')
plt.ylabel('accuracy')
plt.scatter(p_range, pScores)
plt.xticks([0, 2, 4, 6, 8])

# How sensitive is k-NN classification accuracy to the train/test split proportion?
import numpy as np
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
knn = KNeighborsClassifier(n_neighbors=kpractice, p=PPRACTICE)
plt.figure()
for s in t:
    scores = []
    for i in range(1, 1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')
plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy')
plt.show()
