# read our data
import pandas as pd
data = pd.read_csv('vertebral_column_data/column_3C_with_headers.dat', sep=' ')

# create a mapping from category ID to category name
# lookup_category = dict(zip(data.category_id.unique(), data.category.unique()))
# print(lookup_category)

from sklearn.neighbors import KNeighborsClassifier

# initialize k, p, and partition values used in practical calculations
KPRACTICE = 4
PPRACTICE = 2
PERPRACTICE = 0.20

# a constant list of attributes
ATTRIBUTES = [
    'pelvic_incidence',
    'pelvic_tilt',
    'lumbar_lordosis_angle',
    'sacral_slope',
    'pelvic_radius',
    'spondylolisthesis_grade',
]

# create a 3NN classifier for our data
knn = KNeighborsClassifier(n_neighbors=KPRACTICE, p=PPRACTICE)
X = data[ATTRIBUTES]
y = data['category_id']
knn.fit(X, y)

# Create report on data
import numpy as np
print("This dataset has 6 attributes\n")
print("The class distribution of this dataset is:")
for class_name in data['category'].unique():
    rows_matching_class = data[data['category'] == class_name]
    print(f'\tClass {class_name} contains {len(rows_matching_class)} data points')
print(f"Our data is partitioned to {100 * (1 - PERPRACTICE)}% training, {100 * PERPRACTICE}% Testing")
print(f"Our distance calculations use the Minkowski distance function with p={PPRACTICE}")

print("More information about each class:")

# repeat for each unique class
for class_name in data['category'].unique():
    print(f"\nClass {class_name}:")

    # get the rows matching this class
    rows_matching_class = data[data['category'] == class_name]
    # remove the class columns from the data since we don't need it
    rows_attributes_only = rows_matching_class[ATTRIBUTES]

    # calculate statistics
    print("\nMean data point in class:")
    print(rows_attributes_only.mean())
    print("\nMedian data point in class:")
    print(rows_attributes_only.mean())
    print("\nMinimums in class:")
    print(rows_attributes_only.min())
    print("\nMaximums in class:")
    print(rows_attributes_only.mean())

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 20% testing, 80% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PERPRACTICE, random_state=42)

# Train the classifier (fit the estimator) using the training data
knn.fit(X_train, y_train)

# Estimate the accuracy of the classifier on future data, using the test data
print(f"\nOverall model accuracy: {100 * knn.score(X_test, y_test):.2f}%")

# Confusion Matrix
from sklearn.metrics import confusion_matrix
y_prediction = knn.predict(X_test)
confusion_matrix_result = confusion_matrix(y_test, y_prediction)
print("Confusion Matrix:\n", confusion_matrix_result)

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
# from mpl_toolkits.mplot3d import axes3d. must keep

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
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
knn = KNeighborsClassifier(n_neighbors=KPRACTICE, p=PPRACTICE)
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
