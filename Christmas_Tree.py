import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def readData(address):
    with open(address) as csvFile:
        reader = csv.reader(csvFile)
        data = [row for row in reader]
    return data


def cleanData(data):
    return list(filter(lambda thisList: False if '?' in thisList else True, data))


def dummyFeature(features):
    for column in range(features.shape[1]):
        # 0,1,2,3,...,21
        featureStatus = set(features[:, column])
        tranasformer = preprocessing.LabelEncoder()
        tranasformer.fit(list(featureStatus))
        features[:, column] = tranasformer.transform(features[:, column])
    return features


# Read Data
fileAddress = './train+dev+test.csv'
data = readData(fileAddress)
print(len(data))
data = cleanData(data)
print(len(data))
data = np.array(data)[1:]

# Seprate Feature and Lable
X = data[:, 1:]
Y = data[:, 0]
X = dummyFeature(X)

# split train and test subset I use 70% & 30%
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=24)

# Define Decision Thee
tree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
tree.fit(X_train, y_train)
predicted = tree.predict(X_test)

# figure out my tree accuracy
accuracy = classification_report(y_test, predicted)
print(accuracy)