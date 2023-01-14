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


def dummyVariables(features):
    for column in range(features.shape[1]):
        # 0,1,2,3,...,21
        featureStatus = set(features[:, column])
        tranasformer = preprocessing.LabelEncoder()
        tranasformer.fit(list(featureStatus))
        features[:, column] = tranasformer.transform(features[:, column])
    return features


def bootstrap(X, Y):
    dataset = np.column_stack((X, Y))
    newDataset = dataset[np.random.choice(
        dataset.shape[0], size=dataset.shape[0])]
    new_X = newDataset[:, :-1]
    new_Y = newDataset[:, -1]
    return new_X, new_Y


def majority(vote):
    vote=list(vote)
    # find most frequent element in a list
    return max(set(vote), key=vote.count)


# Read Data
fileAddress = './train+dev+test.csv'
data = readData(fileAddress)
data = cleanData(data)
data = np.array(data)[1:]  # remove headers

# Seprate Feature and Lable
X = data[:, 1:]
Y = data[:, 0]
X = dummyVariables(X)

# split train and test subset I use 70% & 30%
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=24)

# make bootstrap subset
NUMBER_OF_BOOTSTRAP = 5
bootstrapDataset = [bootstrap(X_train, y_train)
                    for _ in range(NUMBER_OF_BOOTSTRAP)]

# list bagging classifires
classifiers = []
for index in range(NUMBER_OF_BOOTSTRAP):
    # Define Decision Thee
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    tree.fit(*bootstrapDataset[index])
    classifiers.append(tree)

# Voting from all classifiers
votes = [tree.predict(X_test) for tree in classifiers]
votes = np.array(votes)
votes = np.transpose(votes)

# Finding the majority vote
predicted_Y = [majority(vote) for vote in votes]


# figure out my tree accuracy
accuracy = classification_report(y_test, predicted_Y)
print(accuracy)
