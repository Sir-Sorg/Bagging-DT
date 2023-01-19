import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
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


def standardization(data):
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    STD_data = scaler.transform(data)
    return STD_data


# Read Data
fileAddress = './train+dev+test.csv'
data = readData(fileAddress)
data = cleanData(data)
data = np.array(data)[1:]  # remove headers

# Seprate Feature and Lable
X = data[:, 1:]
Y = data[:, 0]
X = dummyVariables(X)

# Standarding Data
X = standardization(X)

# split train and test subset I use 70% & 30%
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=24)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

# Defind and Train Classifire
logisticRegression = LogisticRegression(solver='liblinear')
logisticRegression.fit(X_train, y_train)
yhat = logisticRegression.predict(X_test)

# figure out my tree accuracy
print(classification_report(y_test, yhat))