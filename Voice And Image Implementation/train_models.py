import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')
# Read dataset from online webserver
dataset = pd.read_csv('https://www.akifagoelectronics.com/UoW_2065655_Project/heart.csv')
# Replace some column values to integers
dataset.loc[dataset['Sex'] == 'M', 'Sex'] = 1
dataset.loc[dataset['Sex'] == 'F', 'Sex'] = 0
dataset.loc[dataset['ChestPainType'] == 'TA', 'ChestPainType'] = 0
dataset.loc[dataset['ChestPainType'] == 'ATA', 'ChestPainType'] = 1
dataset.loc[dataset['ChestPainType'] == 'NAP', 'ChestPainType'] = 2
dataset.loc[dataset['ChestPainType'] == 'ASY', 'ChestPainType'] = 3
dataset.loc[dataset['RestingECG'] == 'Normal', 'RestingECG'] = 1
dataset.loc[dataset['RestingECG'] == 'ST', 'RestingECG'] = 2
dataset.loc[dataset['RestingECG'] == 'LVH', 'RestingECG'] = 3
dataset.loc[dataset['ExerciseAngina'] == 'N', 'ExerciseAngina'] = 0
dataset.loc[dataset['ExerciseAngina'] == 'Y', 'ExerciseAngina'] = 1
dataset.loc[dataset['ST_Slope'] == 'Up', 'ST_Slope'] = 1
dataset.loc[dataset['ST_Slope'] == 'Flat', 'ST_Slope'] = 2
dataset.loc[dataset['ST_Slope'] == 'Down', 'ST_Slope'] = 3

# drop the target column
predictors = np.array(dataset.drop("HeartDisease", axis=1))

# get the target variable
target = dataset["HeartDisease"].to_numpy()

# split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

# Logistic Regression
regression_model = LogisticRegression()
regression_model.fit(X_train, Y_train)
Y_pred_lr = regression_model.predict(X_test)
score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)
# Save trained model
pickle.dump(regression_model, open('hdp_linear.sav', 'wb'))
print("The accuracy score achieved using Logistic Regression is: " + str(score_lr) + " %")

# SVM Model Training
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, Y_train)
Y_pred_svm = svm_model.predict(X_test)
score_svm = round(accuracy_score(Y_pred_svm, Y_test) * 100, 2)
# Save trained model
pickle.dump(svm_model, open('hdp_svm.sav', 'wb'))
print("The accuracy score achieved using Linear SVM is: " + str(score_svm) + " %")

# K - Nearest Neighbor classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, Y_train)
score = cross_val_score(knn_classifier, X_train, Y_train, cv=10)
y_pred_knn = knn_classifier.predict(X_test)
print("The accuracy score achieved using KNN is: ", accuracy_score(Y_test, y_pred_knn))
print("The mean score achieved using tuned KNN is: ", score.mean())
# hyper parameter tuning
knn_classifier = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                      metric_params=None, n_jobs=1, n_neighbors=5, p=1, weights='uniform')
knn_classifier.fit(X_train, Y_train)
score = cross_val_score(knn_classifier, X_train, Y_train, cv=10)
y_pred_knn = knn_classifier.predict(X_test)
print("The accuracy score achieved using tuned KNN is: ", accuracy_score(Y_test, y_pred_knn))
print("The mean score achieved using tuned KNN is: ", score.mean())
# accuracy increases after Hyper parameter tuning
pickle.dump(knn_classifier, open('hdp_knn.pkl', 'wb'))

# Neural Network
# https://stats.stackexchange.com/a/136542 helped a lot in avoiding over fitting
model = Sequential()
model.add(Dense(11, activation='relu', input_dim=11))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# convert a NumPy array to a Tensor (Unsupported object type int)
X_train = np.asarray(X_train).astype(np.int)
Y_train = np.asarray(Y_train).astype(np.int)
X_test = np.asarray(X_test).astype(np.int)
Y_test = np.asarray(Y_test).astype(np.int)
# train Neural Network model
model.fit(X_train, Y_train, epochs=500)
Y_pred_nn = model.predict(X_test)
rounded = [round(x[0]) for x in Y_pred_nn]
Y_pred_nn = rounded
score_nn = round(accuracy_score(Y_pred_nn, Y_test) * 100, 2)
print("The accuracy score achieved using Neural Network is: " + str(score_nn) + " %")
# Save trained model
model.save('hdp_keras.h5')

# Note: Accuracy of 85% can be achieved on the test set, by setting epochs=2000, and number of nodes = 11.
