# Importing essential libraries
from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
from joblib import dump, load
from pathlib import Path
import requests
import pandas as pd
# Import essential libraries for model training
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Logistic Regression
log_regression_model = LogisticRegression()
    
# SVM Model Training
svm_model = svm.SVC(kernel='linear')
    
# K - Nearest Neighbor classifier hyper parameter tuning
knn_classifier = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=1, weights='uniform')
    
# Naive Bayes model
naive_bayes_model = GaussianNB()
    
# Tuned MLPClassifier model
mlp_model = MLPClassifier(solver='adam', hidden_layer_sizes=(20,), learning_rate='constant', activation="relu", random_state=0, max_iter=2000)
   
# Train random Forest
ran_classifier = RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=8, criterion='gini')

# assign the number of estimators
num_estimators = 22

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/bagging', methods=['GET','POST'])
def bagging():
    # Create empty list to store models performance
    model_performance = []
    
    # create the sub-models
    estimators = []
   
    # read dataset csv file
    dataset = pd.read_csv('heart.csv')

    # remove any null values from data rows
    dataset = dataset.dropna()

    # get the column names
    column_names = dataset.columns

    # using label encoder
    le = LabelEncoder()

    # save previous dataset
    dataset.to_csv('prev_heart.csv', index=False)

    # scan for columns with non-numerical labels
    for column in column_names:
        # checks if the first label is string
        if isinstance(dataset[column][0], str):
            # transform to numerical labels
            dataset[column] = le.fit_transform(dataset[column])

    # save refined dataset
    dataset.to_csv('new_heart.csv', index=False)
            
    # read new dataset csv file
    dataset = pd.read_csv('new_heart.csv')
            
    # drop the target column
    predictors = dataset.drop("HeartDisease", axis=1)

    # get the target variable
    target = dataset["HeartDisease"]

    # split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

    # Applying Bagging Techniques
    model = BaggingClassifier(base_estimator=log_regression_model, n_estimators=num_estimators)
    model.fit(X_train,Y_train)
    prediction = model.predict(X_test)
    accuracy = round(accuracy_score(prediction,Y_test) * 100, 2)
    # get the model performance score
    #accuracy = round(ran_classifier.score(X_test, Y_test) * 100, 2)
    f1 = round(f1_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, model.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Logistic Regression', str(str(round(model.score(X_train, Y_train) * 100, 2)) + " %").upper(),\
        str(str(round(model.score(X_test, Y_test) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != model.predict(X_test)).sum()))])
    ###
    model = BaggingClassifier(base_estimator=svm_model, n_estimators=num_estimators)
    model.fit(X_train,Y_train)
    prediction = model.predict(X_test)
    accuracy = round(accuracy_score(prediction,Y_test) * 100, 2)
    # get the model performance score
    #accuracy = round(ran_classifier.score(X_test, Y_test) * 100, 2)
    f1 = round(f1_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, model.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Support Vector Machine', str(str(round(model.score(X_train, Y_train) * 100, 2)) + " %").upper(),\
        str(str(round(model.score(X_test, Y_test) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != model.predict(X_test)).sum()))])
    ###
    model = BaggingClassifier(base_estimator=knn_classifier, n_estimators=num_estimators)
    model.fit(X_train,Y_train)
    prediction = model.predict(X_test)
    accuracy = round(accuracy_score(prediction,Y_test) * 100, 2)
    # get the model performance score
    f1 = round(f1_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, model.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['K-Nearest Neighbours', str(str(round(model.score(X_train, Y_train) * 100, 2)) + " %").upper(),\
        str(str(round(model.score(X_test, Y_test) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != model.predict(X_test)).sum()))])
    ###
    model = BaggingClassifier(base_estimator=naive_bayes_model, n_estimators=num_estimators)
    model.fit(X_train,Y_train)
    prediction = model.predict(X_test)
    accuracy = round(accuracy_score(prediction,Y_test) * 100, 2)
    # get the model performance score
    f1 = round(f1_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, model.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Naive Bayes', str(str(round(model.score(X_train, Y_train) * 100, 2)) + " %").upper(),\
        str(str(round(model.score(X_test, Y_test) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != model.predict(X_test)).sum()))])
    ###
    model = BaggingClassifier(base_estimator=mlp_model, n_estimators=num_estimators)
    model.fit(X_train,Y_train)
    prediction = model.predict(X_test)
    accuracy = round(accuracy_score(prediction,Y_test) * 100, 2)
    # get the model performance score
    f1 = round(f1_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, model.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Multi-Layer Perception', str(str(round(model.score(X_train, Y_train) * 100, 2)) + " %").upper(),\
        str(str(round(model.score(X_test, Y_test) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != model.predict(X_test)).sum()))])

    ##
    model = BaggingClassifier(base_estimator=ran_classifier, n_estimators=num_estimators)
    model.fit(X_train,Y_train)
    prediction = model.predict(X_test)
    accuracy = round(accuracy_score(prediction,Y_test) * 100, 2)
    # get the model performance score
    f1 = round(f1_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, model.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Random Forest', str(str(round(model.score(X_train, Y_train) * 100, 2)) + " %").upper(),\
        str(str(round(model.score(X_test, Y_test) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != model.predict(X_test)).sum()))])
        
    # Hybrid model
    estimators.append(('random forest', ran_classifier))
    estimators.append(('nbs', naive_bayes_model))
    # Defining the ensemble model
    ensemble = VotingClassifier(estimators)
    ensemble.fit(X_train, Y_train)
    prediction = ensemble.predict(X_test)
    accuracy = round(accuracy_score(prediction,Y_test) * 100, 2)
    f1 = round(f1_score(Y_test, ensemble.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, ensemble.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, ensemble.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Proposed Hybrid Model', str(str(round(ensemble.score(X_train, Y_train) * 100, 2)) + " %").upper(),\
        str(str(round(ensemble.score(X_test, Y_test) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != ensemble.predict(X_test)).sum()))])
    
    # convert list to dataframe                    
    df = pd.DataFrame(model_performance, columns = ['Trained Model', 'Training Score', 'Testing Score', 'F1 Score', 'Precision Score', 'Recall Score', 'Mislabelled Points'])
    
    # return dataframe table
    return render_template('evaluation.html',tables= [df.to_html(classes='data')],titles=['na','Trained Machine learning Models Using Bagging Technique']) 
    

@app.route('/boosting', methods=['GET','POST'])
def boosting():
    # Create empty list to store models performance
    model_performance = []
    
    # create the sub-models
    estimators = []
   
    # read dataset csv file
    dataset = pd.read_csv('heart.csv')

    # remove any null values from data rows
    dataset = dataset.dropna()

    # get the column names
    column_names = dataset.columns

    # using label encoder
    le = LabelEncoder()

    # save previous dataset
    dataset.to_csv('prev_heart.csv', index=False)

    # scan for columns with non-numerical labels
    for column in column_names:
        # checks if the first label is string
        if isinstance(dataset[column][0], str):
            # transform to numerical labels
            dataset[column] = le.fit_transform(dataset[column])

    # save refined dataset
    dataset.to_csv('new_heart.csv', index=False)
            
    # read new dataset csv file
    dataset = pd.read_csv('new_heart.csv')
            
    # drop the target column
    predictors = dataset.drop("HeartDisease", axis=1)

    # get the target variable
    target = dataset["HeartDisease"]

    # split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

    # Applying Bagging Techniques
    model = AdaBoostClassifier(base_estimator=log_regression_model, n_estimators=num_estimators)
    model.fit(X_train,Y_train)
    prediction = model.predict(X_test)
    accuracy = round(accuracy_score(prediction,Y_test) * 100, 2)
    # get the model performance score
    f1 = round(f1_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, model.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Logistic Regression', str(str(round(model.score(X_train, Y_train) * 100, 2)) + " %").upper(),\
        str(str(round(model.score(X_test, Y_test) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != model.predict(X_test)).sum()))])
    ###
    model = AdaBoostClassifier(svm.SVC(probability=True,kernel='linear'),n_estimators=num_estimators, learning_rate=1.0, algorithm='SAMME')
    model.fit(X_train,Y_train)
    prediction = model.predict(X_test)
    accuracy = round(accuracy_score(prediction,Y_test) * 100, 2)
    # get the model performance score
    f1 = round(f1_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, model.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Support Vector Machine', str(str(round(model.score(X_train, Y_train) * 100, 2)) + " %").upper(),\
        str(str(round(model.score(X_test, Y_test) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != model.predict(X_test)).sum()))])
    ###
    ###
    # KNN doesn't support sample_weight.
    ###
    # store the models performance text
    model_performance.append(['K-Nearest Neighbours', 'Null','Null','Null','Null','Null','Null'])
    
    ###
    model = AdaBoostClassifier(base_estimator=naive_bayes_model, n_estimators=num_estimators)
    model.fit(X_train,Y_train)
    prediction = model.predict(X_test)
    accuracy = round(accuracy_score(prediction,Y_test) * 100, 2)
    # get the model performance score
    f1 = round(f1_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, model.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Naive Bayes', str(str(round(model.score(X_train, Y_train) * 100, 2)) + " %").upper(),\
        str(str(round(model.score(X_test, Y_test) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != model.predict(X_test)).sum()))])
    
    ###
    # MLPClassifier doesn't support sample_weight.
    # store the models performance text
    model_performance.append(['Multi-Layer Perception', 'Null','Null','Null','Null','Null','Null'])

    ##
    model = AdaBoostClassifier(base_estimator=ran_classifier, n_estimators=num_estimators)
    model.fit(X_train,Y_train)
    prediction = model.predict(X_test)
    accuracy = round(accuracy_score(prediction,Y_test) * 100, 2)
    # get the model performance score
    f1 = round(f1_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, model.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, model.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Random Forest', str(str(round(model.score(X_train, Y_train) * 100, 2)) + " %").upper(),\
        str(str(round(model.score(X_test, Y_test) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != model.predict(X_test)).sum()))])
        
    # Hybrid model
    estimators.append(('random forest', ran_classifier))
    estimators.append(('nbs', naive_bayes_model))
    # Defining the ensemble model
    ensemble = VotingClassifier(estimators)
    ensemble.fit(X_train, Y_train)
    prediction = ensemble.predict(X_test)
    accuracy = round(accuracy_score(prediction,Y_test) * 100, 2)
    f1 = round(f1_score(Y_test, ensemble.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, ensemble.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, ensemble.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Proposed Hybrid Model', str(str(round(ensemble.score(X_train, Y_train) * 100, 2)) + " %").upper(),\
        str(str(round(ensemble.score(X_test, Y_test) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != ensemble.predict(X_test)).sum()))])
    
    # convert list to dataframe                    
    df = pd.DataFrame(model_performance, columns = ['Trained Model', 'Training Score', 'Testing Score', 'F1 Score', 'Precision Score', 'Recall Score', 'Mislabelled Points'])
    
    # return dataframe table
    return render_template('evaluation.html',tables= [df.to_html(classes='data')],titles=['na','Trained Machine learning Models Using Boosting Technique']) 
    
    
@app.route('/retrain', methods=['GET','POST'])
def retrain():
    # Create empty list to store models performance
    model_performance = []
    
    # create the sub-models
    estimators = []
    
    # read dataset csv file
    dataset = pd.read_csv('heart.csv')

    # remove any null values from data rows
    dataset = dataset.dropna()

    # get the column names
    column_names = dataset.columns

    # using label encoder
    le = LabelEncoder()

    # save previous dataset
    dataset.to_csv('prev_heart.csv', index=False)

    # scan for columns with non-numerical labels
    for column in column_names:
        # checks if the first label is string
        if isinstance(dataset[column][0], str):
            # transform to numerical labels
            dataset[column] = le.fit_transform(dataset[column])

    # save refined dataset
    dataset.to_csv('new_heart.csv', index=False)
            
    # read new dataset csv file
    dataset = pd.read_csv('new_heart.csv')
            
    # drop the target column
    predictors = dataset.drop("HeartDisease", axis=1)

    # get the target variable
    target = dataset["HeartDisease"]

    # split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

    # Logistic Regression
    log_regression_model.fit(X_train, Y_train)
    # get the model performance score
    f1 = round(f1_score(Y_test, log_regression_model.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, log_regression_model.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, log_regression_model.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Logistic Regression', str(str(round(accuracy_score(Y_train, log_regression_model.predict(X_train)) * 100, 2)) + " %").upper(),\
        str(str(round(accuracy_score(Y_test, log_regression_model.predict(X_test)) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != log_regression_model.predict(X_test)).sum()))])
        
    # Save trained model
    #pickle.dump(log_regression_model, open('hdp_logistic.sav', 'wb'), protocol=0)

    # SVM Model Training
    svm_model.fit(X_train, Y_train)
    # get the model performance score
    f1 = round(f1_score(Y_test, svm_model.predict(X_test), average="macro") * 100, 2) 
    precision = round(precision_score(Y_test, svm_model.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, svm_model.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Support Vector Machine', str(str(round(accuracy_score(Y_train, svm_model.predict(X_train)) * 100, 2)) + " %").upper(),\
        str(str(round(accuracy_score(Y_test, svm_model.predict(X_test)) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != svm_model.predict(X_test)).sum()))])
        
    # Save trained model
    #pickle.dump(svm_model, open('hdp_svm.sav', 'wb'), protocol=0)

    # K - Nearest Neighbor classifier
    knn_classifier.fit(X_train, Y_train)
    score = cross_val_score(knn_classifier, X_train, Y_train, cv=10)
    # get the model performance score
    f1 = round(f1_score(Y_test, knn_classifier.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, knn_classifier.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, knn_classifier.predict(X_test), average="macro") * 100, 2)
    
    # store the models performance text
    model_performance.append(['K-Nearest Neighbours', str(str(round(accuracy_score(Y_train, knn_classifier.predict(X_train)) * 100, 2)) + " %").upper(),\
        str(str(round(accuracy_score(Y_test, knn_classifier.predict(X_test)) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != knn_classifier.predict(X_test)).sum()))])
        
    # Save trained model
    #pickle.dump(knn_classifier, open('hdp_knn.pkl', 'wb'), protocol=0)
            
    # Naive Bayes model
    naive_bayes_model.fit(X_train, Y_train)
    # get the model performance score
    f1 = round(f1_score(Y_test, naive_bayes_model.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, naive_bayes_model.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, naive_bayes_model.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Naive Bayes', str(str(round(naive_bayes_model.score(X_train, Y_train) * 100, 2)) + " %").upper(),\
        str(str(round(naive_bayes_model.score(X_test, Y_test) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != naive_bayes_model.predict(X_test)).sum()))])
        
    # Save trained model
    #pickle.dump(naive_bayes_model, open('hdp_nb.pkl', 'wb'), protocol=0)

    # MLPClassifier model
    mlp_model.fit(X_train, Y_train)
    # get the model performance score
    accuracy = round(mlp_model.score(X_test, Y_test) * 100, 2)
    f1 = round(f1_score(Y_test, mlp_model.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, mlp_model.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, mlp_model.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Multi-Layer Perception', str(str(round(mlp_model.score(X_train, Y_train) * 100, 2)) + " %").upper(),\
        str(str(round(mlp_model.score(X_test, Y_test) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != mlp_model.predict(X_test)).sum()))])
        
    # Save trained model
    #pickle.dump(mlp_model, open('hdp_mlp.sav', 'wb'), protocol=0)
    
    # Train random Forest
    ran_classifier.fit(X_train,Y_train)
    prediction = ran_classifier.predict(X_test)
    accuracy = round(accuracy_score(prediction,Y_test) * 100, 2)
    # get the model performance score
    #accuracy = round(ran_classifier.score(X_test, Y_test) * 100, 2)
    f1 = round(f1_score(Y_test, ran_classifier.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, ran_classifier.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, ran_classifier.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Random Forest', str(str(round(ran_classifier.score(X_train, Y_train) * 100, 2)) + " %").upper(),\
        str(str(round(ran_classifier.score(X_test, Y_test) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != ran_classifier.predict(X_test)).sum()))])
        
    # Save trained model
    #pickle.dump(ran_classifier, open('hdp_random.sav', 'wb'), protocol=0)
    
    # Hybrid model
    estimators.append(('random forest', ran_classifier))
    estimators.append(('nbs', naive_bayes_model))
    # Defining the ensemble model
    ensemble = VotingClassifier(estimators)
    ensemble.fit(X_train, Y_train)
    prediction = ensemble.predict(X_test)
    accuracy = round(accuracy_score(prediction,Y_test) * 100, 2)
    f1 = round(f1_score(Y_test, ensemble.predict(X_test), average="macro") * 100, 2)
    precision = round(precision_score(Y_test, ensemble.predict(X_test), average="macro") * 100, 2)
    recall = round(recall_score(Y_test, ensemble.predict(X_test), average="macro") * 100 , 2)
    
    # store the models performance text
    model_performance.append(['Proposed Hybrid Model', str(str(round(ensemble.score(X_train, Y_train) * 100, 2)) + " %").upper(),\
        str(str(round(ensemble.score(X_test, Y_test) * 100, 2)) + " %").upper(),  str(str(f1) + " %").upper(),\
        str(str(precision) + " %").upper(), str(str(recall) + " %").upper(), str("Total points: %d, Mislabelled : %d" % (X_test.shape[0], (Y_test != ensemble.predict(X_test)).sum()))])
        
    # Save trained model
    #pickle.dump(ensemble, open('hybrid_model.pkl', 'wb'), protocol=0)
    dump(ensemble, 'hybrid_model.joblib') 
    
    # convert list to dataframe                    
    df = pd.DataFrame(model_performance, columns = ['Trained Model', 'Training Score', 'Testing Score', 'F1 Score', 'Precision Score', 'Recall Score', 'Mislabelled Points'])
    
    # return dataframe table
    return render_template('evaluation.html',tables= [df.to_html(classes='data')],titles=['na','Trained Machine Learning Models Without Using Bagging and Boosting Techniques'])
            
            
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        hospitalno = str(request.form['field1'])
        mobileno = str(request.form['field2'])
        Age = int(request.form['field3'])
        Sex = int(request.form['field4'])
        ChestPainType = int(request.form['field5'])
        RestingBP = int(request.form['field6'])
        Cholesterol = int(request.form['field7'])
        FastingBS = int(request.form['field8'])
        RestingECG = int(request.form['field9'])
        ExerciseAngina = int(request.form['field10'])
        Oldpeak = float(request.form['field11'])
        ST_Slope = int(request.form['field12'])
        HeartDisease = int(request.form['field13'])
        SmokingStatus = int(request.form['field14'])
        DiabetesStatus = float(request.form['field15'])
        Height = int(request.form['field16'])
        Weight = int(request.form['field17'])
        Ethnicity = int(request.form['field18'])
        MaxHR = int(request.form['field19'])
        FoodAllergies = str(request.form['field20'])
        
        # new model input data for prediction
        values = np.array([[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])
        
        # predict heart disease using loaded model
        hdp_model = Path('hybrid_model.joblib')
        if hdp_model.exists():
            new_response = ""
            heart_model = load(hdp_model)
            prediction = heart_model.predict(values)
            prediction = int(prediction[0])
            
            # string new url
            new_url = str("https://www.akifagoelectronics.com/UoW_2065655_Project/saveData2.php?hospitalno=" + str(hospitalno) + "&mobileno=" + mobileno +\
                "&age=" + str(Age) + "&sex=" + str(Sex) + "&chest_pain=" + str(ChestPainType) + "&blood_pressure=" + str(RestingBP) + "&cholesterol=" + str(Cholesterol) +\
                "&blood_sugar=" + str(FastingBS) + "&ecg=" + str(RestingECG) + "&angina=" + str(ExerciseAngina) + "&old_peak=" + str(Oldpeak) + "&peak_exercise=" + str(ST_Slope) +\
                "&heart_status=" + str(prediction) + "&smoking_status=" + str(SmokingStatus) + "&diabetes_status=" + str(DiabetesStatus) + "&height=" + str(Height) + "&weight=" + str(Weight) +\
                "&race=" + str(Ethnicity) + "&heart_rate=" + str(MaxHR) + "&food_allergies=" + str(FoodAllergies)).strip().replace(' ', '')
                
            # use the try block to capture any arising error exception
            try:
                # try to upload new data to database using first attempt
                response = requests.get(new_url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"}) 
  
                # Check for http OK response code
                if response.status_code == 200:
                    # Do this when new data update is successful
                    if prediction == 0:
                        new_response = "CONGRATULATION! YOU DONT HAVE HEART DISEASE. STAY HEALTHY. <br>"
                
                    if prediction == 1:
                        new_response = "OOPS! YOU HAVE CHANCES OF HEART DISEASE. PLEASE CONTACT THE NEAREST HOSPITAL. <br>"
                    
                    if HeartDisease == 0:
                        new_response += "PREVIOUSLY YOU DONT HAVE HEART DISEASE."
                
                    if HeartDisease == 1:
                        new_response += "PREVIOUSLY YOU HAD HEART DISEASE."
                    
                    return str(new_response) 
                        
                else:
                    # try to upload new data to database using second attempt for mac
                    response = requests.get(new_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"}) 
                    
                    # Check for http OK response code
                    if response.status_code == 200:
                        # Do this when new data update is successful
                        if prediction == 0:
                            new_response = "CONGRATULATION! YOU DONT HAVE HEART DISEASE. STAY HEALTHY. <br>"
                                
                        if prediction == 1:
                            new_response = "OOPS! YOU HAVE CHANCES OF HEART DISEASE. PLEASE CONTACT THE NEAREST HOSPITAL. <br>"
                    
                        if HeartDisease == 0:
                            new_response += "PREVIOUSLY YOU DONT HAVE HEART DISEASE."
                
                        if HeartDisease == 1:
                            new_response += "PREVIOUSLY YOU HAD HEART DISEASE."
                    
                        return str(new_response)
                    else:
                        # Do this when new data update is not successful
                        if prediction == 0:
                            new_response = "CONGRATULATION! YOU DONT HAVE HEART DISEASE. STAY HEALTHY. <br>"
                                
                        if prediction == 1:
                            new_response = "OOPS! YOU HAVE CHANCES OF HEART DISEASE. PLEASE CONTACT THE NEAREST HOSPITAL. <br>"
                    
                        if HeartDisease == 0:
                            new_response += "PREVIOUSLY YOU DONT HAVE HEART DISEASE."
                
                        if HeartDisease == 1:
                            new_response += "PREVIOUSLY YOU HAD HEART DISEASE."
                            
                        return 'ATTENTION PLEASE! THE NEW PREDICTION FAILED TO UPLOAD, PLEASE TRY AGAIN. ERROR MESSAGE: 00000' + str(response.status_code).upper() + "<br><br>" + str(new_response) + "<br>" + new_url
                    
            except requests.exceptions.RequestException as e:
                return "ERROR: " + str(e).upper()
                
        else:
            return redirect('https://www.akifagoelectronics.com/2065655_Hospital/retrain')
           
        #return str(values)
        
        

if __name__ == '__main__':
	app.run(debug=True)
