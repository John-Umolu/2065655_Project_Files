from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# read new dataset csv file
dataset = pd.read_csv('heart.csv')

# get the column names
column_names = dataset.columns
# using label encoder
le = LabelEncoder()
# scan for columns with non-numerical labels
for column in column_names:
    # checks if the first label is string
    if isinstance(dataset[column][0], str):
        # transform to numerical labels
        dataset[column] = le.fit_transform(dataset[column])

# drop the target column
predictors = dataset.drop("HeartDisease", axis=1)

# get the target variable
target = dataset["HeartDisease"]

# split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

# using grid search CV
ran_classifier=RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [200, 500],'max_features': ['auto', 'sqrt', 'log2'],'max_depth' : [4,5,6,7,8],'criterion' :['gini', 'entropy']}
CV_rfc = GridSearchCV(estimator=ran_classifier, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, Y_train)
print(CV_rfc.best_params_)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

#Create KNN Object
knn = KNeighborsClassifier()
#List Hyperparameters to tune
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
#convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Making model
clf = GridSearchCV(knn, hyperparameters, cv=10)
best_model = clf.fit(X_train,Y_train)
#Best Hyperparameters Value
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
#Predict testing set
y_pred = best_model.predict(X_test)
#Check performance using accuracy
print(accuracy_score(Y_test, y_pred))
#Check performance using ROC
roc_auc_score(Y_test, y_pred)