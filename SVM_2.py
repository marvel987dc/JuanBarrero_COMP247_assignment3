import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

data = pd.read_csv("./Data/breast_cancer.csv")
data_juanDavid_df2 = pd.DataFrame(data)

#replace the '?' with NaN
data_juanDavid_df2['bare'] = data_juanDavid_df2['bare'].replace('?', np.nan)

#drop the ID column
data_juanDavid = data_juanDavid_df2.drop('ID', axis=1)

#separae the features from the class
X = data_juanDavid.drop('class', axis=1)
y = data_juanDavid['class']

#splitting the data 80% training and 20% testing
seed = 53
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

#preprocessing library to define two transformers
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#Fill the missing values with the median (hint: checkout SimpleImputer)
checkout_simple_imputer = SimpleImputer(strategy='median')
#Scale the features (hint: checkout StandardScaler)
checkout_standard_scaler = StandardScaler()
#Create a pipeline for the numerical features
num_pipe_Juan = Pipeline(steps=[('imputer', checkout_simple_imputer), ('scaler', checkout_standard_scaler)])
#Create a column transformer for the numerical features
svm_classifier = SVC(random_state=53)
#Create a pipeline for the SVM classifier
pipe_svm_Juan = Pipeline([
    ('num_pipe_firstname', num_pipe_Juan),
    ('svm', svm_classifier)
])
#print the pipeline object
print(num_pipe_Juan)

#define the grid search parameters
param_grid = {
    'svm__kernel': ['linear', 'rbf','poly'],
    'svm__C': [0.01,0.1, 1, 10, 100],
    'svm__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    'svm__degree':[2,3]
}
print(param_grid)

#create a grid search object
grid_search_Juan = GridSearchCV(estimator=pipe_svm_Juan, param_grid=param_grid, scoring='accuracy', refit=True, verbose=3)

#fit the grid search object
grid_search_Juan.fit(X_train, y_train)

#print the best parameters
print(grid_search_Juan.best_params_)
#print the best estimator
print(grid_search_Juan.best_estimator_)
#save the variable of the best model into best_model_Juan
best_model_Juan = grid_search_Juan.best_estimator_

#accuracy score of the model
from sklearn.metrics import accuracy_score
y_pred = best_model_Juan.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

import joblib as jb
#save the best model
jb.dump(best_model_Juan, 'best_model_Juan.joblib')
#save the pipeline
jb.dump(pipe_svm_Juan, 'pipe_svm_Juan.joblib')

#Joblib module in Python is especially used to execute tasks
# parallelly using Pipelines rather than executing them sequentially one after another.