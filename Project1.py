
# AER850 PROJECT 1 FALL 2025
# AIDAN MIZIOLEK


# imports
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.ensemble import StackingClassifier
from joblib import dump, load
import joblib


# READ DATA
data = pd.read_csv("Project 1 Data.csv")

# identify step index bounds
stepBounds = [0]
for i in range (len(data['Step'])):
    if data.iloc[i,3] != data.iloc[i-1, 3]:
        stepBounds.append(i)
# this is done so each step can be visualized 


# VISUALIZE THE DATA
##data.hist(bins=13,edgecolor='black')
# this simple histogram of the data shows how many times each x, y, 
# and z coordinate occur. This helps us see that z has a good distribution
# of values, while x and y have a little more random distribution.
# This also helps us see that most of the movement is done in the z direction
# 
# this also shows that most of the amount of steps are made up of
# steps 7, 8, and 9, something we have to account for when splitting the 
# data for training and testing

# plot x, y, and z for each step 7, 8, 9
# step 7 plot
fig, plot789 = plt.subplots(nrows=2,ncols=2)
plot789[0,0].plot(data.iloc[stepBounds[7]:stepBounds[8],0], np.linspace(0, stepBounds[8]-stepBounds[7], num=stepBounds[8]-stepBounds[7]))
plot789[0,0].plot(data.iloc[stepBounds[7]:stepBounds[8],1], np.linspace(0, stepBounds[8]-stepBounds[7], num=stepBounds[8]-stepBounds[7]))
plot789[0,0].plot(data.iloc[stepBounds[7]:stepBounds[8],2], np.linspace(0, stepBounds[8]-stepBounds[7], num=stepBounds[8]-stepBounds[7]))
# step 8 plot
plot789[0,1].plot(data.iloc[stepBounds[8]:stepBounds[9],0], np.linspace(0, stepBounds[9]-stepBounds[8], num=stepBounds[9]-stepBounds[8]))
plot789[0,1].plot(data.iloc[stepBounds[8]:stepBounds[9],1], np.linspace(0, stepBounds[9]-stepBounds[8], num=stepBounds[9]-stepBounds[8]))
plot789[0,1].plot(data.iloc[stepBounds[8]:stepBounds[9],2], np.linspace(0, stepBounds[9]-stepBounds[8], num=stepBounds[9]-stepBounds[8]))
# step 9 plot
plot789[1,0].plot(data.iloc[stepBounds[9]:stepBounds[10],0], np.linspace(0, stepBounds[10]-stepBounds[9], num=stepBounds[10]-stepBounds[9]))
plot789[1,0].plot(data.iloc[stepBounds[9]:stepBounds[10],1], np.linspace(0, stepBounds[10]-stepBounds[9], num=stepBounds[10]-stepBounds[9]))
plot789[1,0].plot(data.iloc[stepBounds[9]:stepBounds[10],2], np.linspace(0, stepBounds[10]-stepBounds[9], num=stepBounds[10]-stepBounds[9]))
# these histograms show the changes in x, y, and z over the steps 7 to 9, 
# which can help us visualize the shape of line of best
# fit the ML model might apply to each


# CORRELATION ANALYSIS
# plot heatmap
plt.figure()
corr_matrix = data.corr()
sns.heatmap(np.abs(corr_matrix))
# find correlation values for x, y, and z with step
# correlation with step and x
corrX = np.abs(data.iloc[:,3].corr(data.iloc[:,0]))
print("Step correlation value with X: ", corrX)
# correlation with step and y
corrY = np.abs(data.iloc[:,3].corr(data.iloc[:,1]))
print("Step correlation value with Y: ", corrY)
# correlation with step and z
corrZ = np.abs(data.iloc[:,3].corr(data.iloc[:,2]))
print("Step correlation value with Z: ", corrZ)
# Based on the correlation values, I can see that X has the highest correlation
# to step size, while Z has the lowest. This tells me I should use the X data
# to train my models against the step number


# CLASSIFICATION MODEL DEVELOPMENT
# will use logistic regression, support vector machines, and random forest

# split data into training and testing data evenly across each step
my_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in my_splitter.split(data, data['Step']):
    trainData = data.iloc[train_index].reset_index(drop=True)
    testData = data.iloc[test_index].reset_index(drop=True)
#stratDataTrain = stratDataTrain.drop(columns=["Step"], axis=1)
#stratDataTest = stratDataTest.drop(columns=["Step"], axis=1)

# define variables
# since we want to predict the step value based on coordinates,
# 
y_train = trainData['Step']
x_train = trainData.drop(columns=['Step'])
y_test = testData['Step']
x_test = testData.drop(columns=['Step'])

# scale the variables so their weights are influenced by their values
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

# Use GridSearchCV to find best parameters for each model and 
# train models, then produce predictions from training data
scoringArg = ['f1', 'precision', 'accuracy']
# model 1 - logistic regression
model1 = LogisticRegression()
param_grid1 = {
    'penalty': ['l2', 'elasticnet'],
    'C': [1.0, 0.1, 0.001] }
gridSearch1 = GridSearchCV(estimator=model1, param_grid=param_grid1, scoring=scoringArg, refit='accuracy')
gridSearch1.fit(x_train, y_train)
print("Best parameters for Logistic Regression model: \n", gridSearch1.best_params_)


# model 2 - support vector machine
model2 = svm.SVC()
param_grid2 = {
    'C': [0.001, 0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto'] }
gridSearch2 = GridSearchCV(estimator=model2, param_grid=param_grid2, scoring=scoringArg, refit='accuracy')
gridSearch2.fit(x_train, y_train)
print("Best parameters for SVM model: \n", gridSearch2.best_params_)

# model 3 - random forest
model3 = RandomForestClassifier()
param_grid3 = {
    'n_estimators': [50, 100, 150] }
gridSearch3 = GridSearchCV(estimator=model3, param_grid=param_grid3, scoring=scoringArg, refit='accuracy')
gridSearch3.fit(x_train, y_train)
print("Best parameters for Random Forest model: \n", gridSearch3.best_params_)

# model 4 - logistic regression (RandomizedSearchCV)
model4 = LogisticRegression()
param_grid4 = {
        'C': np.linspace(1, 0.001, 3) }
gridSearch4 = RandomizedSearchCV(estimator=model4, param_distributions=param_grid4, scoring=scoringArg, refit='accuracy')
gridSearch4.fit(x_train, y_train)
print("Best parameters for Logistic Regression model (with RandomizedSearchCV): \n", gridSearch4.best_params_)


# MODEL PERFORMANCE ANALYSIS
# recreate models with best found parameters
model1 = LogisticRegression(**gridSearch1.best_params_)
model2 = svm.SVC(**gridSearch2.best_params_)
model3 = RandomForestClassifier(**gridSearch3.best_params_)
model4 = LogisticRegression(**gridSearch4.best_params_)
# fit all the models to testing data
model1.fit(x_test, y_test)
model2.fit(x_test, y_test)
model3.fit(x_test, y_test)
model4.fit(x_test, y_test)
# predict all models
y_pred1 = model1.predict(x_test)
y_pred2 = model2.predict(x_test)
y_pred3 = model3.predict(x_test)
y_pred4 = model4.predict(x_test)
# produce classification reports to compare performance
report1 = classification_report(y_test, y_pred1)
report2 = classification_report(y_test, y_pred2)
report3 = classification_report(y_test, y_pred3)
report4 = classification_report(y_test, y_pred4)
print("Logistic Regression performance: \n", report1)
print("Support Vector Machines performance: \n", report2)
print("Random Forest Classifier performance: \n", report3)
print("Logistic Regression with RandomizedSearchCV performance: \n", report4)

# based on the results, I choose randomf forest classifier since it got all
# the predictions correct

# produce confusion matrix for selected model (random forest)
conf = confusion_matrix(y_test, y_pred3)

# display the confusion matrix
display_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
dispConf = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=display_labels)
dispConf.plot()


# STACKED MODEL PERFORMANCE ANALYSIS
# model 1 and model 2 were chosen for this part since they were the two
# worst performing models and thus have the most room to perform better.
# model 3 performed perfectly so there can't be any improvement, and model 4
# is the same as model 1 but with RandomizedSearchCV and performed the same
estimatorsList = [
    ('lr', LogisticRegression())    
]
model5 = StackingClassifier(estimators=estimatorsList, final_estimator=svm.SVC())
model5.fit(x_train, y_train)
y_pred5 = model5.predict(x_test)
report5 = classification_report(y_test, y_pred5)
print("Logistic Regression stack with Support Vector Machines performance: \n", report5)
conf2 = confusion_matrix(y_test, y_pred5)
dispconf2 = ConfusionMatrixDisplay(confusion_matrix=conf2, display_labels=display_labels)
dispconf2.plot()



# MODEL EVALUATION
# save the model
modelName = 'Project1_ML_Model.joblib'
joblib.dump(model1, modelName)
print("\nModel 3 - Random Forest Classifier - saved as Project1_ML_Model.joblib")

# load the model and 
loadedModel = joblib.load(modelName)
print("\nModel loaded")
givenData = pd.DataFrame([[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]])
loadedPredictions = loadedModel.predict(givenData)
print("Predictions for the given data: \n", givenData, "\n", loadedPredictions)












