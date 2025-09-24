
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


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
    stratDataTrain = data.iloc[train_index].reset_index(drop=True)
    stratDataTest = data.iloc[test_index].reset_index(drop=True)
#stratDataTrain = stratDataTrain.drop(columns=["Step"], axis=1)
#stratDataTest = stratDataTest.drop(columns=["Step"], axis=1)

# define variables
y_train = stratDataTrain['Step']
x_train = stratDataTrain.drop(columns=['Step'])
y_test = stratDataTest['Step']
x_test = stratDataTest.drop(columns=['Step'])



# # The use of stratified sampling is strongly recommended
# data["income_categories"] = pd.cut(data["median_income"],
#                           bins=[0, 2, 4, 6, np.inf],
#                           labels=[1, 2, 3, 4])
# my_splitter = StratifiedShuffleSplit(n_splits = 1,
#                                test_size = 0.2,
#                                random_state = 42)
# for train_index, test_index in my_splitter.split(data, data["income_categories"]):
#     strat_data_train = data.loc[train_index].reset_index(drop=True)
#     strat_data_test = data.loc[test_index].reset_index(drop=True)
# strat_data_train = strat_data_train.drop(columns=["income_categories"], axis = 1)
# strat_data_test = strat_data_test.drop(columns=["income_categories"], axis = 1)












