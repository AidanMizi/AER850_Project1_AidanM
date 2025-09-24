
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


# read data
data = pd.read_csv("Project 1 Data.csv")


# visualize the data
data.hist()
# this simple histogram of the data shows the changes in x, y, and z
# over the steps, which can help us visualize the shape of line of best
# fit the ML model might apply to each
# 
# this also shows that most of the amount of steps are made up of
# steps 8 and 9, something we have to account for when splitting the 
# data for training and testing



