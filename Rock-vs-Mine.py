#importing the dependencies
import numpy as np  #np for arrays
import pandas as pd #processing data / loading pur data
from sklearn.model_selection import train_test_split #splitting the data
from sklearn.linear_model import LogisticRegression #the regression model
from sklearn.metrics import accuracy_score #accuracy score


#Data collection and processing 
#loading the dataset to a pandas dataframe
sonar_data = pd.read_csv(r'c:\Users\user\Desktop\ML beginner Projects\Copy of sonar data.csv', header=None)
 #we need to mention to our data frame that there s no header

#displaying the first 5 rows of the dataset
print(sonar_data.head())

#number of rows and columns
print(sonar_data.shape) #(208, 61)

#gives the statistical measures of the data
print(sonar_data.describe()) 

#count of rocks and mines
print(sonar_data[60].value_counts()) #R: 97, M: 111
#almost equal number of rocks and mines means the data is balanced and our model will be able to learn properly and perform well






