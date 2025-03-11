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

#M--->Mine R--->Rock
print(sonar_data.groupby(60).mean())  #mean of all the columns grouped by the target column the 60th column

#separating the data and the labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

print(X)
print(Y)

#training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, stratify=Y, random_state=1) #stratify is used to split the data in a balanced way 
# 10% of the data is test data
#X_train-->training data
#Y_train-->training label
print(X_train)
print(Y_train)


print(X.shape, X_train.shape, X_test.shape) #(208, 60) (187, 60) (21, 60)

#model training--> logistic regression
model= LogisticRegression()

#training the logistic regression model with training data
model.fit(X_train, Y_train)

#model evaluation
#accuracy on training data
#if accuracy is somewhere above 70 it is good
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on training data : ', training_data_accuracy) #0.8342245989304813

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on test data : ', test_data_accuracy) #0.7619047619047619

#making a predictive system
input_data=(0.0317,0.0956,0.1321,0.1408,0.1674,0.1710,0.0731,0.1401,0.2083,0.3513,0.1786,0.0658,0.0513,0.3752,0.5419,0.5440,0.5150,0.4262,0.2024,0.4233,0.7723,0.9735,0.9390,0.5559,0.5268,0.6826,0.5713,0.5429,0.2177,0.2149,0.5811,0.6323,0.2965,0.1873,0.2969,0.5163,0.6153,0.4283,0.5479,0.6133,0.5017,0.2377,0.1957,0.1749,0.1304,0.0597,0.1124,0.1047,0.0507,0.0159,0.0195,0.0201,0.0248,0.0131,0.0070,0.0138,0.0092,0.0143,0.0036,0.0103)


#changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)  #1 row and 60 columns

prediction= model.predict(input_data_reshaped) 
print(prediction) #gives either R or M as output
if (prediction[0]=='R'):
    print('The object is a Rock')
else:
    print('The object is a Mine')






