#we will be using support vector machine to predict the diabetes
#importing libraries 

import numpy as np  #make np arrays
import pandas as pd #structed data operations
from sklearn.preprocessing import StandardScaler #used to standardize the data to a common range
from sklearn.model_selection import train_test_split    #splitting the data
from sklearn import svm #support vector machine
from sklearn.metrics import accuracy_score #accuracy score

#data collection and analysis
#loading the dataset to a pandas dataframe
diabetes_dataset = pd.read_csv(r'c:\\Users\\user\\Desktop\\ML beginner Projects\\diabetes.csv')

#displaying the first 5 rows of the dataset
print(diabetes_dataset.head())

#number of rows and columns
diabetes_dataset.shape #(768, 9)

#statistical measures of the data
diabetes_dataset.describe()

#number of diabetic and non-diabetic patients
diabetes_dataset['Outcome'].value_counts() #1:268, 0:500

# 0 ---> non diabetic 1---> diabetic
diabetes_dataset.groupby('Outcome').mean() #mean of all the columns grouped by outcome

#separating the data and labels
X= diabetes_dataset.drop(columns='Outcome',axis=1) #droping the outcome column
Y=diabetes_dataset['Outcome'] #storing the outcome column in Y

print(X)
print(Y)

#data standardization
scaler= StandardScaler() #standardize the data
scaler.fit(X) 
#transform this data
standarized_data= scaler.transform(X) #we can use scaler.fit_transform(X) to do both in one step

print(standarized_data) #all the values between 0 and 1 

X=standarized_data
Y=diabetes_dataset['Outcome']

print(X)
print(Y)

#splitting the data into training data and test data
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2) #stratify is used to maintain the ratio of 0 and 1 in the training and test data

print(X.shape, X_train.shape, X_test.shape) #768,614,154

#training the model
classifier= svm.SVC(kernel='linear') #using linear kernel

#training the support vector machine classifier
classifier.fit(X_train, Y_train)

#model evaluation
#accuracy score on the training data
X_train_prediction= classifier.predict(X_train)
trainin_data_accuracy= accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data :', trainin_data_accuracy) #0.7866449511400652

#accuracy score on the test data 
X_test_prediction = classifier.predict(X_test) 
test_data_accuracy= accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data :', test_data_accuracy) #0.7727272727272727


#making a predictive system
input_data=(6,148,72,35,0,33.6,0.627,50)

#changing the input data to a numpy array
input_data_array=np.asarray(input_data)

#reshape the array
input_data_reshaped=input_data_array.reshape(-1,1)

#standardize the input data
std_data=scaler.transform(input_data_reshaped)

print(std_data)

#prediction
prediction= classifier.predict(std_data)  #prediction is a list [0] or [1]

print(prediction) #1

if(prediction[0]==0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')





