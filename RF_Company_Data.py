import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
CompanyData =pd.read_csv("D:\\ExcelR Data\\Assignments\\Rendom Forests\\Company_Data.csv")
CompanyData.head()
CompanyData.columns
Le = preprocessing.LabelEncoder()
#convertong catogorical to numerical
CompanyData['shelveLoc']=Le.fit_transform(CompanyData['ShelveLoc'])
CompanyData['urban']=Le.fit_transform(CompanyData['Urban'])
CompanyData['us']=Le.fit_transform(CompanyData['US'])
#Droping
CompanyData.drop(["ShelveLoc"],inplace=True,axis=1)
CompanyData.drop(["Urban"],inplace=True,axis=1)
CompanyData.drop(["US"],inplace=True,axis=1)
# converting float value to integers we are converting float to catogorical data
bins=[-1,6,12,18]
CompanyData["Sales"]=pd.cut(CompanyData["Sales"],bins,labels=["lower","medium","high"])

colnames = list(CompanyData.columns)
predictors = colnames[1:11] #input
target = colnames[0] #output

X = CompanyData[predictors]
Y = CompanyData[target]

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=1000,criterion="entropy")

# n_estimators -> Number of trees ( you can increase for better accuracy)
# n_jobs -> Parallelization of the computing and signifies the number of jobs running parallel for both fit and predict
# oob_score = True means model has done out of box sampling to make predictions

np.shape(CompanyData) 
# Shape = 400, 11 


# Attributes that comes along with RandomForest function
rf.fit(X,Y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels = 3
rf.n_features_  # Number of input features in model = 10

rf.n_outputs_ # Number of outputs when fit performed = 1

rf.predict(X)


CompanyData['rf_pred'] = rf.predict(X)

from sklearn.metrics import confusion_matrix
confusion_matrix(CompanyData['Sales'],CompanyData['rf_pred']) # Confusion matrix

pd.crosstab(CompanyData['Sales'],CompanyData['rf_pred'])



print("Accuracy",(27+130+243)/(27+130+243)*100)
# Accuracy is 100.0
CompanyData["rf_pred"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.2, random_state= 4)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier


dtree = DecisionTreeClassifier() 
dtree.fit(x_train,y_train)
dtree.score(x_test,y_test)
# 0.625

dtree.score(x_train,y_train)
# 100%
# model is overfitting 
################ Now With BaggingClassifier ############
bg = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5,max_features=1.0, n_estimators=20)
bg.fit(x_train,y_train) #fitting the model 
bg.score(x_test,y_test)
# 66.25
bg.score(x_train,y_train)
# 94.375
# model is Underfitting
################ Now With BoostingClassifier ###########
ada = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=10,learning_rate=1)
ada.fit(x_train,y_train)
ada.score(x_test,y_test)

ada.score(x_train,y_train)
# 100%
# model is overfitting 

# with by looking at all the above model we can tell that "Bagging model" is gives the good result