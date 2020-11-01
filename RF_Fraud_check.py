import pandas as pd
import numpy as np
from sklearn import preprocessing

Fraudcheck = pd.read_csv("D:\\ExcelR Data\\Assignments\\Rendom Forests\\Fraud_check.csv")
Fraudcheck.head()
Fraudcheck.columns

Le = preprocessing.LabelEncoder()
Fraudcheck['undergrad']=Le.fit_transform(Fraudcheck['Undergrad'])
Fraudcheck['marital_Status']=Le.fit_transform(Fraudcheck['Marital_Status'])
Fraudcheck['urban']=Le.fit_transform(Fraudcheck['Urban'])
#Droping
Fraudcheck.drop(["Undergrad"],inplace=True,axis=1)
Fraudcheck.drop(["Marital_Status"],inplace=True,axis=1)
Fraudcheck.drop(["Urban"],inplace=True,axis=1)

# converting float value to integer S we are converting float to catogorical data
bins=[-1,30000,100000]
Fraudcheck["Taxable_Income"]=pd.cut(Fraudcheck["Taxable_Income"],bins,labels=["Risky","Good"])


colnames = list(Fraudcheck.columns)
predictors = colnames[1:6] #inputs
target = colnames[0] #outputs

X = Fraudcheck[predictors]
Y = Fraudcheck[target]
####### GridSearch 

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=1000,criterion="entropy")

# n_estimators -> Number of trees ( you can increase for better accuracy)
# n_jobs -> Parallelization of the computing and signifies the number of jobs 
# running parallel for both fit and predict
# oob_score = True means model has done out of box sampling to make predictions

np.shape(Fraudcheck)
# 600, 6 = Shape

#### Attributes that comes along with RandomForest function
rf.fit(predictors,target) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels = 2
rf.n_features_  # Number of input features in model = 5

rf.n_outputs_ # Number of outputs when fit performed = 1

rf.predict(predictors)


Fraudcheck['rf_pred'] = rf.predict(predictors)

from sklearn.metrics import confusion_matrix
confusion_matrix(Fraudcheck['Taxable_Income'],Fraudcheck['rf_pred']) # Confusion matrix

pd.crosstab(Fraudcheck['Taxable_Income'],Fraudcheck['rf_pred'])



print("Accuracy",(476+124)/(476+124)*100)
# Accuracy is 100.0
Fraudcheck["rf_pred"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.2, random_state= 4)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier


dtree = DecisionTreeClassifier() 
dtree.fit(x_train,y_train)
dtree.score(x_test,y_test)
# 67.5%

dtree.score(x_train,y_train)
# 100%
# model is overfitting 
################ Now With BaggingClassifier ############
bg = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5,max_features=1.0, n_estimators=20)
bg.fit(x_train,y_train) #fitting the model 
bg.score(x_test,y_test)
# 79.1%
bg.score(x_train,y_train)
# 87.7%
# model is Underfitting

################ Now With BoostingClassifier ###########
ada = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=10,learning_rate=1)
ada.fit(x_train,y_train)
ada.score(x_test,y_test)
# 66.6%
ada.score(x_train,y_train)
# 100%
# model is overfitting

# with by looking at all the above model we can tell that " Bagging Tecq " is gives the good result