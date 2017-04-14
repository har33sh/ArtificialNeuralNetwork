
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd 

# machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier


# In[4]:

def normalize(inputData):
    return (inputData - inputData.min()) / (inputData.max() - inputData.min())

#reads the datafiles and returns the training and the testing data
def get_data():
    # get test & test csv files as a DataFrame
    train_df = pd.read_csv("train.csv")
    test_df    = pd.read_csv("kaggle_test_data.csv")
    
    #feature engineering and removing features after analysis
    cols_to_drop=['race','native-country','fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    for col in cols_to_drop:
        train_df=train_df.drop([col],axis=1)
        test_df=test_df.drop([col],axis=1)

    numericalColumns = ('age', 'education-num')
    for i in numericalColumns:
        train_df[i] = normalize(train_df[i])
        test_df[i] = normalize(test_df[i])

    
    #creating dummies of the data
    train_df=pd.get_dummies(train_df)
    test_df=pd.get_dummies(test_df)

    #remove unwanted columns and the columns that are created for ?
#     columns_to_remove=set(list(train_df)).symmetric_difference(set(list(test_df)))
#     columns_to_remove.remove('salary')
#     for col in list(train_df):
#         if (col in columns_to_remove) or ("?" in col) :
#             train_df=train_df.drop(col,1)
#     for col in list(test_df):
#         if (col in columns_to_remove) or ("?" in col) :
#             test_df=test_df.drop(col,1)
    
    return train_df,test_df


def process_data(percent):
    train_df,test_df=get_data()
    test_ids=test_df['id'].as_matrix()
    train_df=train_df.drop(['id'],1)
    test_df=test_df.drop(['id'],1)
    train_df['const']=1
    test_df['const']=1
    Y=train_df['salary'].as_matrix()
    X=train_df.drop(['salary'], axis=1).as_matrix()
    Y=Y.reshape(len(Y),1)
    end=int(X.shape[0] * percent)
    #training data
    train_X=X[:end,:]
    train_Y=Y[:end,:]
    #data for cross validation
    cross_X=X[end:,:]
    cross_Y=Y[end:,:]
    #testing data
    test_X=test_df.as_matrix()
    return train_X,train_Y,cross_X,cross_Y,test_X,test_ids



#writes the predicted values to file 
def write_result(ids,predicted,file_name):
    output=np.column_stack((ids,predicted))
    np.savetxt(file_name,output,delimiter=",",fmt="%d,%d",header="id,salary",comments ='')
    


# In[7]:

train_X,train_Y,cross_X,cross_Y,test_X,test_ids= process_data(0.80)
X_train=train_X
Y_train=train_Y.ravel()
X_test=cross_X
Y_test=cross_Y.ravel()


print ("Training Logistic Regression.. ")
#----------- Logistic Regression------------------
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
print("Logistic Regression : "+ str(logreg.score(X_test, Y_test)))


print ("Training SVM.. ")
#----------- Support Vector Machines------------------
svc = SVC()
svc.fit(X_train, Y_train)
print("Support Vector Machines : "+str(svc.score(X_test, Y_test)))


print ("Training KNN Classification.. ")
#----------- K NN Classification------------------
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, Y_train)
print("K NN Classification : "+str(knn.score(X_test, Y_test)))



# In[ ]:




# In[11]:

print("Predicting.. ")
logreg_predict=logreg.predict(test_X)
svc_predict=svc.predict(test_X)
knn_predict=knn.predict(test_X)

print("Writing.. ")

write_result(test_ids,logreg_predict,"predictions_1.csv")
write_result(test_ids,svc_predict,"predictions_2.csv")
write_result(test_ids,knn_predict,"predictions_3.csv")

print("Done")


# In[ ]:



