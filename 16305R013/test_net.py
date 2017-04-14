
# coding: utf-8

# In[21]:

import numpy as np
import pandas as pd 
import pickle 


class neural_network:
    def __init__(self,hidden_layer_size=3,learning_rate=0.01,neurons=20,iterations=60000,activation_function='tan',decay_factor=0.95):
        self.hidden_layer_size=hidden_layer_size
        self.activation_function=activation_function
        self.learning_rate=learning_rate
        self.layer=list()
        self.layer_weights=list()
        self.output_layer=1
        self.iterations=iterations
        self.neurons=neurons
        self.decay_factor=decay_factor
        
        
    def create_network(self,X):
    
#         np.random.seed(1) #to have random in between the specific range
        random_weights=2*np.random.random((X.shape[1],self.neurons))-1
        self.layer_weights.append(random_weights)
        for i in range(self.hidden_layer_size-2):
            random_weights=2*np.random.random((self.neurons,self.neurons))-1
            self.layer_weights.append(random_weights)
        random_weights=2*np.random.random((self.neurons,self.output_layer))-1
        self.layer_weights.append(random_weights)
        
        
    def activation(self,x,derivative=False):
        if derivative:
            if self.activation_function == "sigmoid":
                return x * (1 - x)
            if self.activation_function=="tan":
                return 1.0 - np.tanh(x)**2
            if self.activation_function == "ReLU":
                return (x > 0).astype(int)        
        else:
            if self.activation_function == "sigmoid":
                return 1 / (1 + np.exp(-x))
            if self.activation_function=="tan":
                    return np.tanh(x)
            if self.activation_function == "ReLU":
                return x * (x > 0)
            
        
    def fit(self,X,Y):
        end_error=0
        self.create_network(X)
        for _ in range(self.iterations):
            #feed forward throught the network
            self.layer=list()
            self.layer.append(X)
            for i in range(self.hidden_layer_size):
                hidden_layer=self.activation(np.dot(self.layer[i],self.layer_weights[i]))
                self.layer.append(hidden_layer)
            
            error=Y-self.layer[-1]
            end_error=np.mean(np.abs(error))
#             if(_%100==1):
#                 print(str(_)+" Error "+str(end_error))
            for i in range(self.hidden_layer_size,0,-1):
                delta = error*self.activation(self.layer[i],derivative=True)
                error = delta.dot(self.layer_weights[i-1].T)
                self.layer_weights[i-1] += self.learning_rate * (self.layer[i-1].T.dot(delta))
            
            self.learning_rate=self.learning_rate*self.decay_factor

    
    def predict(self,X):
        predicted=X
        for i in range(self.hidden_layer_size):
            predicted=self.activation(np.dot(predicted,self.layer_weights[i]))
        predict=predicted
        if (self.activation_function=='sigmoid'):
            predict[predict>0.5]=1
            predict[predict<=0.5]=0
        if(self.activation_function=='tan'):
            predict[predict>0]=1
            predict[predict<=0]=0
        return predict.ravel()
    
    def score(self,X_test,Y_true):
        predict=self.predict(X_test)
        return np.sum(predict.ravel()==Y_true.ravel())/Y_true.shape[0]
    
    #weights can be passed and then the prediction is done based on the weight
    def predict_using_weights(self,X,file_name):
        with open(file_name, "rb") as fp: 
            weights = pickle.load(fp)
        predicted=X
        self.layer_weights=weights
        for i in range(self.hidden_layer_size):
            predicted=self.activation(np.dot(predicted,self.layer_weights[i]))
        predict=predicted
        if (self.activation_function=='sigmoid'):
            predict[predict>0.5]=1
            predict[predict<=0.5]=0
        if(self.activation_function=='tan'):
            predict[predict>0]=1
            predict[predict<=0]=0
        return predict.ravel()
    
    #Stores the weigths
    def store_weights(self,file_name):
        with open(file_name, "wb") as fp:
            pickle.dump(self.layer_weights, fp)
  


# In[18]:

def normalize(inputData):
    return (inputData - inputData.min()) / (inputData.max() - inputData.min())

#reads the datafiles and returns the training and the testing data
def get_training_data(percent):
    # get test & test csv files as a DataFrame
    train_df = pd.read_csv("train.csv")
    
    #feature engineering and removing features after analysis
    cols_to_drop=['race','native-country','fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    for col in cols_to_drop:
        train_df=train_df.drop([col],axis=1)

    numericalColumns = ('age', 'education-num')
    for i in numericalColumns:
        train_df[i] = normalize(train_df[i])

    #creating dummies of the data
    train_df=pd.get_dummies(train_df)
    
    train_df=train_df.drop(['id'],1)
    train_df['const']=1
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
    return train_X,train_Y,cross_X,cross_Y

def get_testing_data():
    test_df    = pd.read_csv("kaggle_test_data.csv")
    cols_to_drop=['race','native-country','fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    for col in cols_to_drop:
        test_df=test_df.drop([col],axis=1)

    numericalColumns = ('age', 'education-num')
    for i in numericalColumns:
        test_df[i] = normalize(test_df[i])

    #creating dummies of the data
    test_df=pd.get_dummies(test_df)
    test_ids=test_df['id'].as_matrix()
    test_df=test_df.drop(['id'],1)
    test_df['const']=1
    test_X=test_df.as_matrix()
    return test_X,test_ids


#writes the predicted values to file 
def write_result(ids,predicted,file_name):
    output=np.column_stack((ids,predicted))
    np.savetxt(file_name,output,delimiter=",",fmt="%d,%d",header="id,salary",comments ='')
    

# In[24]:

test_X,test_ids=get_testing_data()
nn_test=neural_network(hidden_layer_size=3,neurons=20,iterations=1,learning_rate=0.01,activation_function='tan')
predicted=nn_test.predict_using_weights(test_X,'weights.txt')
write_result(test_ids,predicted,"predictions.csv ")



