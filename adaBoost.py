import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

def change_weight(X,Y,weights):
    clf1 = DecisionTreeClassifier(random_state=0,max_depth=1)
    clf1.fit(X, Y, sample_weight= weights)
    pred= clf1.predict(X)
    classifiers.append(clf1)
    return np.where(pred==0, -1, pred)

def adaBoost(iterators,train_X,test_X,Y,Yt):
    beta= np.zeros(iterators)
    n_train, n_test = len(train_X), len(test_X)
    weights = np.ones(n_train) / n_train
    pred_train= np.zeros((iterators, n_train))    
    pred_test= np.zeros((iterators, n_test))
    pred_test= np.where(pred_test==0, -1, pred_test) 

    for i in range(iterators):
        predStumpTrain= change_weight(train_X,Y,weights)
        pred_train[i]= predStumpTrain
        pred_test[i]= classifiers[i].predict(test_X)
        pred_test[i]= np.where(pred_test[i]==0, -1, pred_test[i]) 
        ### Calculate Model Error
        weightPred = pd.DataFrame(list(zip(list(predStumpTrain),list(Y),weights)),columns= ['Predicted','Actual','sw'])
        miss= weightPred[weightPred['Predicted']!=weightPred['Actual']]
        error= sum(miss['sw'])
        beta[i] = 0.5 * np.log((1/error)-1)
        weights= weights * np.exp((-1*beta[i])*(train_y * predStumpTrain))
        # Normalize weights
        weights= weights/sum(weights)


    p = np.sign(np.matmul(beta,pred_train))
    pTest = np.sign(np.matmul(beta,pred_test))
    accuracy= accuracy_score(Y,p)
    error= 1-accuracy
    recall= recall_score(Y,p)
    precision= precision_score(Y,p)
    print("For %d Iterations" %iterators)
    print("--Train Data")
    print("Accuracy= {} \nError= {} \nPrecision={} \nRecall={}\n".format(accuracy,error, recall, precision))
    
    ### Predicting performance metrics on Test data
    
    accuracy= accuracy_score(Yt,pTest)
    error= 1-accuracy
    recall= recall_score(Yt,pTest)
    precision= precision_score(Yt,pTest)
    print("--For Test Data:")
    print("Accuracy= {} \nError= {} \nPrecision={} \nRecall={}\n\n".format(accuracy,error, precision, recall))
    print("************************************************************************************")



df=pd.read_csv(r"C:\Users\2ativ\Documents\spambase.csv",encoding="ANSI")
df.columns=df.columns.str.replace("word_freq_","")
df.columns=df.columns.str.replace("char_freq_","")
df.columns=df.columns.str.replace("capital_run_length_","")
df=df.rename(columns={"1":"output"})
df_train,df_test=train_test_split(df,test_size=0.25,random_state=1)
df_train_y=df_train["output"]
df_test_y=df_test["output"]
df_train.drop("output",axis=1,inplace=True)
df_test.drop("output",axis=1,inplace=True)



train_y=[x if x==1 else -1 for x in df_train_y]
test_y=[x if x==1 else -1 for x in df_test_y]
n_estimators=[1,50,100,150]
for i in n_estimators:
    classifiers=[]
    adaBoost(i,df_train,df_test,train_y,test_y)