def mypred(features,weights):
    p=np.dot(features,weights)
    e=1+(np.exp(-1*p))
    sig=1/e
    return sig

def error(target,features,weights):
    er1=mypred(features,weights)
    err=er1-target
    return(err)


def gradient(target,features,weights):
    e1=error(target,features,weights)
    gd=np.dot(features.T,e1)#Normalization
    #print('Gradient: ',gd)
    return(gd)

#gradient descent function
def my_learn_gd(target,features,learning_rate,iterations):
    #initializing theta
    weights=np.random.random(features.shape[1])
    for i in range(iterations):
        weights-=learning_rate*gradient(target,features,weights)
    return weights


def decision_boundary(prob):
    
    for k , i in enumerate(prob):
        if i>=0.5:
            prob[k]=1
        else:
            prob[k]=0
    return prob

def accuracy2(prob, target):
    from sklearn import metrics
    predict=decision_boundary(prob)
    return metrics.accuracy_score(predict,target)

def m(features):
    
    mn=(features-features.mean())/features.std()
    return mn

def cross_entropy_error(target,features,weights):
    predict=mypred(features,weights)
    ce=0
    threshold=0.0001
    for k,i in enumerate(target):
        if i==1:
            ce=(np.log(predict[k]+threshold))
        else:
            ce=np.log(1-predict[k]+threshold)
    return ce


df_train_1d=m(df_train)
df_test_1d=m(df_test)

theta11=my_learn_gd(df_train_y,df_train_1d,0.1,10)
print("Cross Entropy Loss After 10 iterations for learning rate 0.1:",cross_entropy_error(df_test_y,df_train,theta11))

theta12=my_learn_gd(df_train_y,df_train_1d,0.1,50)
print("Cross Entropy Loss After 50 iterations for learning rate 0.1:",cross_entropy_error(df_test_y,df_train,theta12))


theta13=my_learn_gd(df_train_y,df_train_1d,0.1,100)
print("Cross Entropy Loss After 100 iterations for learning rate 0.1:",cross_entropy_error(df_test_y,df_train,theta13))





theta21=my_learn_gd(df_train_y,df_train_1d,0.01,10)
print("Cross Entropy Loss After 10 iterations for learning rate 0.01:",cross_entropy_error(df_test_y,df_train,theta21))

theta22=my_learn_gd(df_train_y,df_train_1d,0.01,50)
print("Cross Entropy Loss After 50 iterations for learning rate 0.01:",cross_entropy_error(df_test_y,df_train,theta22))

theta23=my_learn_gd(df_train_y,df_train_1d,0.01,100)
print("Cross Entropy Loss After 100 iterations for learning rate 0.01:",cross_entropy_error(df_test_y,df_train,theta23))





theta31=my_learn_gd(df_train_y,df_train_1d,0.05,10)
print("Cross Entropy Loss After 10 iterations for learning rate 0.05:",cross_entropy_error(df_test_y,df_train,theta31))

theta32=my_learn_gd(df_train_y,df_train_1d,0.05,50)
print("Cross Entropy Loss After 50 iterations for learning rate 0.05:",cross_entropy_error(df_test_y,df_train,theta32))


theta33=my_learn_gd(df_train_y,df_train_1d,0.05,100)
print("Cross Entropy Loss After 100 iterations for learning rate 0.05:",cross_entropy_error(df_test_y,df_train,theta33))




thetat=my_learn_gd(df_train_y,df_train_1d,0.1,100)
x=mypred(df_test_1d,thetat)
print("Accuracy after 100 iterations for learning rate 0.1:",accuracy2(x,df_test_y))

x=mypred(df_test_1d,theta13)
print("Accuracy after 100 iterations for learning rate 0.1:",accuracy2(x,df_test_y))

x=mypred(df_test_1d,theta23)
print("Accuracy after 100 iterations for learning rate 0.01:",accuracy2(x,df_test_y))

x=mypred(df_test_1d,theta33)
print("Accuracy after 100 iterations for learning rate 0.05:",accuracy2(x,df_test_y))