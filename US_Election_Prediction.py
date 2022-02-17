import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import  f1_score

!kaggle competitions download -c cs-4780-final-project-county-prediction-basic


#Used for training and validation

def weighted_accuracy(pred, true):
    assert(len(pred) == len(true))
    num_labels = len(true)
    num_pos = sum(true)
    num_neg = num_labels - num_pos
    frac_pos = num_pos/num_labels
    weight_pos = 1/frac_pos
    weight_neg = 1/(1-frac_pos)
    num_pos_correct = 0
    num_neg_correct = 0
    for pred_i, true_i in zip(pred, true):
        num_pos_correct += (pred_i == true_i and true_i == 1)
        num_neg_correct += (pred_i == true_i and true_i == 0)
    weighted_accuracy = ((weight_pos * num_pos_correct) 
                         + (weight_neg * num_neg_correct))/((weight_pos * num_pos) + (weight_neg * num_neg))
    return weighted_accuracy



#Data cleaning and Feature Extraction from csv file


train = pd.read_csv("drive/MyDrive/CS_4780_Project/train_2016.csv", sep=',',header=0, encoding='unicode_escape')
test=pd.read_csv("drive/MyDrive/CS_4780_Project/test_2016_no_label.csv",sep=',',header=0, encoding='unicode_escape')
#Determining which party won each county, 0= GOP win, 1- DEM win
labels= train["DEM"]>train["GOP"]

#label dataset
labels=labels.astype(int)
#using FIPS code as index
train=train.set_index("FIPS")
test=test.set_index("FIPS")
#dropped county since it is not a numerical feature
train=train.drop('County', axis=1)
test=test.drop('County', axis=1)
#------------------------
train=train.drop('DEM', axis=1)
train=train.drop('GOP', axis=1)

#----------------

#Converting income numbers to floats

train["MedianIncome"]=train["MedianIncome"].str.replace(",","")

train["MedianIncome"]=pd.to_numeric(train ["MedianIncome"], downcast="float")

test["MedianIncome"]=test["MedianIncome"].str.replace(",","")

test["MedianIncome"]=pd.to_numeric(test["MedianIncome"], downcast="float")
print(train)



#Neural net preprocessing
#extra data cleaning has to be done for the neural network, data has to be converted to numpy arrays

nntrain=train.to_numpy()
nnlabels=labels.to_numpy()
nntest=test.to_numpy()



#Using DECISION TREE with information gain as the split criterion- this 
#ensures tree does not get too deep and thus prevents overfitting

D_tree_A=sklearn.tree.DecisionTreeClassifier(criterion="entropy", splitter= "random")



#Split training data into training and validation sets
#Split the training data into a training and validation set, with 3/4 used for training and 1/4 used for validation

x_train, x_val, y_train, y_val=sklearn.model_selection.train_test_split(train, labels, test_size=0.25)



#Train Decision trees with different depths

#average best depths of multiple groups of trees
bestdepths=[]

for x in range (1,10):
  acc_dictionary={}
#Test for best depth over a number of trees
  for i in range(1,30):
    D_tree=sklearn.tree.DecisionTreeClassifier(criterion="entropy", splitter= "random",max_depth=i)
    D_tree.fit(x_train,y_train)

    # Measure accuracy each has on validation set using weighted accuracy function
    val_preds=D_tree.predict(x_val)
    acc= weighted_accuracy(val_preds,y_val )
    acc=acc*100

    # Store dictionary of accuracies and their depths
    acc_dictionary[acc] = "{}".format(i)    
    best= acc_dictionary[max(acc_dictionary)]    
    bestdepths.append(int(best))


#Choose average of best depths

finaldp=sum(bestdepths)/len(bestdepths)


#Find best number of leaf nodes

acc_dictionary={}
for i in [20,50,100,500,1000]:
    D_tree=sklearn.tree.DecisionTreeClassifier(criterion="entropy", splitter= "random",max_depth=finaldp, max_leaf_nodes=i)
    D_tree.fit(x_train,y_train)
    
    
    # Measure accuracy each has on validation set using weighted accuracy function
    val_preds=D_tree.predict(x_val)
    acc= weighted_accuracy(val_preds,y_val )
    acc=acc*100
    
    

    # Store dictionary of accuracies and their depths
    acc_dictionary[acc] = i
best= acc_dictionary[max(acc_dictionary)]
final_max_nodes=best



#NEURAL NETWORK

#Split training data into training and validation sets
xnn_train, xnn_val, ynn_train, ynn_val=sklearn.model_selection.train_test_split(nntrain, labels, test_size=0.25)


#Tune activiation function
#Create a classifier with each activation function and test to see which function gives the best weighted accuracy
acc_dictionary={}
for func in ['identity', 'logistic', 'tanh', 'relu']:
  net=sklearn.neural_network.MLPClassifier(activation=func)
  net.fit(xnn_train,ynn_train)
  valnn_preds=net.predict(xnn_val)
  acc= weighted_accuracy(valnn_preds,ynn_val) 
  acc=acc*100

  # Store dictionary of accuracies and the function that gave them
  acc_dictionary[acc] = func        
  best= acc_dictionary[max(acc_dictionary)] 

print("best func is", best)
    
    

#Tune hidden layer 
acc_dictionary={}
for k in [10,25,50,75,100]:
  net=sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(k,))  
  net.fit(xnn_train, ynn_train)
  valnn_preds=net.predict(xnn_val)
  acc= weighted_accuracy(valnn_preds,ynn_val)
  print (acc)  

  # Store dictionary of accuracies and the function that gave them
  acc_dictionary[acc] = k
        
best= acc_dictionary[max(acc_dictionary)] 
print("best lay num is", best)



#Tune max iter

acc_dictionary={}

for m in [50,100,200,500,1000]:
    net=sklearn.neural_network.MLPClassifier(hidden_layer_sizes=50 ,activation='relu', max_iter=m)
    net.fit(xnn_train, ynn_train)
    valnn_preds=net.predict(xnn_val)
    acc= weighted_accuracy(valnn_preds,ynn_val)   

    # Store dictionary of accuracies and the max_iter value that gave them
    acc_dictionary[acc] = m          
best= acc_dictionary[max(acc_dictionary)] 
print("best max_iter is", m)

#Tune alpha

acc_dictionary={}

for alpha in[0.0010, 0.0015,0.0020,0.0100,0.0200] :
    net=sklearn.neural_network.MLPClassifier(hidden_layer_sizes=50 ,activation='relu', max_iter=1000, alpha=alpha)
    net.fit(xnn_train, ynn_train)
    valnn_preds=net.predict(xnn_val)
    acc= weighted_accuracy(valnn_preds,ynn_val)
    
    # Store dictionary of accuracies and the learning rate that gave them
    acc_dictionary[acc] = alpha
          
best= acc_dictionary[max(acc_dictionary)] 

print("best alpha is", best)

#Tuning learning rate init
acc_dictionary={}

for lrate in[0.0001, 0.0005,0.0010, 0.0015,0.0020] :
    net=sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100) ,activation='relu', max_iter=1000, learning_rate_init=lrate)
    net.fit(xnn_train, ynn_train)
    valnn_preds=net.predict(xnn_val)
    acc= weighted_accuracy(valnn_preds,ynn_val)
       
    # Store dictionary of accuracies and the learning rate that gave them
    acc_dictionary[acc] = lrate
          
best= acc_dictionary[max(acc_dictionary)] 

print("best learning rate is", best)





#TRAINING AND TESTING MODELS WITH BEST PARAMETER COMBINATIONS



#train and test best neural netowrk

net=sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100) ,activation='relu', max_iter=1000, alpha=0.01, learning_rate_init=0.002)
net.fit(xnn_train, ynn_train)
nnpreds=net.predict(nntest)
nnsub= pd.DataFrame({"FIPS": test.index ,
             "Result": nnpreds 
             })
nnsub=nnsub.set_index("FIPS")
nnsub.to_csv( "drive/MyDrive/CS_4780_Project/nnsoln.csv")




#train and test best tree

D_tree_final=sklearn.tree.DecisionTreeClassifier(criterion="entropy", splitter= "random",
                                                 max_depth=finaldp,max_leaf_nodes= final_max_nodes )
D_tree_final.fit(x_train,y_train)
preds=D_tree_final.predict(x_val)
val_accy=weighted_accuracy(preds,y_val)
submissionpreds=D_tree_final.predict(test)
submission= pd.DataFrame({"FIPS": test.index ,
             "Result": submissionpreds 
             })
submission=submission.set_index("FIPS")
submission.to_csv( "drive/MyDrive/CS_4780_Project/Basic_Soln1.2.csv")




#-----IMPROVED SOLUTION-----#

#Random Forest
#Use randomizezed cross validation to conduct k fold cross validation for a large number of different parameter combinations
#List of possible valuse to be considered for each parameter. Each iteration of randomsearchcv will use a combination of values
#from these lists and record the accuracy using weighted accuracy function

grid = {'n_estimators': [int(q) for q in np.linspace(start=100, stop=1500, num=100)],
               'max_features': ['auto'],
               'max_depth': [int(q) for q in (range(10,100))],             #10,100
               'bootstrap':[True, False]}

#scorer=make_scorer(sklearn.metrics.f1_score)
from sklearn.metrics import make_scorer
scorer = make_scorer(weighted_accuracy, greater_is_better=True)
searcher=sklearn.model_selection.RandomizedSearchCV(estimator = forest, param_distributions = grid, n_iter = 10, cv = 5, scoring=scorer )
searcher.fit(nntrain,nnlabels)

#give best parameters
print(searcher.best_params_)


forest= sklearn.ensemble.RandomForestClassifier(n_estimators=1000, max_features='auto', max_depth=18, bootstrap=True)
forest.fit(xnn_train,ynn_train)
frstpred=forest.predict(xnn_val)



#Create submission
creativepreds=forest.predict(test)
creativesub= pd.DataFrame({"FIPS": test.index ,
             "Result": creativepreds 
             })
creativesub=creativesub.set_index("FIPS")
creativesub.to_csv("drive/MyDrive/CS_4780_Project/creativesol.csv")













