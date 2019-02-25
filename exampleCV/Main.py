import scipy.io as sio
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use("ggplot")
from sklearn import svm
from cvpartition import *
from sklearn.model_selection import *
from sklearn.metrics import accuracy_score
from pdb import set_trace as bp
########################
random.seed(0) ###equivalent of rng(1)
#generate pseduo random numbers
################
#reading the data.mat
Data = sio.loadmat('data.mat')
###########START OF USER DEFINED PARAMETER
fold_type="KFold" ###Specify what is the folding method here, "KFold" or "LOO"
kfold=10			###if KFold it is chosen then kfold can be any number greater than 1 if fold_type is "LOO" then kfold=1

###########END OF USER DEFINED PARAMETER

####################let's sieve the data.mat file into features and labels 
data=Data['data']
data=data[0,0]
features=data['features']
label=data['labels']
# label=np.ravel(label)
size_label=label.shape[0]

#######################################################
##crossvalidation, this function does exactly the same as matlab function return indices for the values that must be kept for training or testing
c=cvpartition(features,fold_type, kfold)
######
training=c[0]
test=c[1]
######
test_labels_vector=[]#store ground truth in test order
predicted_labels_vector=[] #vector to collect test results (labels assigned by SVM)
if(kfold==1):
	kfold=size_label

#########################################################
for i in range(kfold):#(LOO) run SVM 84 times
	trainIndex=training["Row"+str(i)].T.astype(int)
	testIndex=test["Row"+str(i)].T.astype(int)
	##############################################
	train_labels=np.ravel(extract_data(label,trainIndex))
	train_data=extract_data(features,trainIndex)
	test_labels=np.ravel(extract_data(label,testIndex))
	test_data=extract_data(features,testIndex)
	# print(train_data.shape)
	# print(test_labels.shape)

	##############################################
	clf = svm.SVC(kernel='linear', C = 1) #define the classifier type is linear and C is tuning parameter

	clf.fit(train_data,train_labels) #classifier
	prediction=clf.predict(test_data)#test on our data set 

	print("Acuracy on " +str(i), "run is :" +str(100*round(accuracy_score(test_labels, prediction),2)),"%") #return the accuracy

	predicted_labels_vector=np.concatenate((predicted_labels_vector, prediction))#concatenate predicted labels
	test_labels_vector=np.concatenate((test_labels_vector,test_labels))
	
###confusion Matrix
CM=confusionmat(test_labels_vector,predicted_labels_vector)
Accuracy = (CM[1,1] + CM[0,0])/(size_label) * 100
Sensitivity=(CM[1,1])/(CM[1,1]+CM[1,0]) * 100
Specificity=CM[0,0]/(CM[0,0]+CM[0,1]) * 100
print("Accuracy: " +str(Accuracy))
print("Sensitivity: " +str(Sensitivity))
print("Specificity: " +str(Specificity))
