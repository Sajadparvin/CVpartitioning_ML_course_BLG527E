import scipy.io as sio
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use("ggplot")
from sklearn import svm
from cvpartition import *
from sklearn.model_selection import *
from sklearn.metrics import accuracy_score
########################
np.random.seed(0) ###equivalent of rng(1)

Data = sio.loadmat('data.mat')
###########

###########
data=Data['data']
data=data[0,0]
features=data['features']
label=data['labels']
# label=np.ravel(label)
size_label=label.shape[0]

#######################################################
fold_number=15 #### USER DEFINED PARAMETER
c=cvpartition(features,"KFold", fold_number)
######
training=c[0]
test=c[1]
######
test_labels_vector=[]
predicted_labels_vector=[]


#######################################################
for i in range(fold_number):
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
	clf = svm.SVC(kernel='linear', C = 1)
	clf.fit(train_data,train_labels)
	prediction=clf.predict(test_data)

	print("Acuracy on " +str(i), "run is :" +str(100*round(accuracy_score(test_labels, prediction),2)),"%")

	predicted_labels_vector=np.concatenate((predicted_labels_vector, prediction))
	test_labels_vector=np.concatenate((test_labels_vector,test_labels))
###confusion Matrix
CM=confusionmat(test_labels_vector,predicted_labels_vector)
Accuracy = (CM[1,1] + CM[0,0])/(size_label) * 100
Sensitivity=(CM[1,1])/(CM[1,1]+CM[1,0]) * 100
Specificity=CM[0,0]/(CM[0,0]+CM[0,1]) * 100
print("Accuracy: " +str(Accuracy))
print("Sensitivity: " +str(Sensitivity))
print("Specificity: " +str(Specificity))
