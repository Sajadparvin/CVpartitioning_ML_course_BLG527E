import numpy as np
import random
import pdb
###############################################conusionMat
def confusionmat(test_vec,test_pred):
	CM=np.zeros((2,2))
	for i in range(len(test_vec)):
		if(test_vec[i]<0 and test_pred[i]<0): #true negative
			CM[0,0]+=1
		if(test_vec[i]>0 and test_pred[i]>0): #true postive
			CM[1,1]+=1
		if(test_vec[i]<0 and test_pred[i]>0): #false positive
			CM[0,1]+=1
		if(test_vec[i]>0 and test_pred[i]<0): #false negative
			CM[1,0]+=1
	return CM
###############################################cvpartition
def cvpartition(sample,fold_type, fold_number):
	np.random.seed(0)
	###I did not needed the fold_type ,I just put it there to resemble the MATLab code
	size_sample=sample.shape[0]
	parameter_random_vectors={}
	vector=np.random.rand(1,fold_number)
	# training=np.ones(1,size_sample)
	# test=np.ones(1,size_sample)
	k_fold_vect_size=size_sample/fold_number
	flag_equal_size_vect=0
	if(size_sample%fold_number==0):
		all_vect_elem=k_fold_vect_size
		flag_equal_size_vect=1
		firs=int(k_fold_vect_size)
	elif(size_sample%fold_number!=0):
		firs=int(np.floor(k_fold_vect_size))
		rest=int(np.ceil(k_fold_vect_size))
		flag_equal_size_vect=0
	
	for i in range(fold_number):
		if(flag_equal_size_vect==1):
			parameter_random_vectors["fold_rand" +str(i)]=random.sample(range(size_sample), firs)
		if(flag_equal_size_vect==0):
			if(i==0):
				parameter_random_vectors["fold_rand" +str(i)]=random.sample(range(size_sample), firs)
			if(i!=0):
				parameter_random_vectors["fold_rand" +str(i)]=random.sample(range(size_sample), rest)
	# for i in range(fold_number):
	# 	print(len(parameter_random_vectors["fold_rand" +str(i)]))
	test_index=parameter_random_vectors
	train_index={}
	zero_one_index_test={}
	zero_one_index_train={}
	for i in range(len(test_index)):
		x_test=np.zeros(size_sample).reshape(1,size_sample)
		x_train=np.ones(size_sample).reshape(1,size_sample)
		for j in test_index["fold_rand" +str(i)]:
			for k in range(size_sample):
				if(k==j):
					# print("ok")
					x_test[0,k]=1
					x_train[0,k]=0

		zero_one_index_train["Row"+str(i)]=x_train
		zero_one_index_test["Row"+str(i)]=x_test
	tr_size=np.arange(len(parameter_random_vectors))

	for i in range(fold_number):
		tr_size[i]=len(parameter_random_vectors["fold_rand" +str(i)])
	##################################################################assign return values
	training=zero_one_index_train
	test=zero_one_index_test

	##################################################################
	print("K-fold cross validation partition")
	print("NumObservations:" +str(size_sample))
	print("NumTestSets:" +str(fold_number))
	print("TrainSize:" +str(size_sample-tr_size))
	print("TestSize:" +str(tr_size))
	return training,test

# a=np.arange(84).reshape(84,1)

# cvpartition(a,"KFold", 5)

###############################################extract_data
def extract_data(set_orig,index):
	ones=np.sum(index)
	column=set_orig.shape[1]
	vect=np.zeros((ones,column))
	count_one=0
	cntr=0
	# pdb.set_trace()
	for i in index:
		if(i==1):
			vect[count_one,:]=set_orig[cntr,:]
			count_one+=1
		cntr+=1
	return vect

a=np.array([0,1,0,1]).reshape(4,1)
q=np.array([[1,2,3,4,5],
			[5,6,7,8,5],
			[9,10,11,12,5],
			[13,14,15,16,5]])
# print(q)
print(extract_data(q,a))
