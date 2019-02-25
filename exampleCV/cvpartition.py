import numpy as np
import random
from pdb import set_trace as bp
###############################################conusionMat
def confusionmat(test_vec,test_pred):
	CM=np.zeros((2,2))													#This is confusionmat function which was used in the MATLAB code to return 
	for i in range(len(test_vec)):										#Accuracy, Specificity and Sensitivity.
		if(test_vec[i]<0 and test_pred[i]<0): #true negative			#
			CM[0,0]+=1													#
		if(test_vec[i]>0 and test_pred[i]>0): #true positive
			CM[1,1]+=1											
		if(test_vec[i]<0 and test_pred[i]>0): #false positive
			CM[0,1]+=1
		if(test_vec[i]>0 and test_pred[i]<0): #false negative
			CM[1,0]+=1
	return CM
###############################################cvpartition
def cvpartition(sample,fold_type, fold_number):
	
	size_sample=sample.shape[0]
	parameter_random_vectors={}
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
		###firs and rest determine how many elements must be inside each folding
	if(fold_type=="LOO"):																					#if LOO is chosen then here we allocate the 
		for i in range(size_sample):																		#sample size folds into parameter_random_vectors
			parameter_random_vectors["fold_rand" +str(i)]=random.sample(range(size_sample), 1)				#
	
	if(fold_type=="KFold"):
		for i in range(fold_number):																		#Here we allocate random numbers for each 
			if(flag_equal_size_vect==1):																	#iteration (for each FOLDS)
				parameter_random_vectors["fold_rand" +str(i)]=random.sample(range(size_sample), firs)		#
			if(flag_equal_size_vect==0):																	#
				if(i==0):																					#
					parameter_random_vectors["fold_rand" +str(i)]=random.sample(range(size_sample), firs)	#
				if(i!=0):																					#
					parameter_random_vectors["fold_rand" +str(i)]=random.sample(range(size_sample), rest)	#
		# for i in range(fold_number):
		# 	print(len(parameter_random_vectors["fold_rand" +str(i)]))
	test_index=parameter_random_vectors
	train_index={}
	zero_one_index_test={}
	zero_one_index_train={}
	if(fold_type=="KFold"):																				 	#Here we return index values same as MATLAB Code of function cvpartion
		for i in range(len(test_index)):																	#it is easy to go through all this code. those numbers that are chosen
			x_test=np.zeros(size_sample).reshape(1,size_sample)												#as fold will be indexed zero in the training set and those which are
			x_train=np.ones(size_sample).reshape(1,size_sample)												#not chosen as fold randomly will be indexed as one. For the test set 
			for j in test_index["fold_rand" +str(i)]:														#the same procedure applies but the other way around. Those which are
				for k in range(size_sample):																#chosen randomly will be indexed 1 and those are not will be indexed 0
					if(k==j):																				#Then this index vector will be used in extract_data function to 
						x_test[0,k]=1																		#return correpsonding samples in the data set.
						x_train[0,k]=0																		#
																											#The same procedure applies for "LOO"
			zero_one_index_train["Row"+str(i)]=x_train														#
			zero_one_index_test["Row"+str(i)]=x_test														#
		tr_size=np.arange(len(parameter_random_vectors))													#
																											#
		for i in range(fold_number):																		#
			tr_size[i]=len(parameter_random_vectors["fold_rand" +str(i)])									#
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
	
	if(fold_type=="LOO"):
		for i in range(len(test_index)):
			x_test=np.zeros(size_sample).reshape(1,size_sample)
			x_train=np.ones(size_sample).reshape(1,size_sample)
			for j in test_index["fold_rand" +str(i)]:
				for k in range(size_sample):
					if(k==j):
						x_test[0,k]=1
						x_train[0,k]=0
						break
			zero_one_index_train["Row"+str(i)]=x_train
			zero_one_index_test["Row"+str(i)]=x_test	
		# tr_size=np.arange(len(parameter_random_vectors))
		# for i in range(size_sample):
		# 	tr_size[i]=len(parameter_random_vectors["fold_rand" +str(i)])
		##################################################################assign return values
		
		training=zero_one_index_train
		test=zero_one_index_test		
		##################################################################
		print("LOO cross validation partition")
		print("NumObservations:" +str(size_sample))
		print("NumTestSets:" +str(size_sample))
		print("TrainSize: ALL" +str(size_sample-1))
		print("TestSize: you can guess!!!")
		return training,test
# a=np.arange(84).reshape(84,1)

# cvpartition(a,"KFold", 5)

###############################################extract_data
def extract_data(set_orig,index):					#This function take the indexes from the cvpartition function and return the values that are indexed as one in the index
													#vector.
	ones=np.sum(index)								#
	column=set_orig.shape[1]						#
	vect=np.zeros((ones,column))					#
	count_one=0										#
	cntr=0											#
	# pdb.set_trace()								#
	for i in index:									#
		if(i==1):									#
			vect[count_one,:]=set_orig[cntr,:]		#
			count_one+=1							#
		cntr+=1										#
	return vect

