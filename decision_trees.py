import numpy as np

#Decision Tree node
#DT stores a DTnode as root. DTnode is the class that makes up the nodes in the decision tree. It stores
#references to its left and right children and decision information
class dtNode:
	def __init__(self, key):
		self.left = None
		self.right = None
		self.decision = key

#Decision Tree class
class dt:
	def __init__(self, key):
		self.root = dtNode(root, key)

#function that takes training data as input. The labels and the features are binary, but the feature vectors can be of
#any finite dimension. The training feature data (X) should be structured as a 2D numpy array, with each
#row corresponding to a single sample. The training labels (Y) should be structured as a 2D numpy array,
#with each row corresponding to a single label. X and Y should have the same number of rows, with each row
#corresponding to a single sample and its label. max depth is an integer that indicates the maximum depth
#for the resulting decision tree. DT train binary(X,Y,max depth) should return the decision tree trained
#using information gain, limited by some given maximum depth. If max depth is set to -1 then learning only
#stops when we run out of features of our information gain is 0. You may store a decision tree however you
#would like, i.e. a list of lists, a class, a dictionary, etc.
def DT_train_binary(X,Y,max_depth):
	return dt

#function that takes test data X and test labels Y and a learned decision tree model DT, and returns the accuracy on
#the test data using the decision tree for predictions
def DT_test_binary(X,Y,DT):
	accuracy = 0
	return accuracy

#Generate decision trees based on information gain on the training data, and return the tree that gives the
#best accuracy on the validation data.
def DT_train_binary_best(X_train, Y_train, X_val, Y_val):
	return dt

#This function should take a single sample and a trained decision tree and return a single classification.
#The output should be a scalar value. You will use this function with the three trees to generate three
#predictions per sample and use a majority vote to make a final decision. Present your results and accuracy
#on the test data presented below.
def DT_make_prediction(x,DT):
	prediction = 0
	return prediction



#These functions are defined similarly to those above except that the features are now real values. The labels
#are still binary. Your decision tree will need to use questions with inequalities: >, ≥, <, ≤. THINK
#ABOUT HOW THIS CAN BE DONE EFFICIENTLY.
def DT_train_real(X,Y,max_depth):
	return dt

def DT_test_real(X,Y,DT):
	accuracy = 0
	return accuracy

def DT_train_real_best(X_train,Y_train,X_val,Y_val):
	return dt
