import numpy as np
import math
#Decision Tree node
#DTree stores a Node as root. Node is the class that makes up the nodes in the decision tree. It stores
#references to its left and right children and decision information
class Node:
	def __init__(self, depth):
		self.left = None
		self.right = None
		self.feat = None
		self.label = None
		self.depth = depth

	#traversing through the nodes, used for prediction and accuracy
	def trav(self, x):
		#base case is at a leaf node, there is no feature
		if self.feat is None:
			return self.label
		elif x[self.feat] == 0:
			return left.trav(x)
		else:
			return right.trav(x)

	def split(self, X, Y, max_depth):
		#assign node a label
		zeros = 0
		ones = 0
		for y in Y:
			if y[0] == 0:
				zeros += 1
			else:
				ones += 1
		if zeros > ones:
			self.label = 0
		else:
			self.label = 1
		#check base cases: at max depth, no samples, out of features, or no IG  (prob don't need to check no samples or features... IG will take care of it)
		if self.depth == max_depth or len(X) == 0 or self.depth == len(X[0]) or best_IG(X,Y) is None:
			return
		#pick feature to split on based on max IG
		self.feat = best_IG(X,Y)
		#split X and Y into lefts and rights
		left_X = np.array([[]])
		left_Y = np.array([[]])
		right_X = np.array([[]])
		right_Y = np.array([[]])
		for i,x in enumerate(X):
			if x[self.feat] == 0:
				print('left') #DEBUGGING
				#HAVING TROUBLE WITH THE ARRAYS :(
				left_X = np.append(left_X,x)
				left_Y = np.append(left_Y,Y[i])
			else:
				print('right') #DEBUGGING
				right_X = np.append(right_X,x)
				right_Y = np.append(right_Y,Y[i])
		#make left and right nodes and recurse
		print(left_Y) #DEBUGGING
		print(left_X) #DEBUGGING
		self.left = Node(self.depth + 1)
		self.right = Node(self.depth + 1)
		self.left.split(left_X, left_Y, max_depth)
		self.right.split(right_X, right_Y, max_depth)

	def toString(self):
		print("Feat", self.feat, "Label", self.label, "Depth ", self.depth)
		if self.left is not None:
			self.left.toString()
		if self.right is not None:
			self.right.toString()


#Decision Tree class
class DTree:
	def __init__(self):
		self.root = Node(0)
	def predict(self, x):
		value = self.root.trav(x)
		return value
	def build(self, X, Y, max_depth):
		self.root.split(X, Y, max_depth)
		return
	def toString(self):
		self.root.toString()


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
	dt = DTree()
	dt.build(X, Y, max_depth)
	return dt

#function that takes test data X and test labels Y and a learned decision tree model DT, and returns the accuracy on
#the test data using the decision tree for predictions
def DT_test_binary(X,Y,DT):
	correct = 0
	for i, x in enumerate(X):
		if DT.predict(x) == Y[i][0]:
			correct+=1
	accuracy = correct/len(X)
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
	return DT.predict(x)



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

#returns index of the max IG, none if all zeros NEED THIS IMPLEMENTED
def best_IG(X,Y):
	return 0 #DEBUGGING
	H = 0 # math.log(a,base)
	maxIG = 0
	index = None
	for i,x in enumerate(X):
		H_left = 0
		H_right = 0
		IG = 0 #H - amountLeft/total*H_left - amountRight/total*H_right
		if IG > maxIG:
			maxIG = IG
			index = i
	return index
