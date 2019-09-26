import numpy as np
import math
#Decision Tree node
#DTree stores a Node as root. Node is the class that makes up the nodes in the decision tree. It stores
#references to its left and right children and decision information
class Node:
	def __init__(self, depth):
		#left child (aka no side,  aka 0 side)
		self.left = None
		#right child (aka no side,  aka 1 side)
		self.right = None
		#the question it asks (which is just an index)
		self.feat = None
		#the decision
		self.label = None
		self.depth = depth

	#searching through the nodes, used for prediction and accuracy
	def search(self, x):
		#base case is at a leaf node, there is no feature
		if self.feat is None:
			return self.label
		elif x[self.feat] == 0:
			return self.left.search(x)
		else:
			return self.right.search(x)

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
		#pick feature to split on based on max IG
		if self.depth == max_depth:
			return
		feat = best_IG(X,Y)
		#check base cases: at max depth, no samples, out of features, or no IG  (prob don't need to check no samples or features... IG will take care of it)
		if self.depth == max_depth or len(X) == 0 or self.depth == len(X[0]) or feat is None:
			return
		self.feat = feat
		#split X and Y into lefts and rights
		#left_X = np.array([[]])
		#left_Y = np.array([[]])
		#right_X = np.array([[]])
		#right_Y = np.array([[]])
		left_X = []
		left_Y = []
		right_X = []
		right_Y = []
		for i,x in enumerate(X):
			if x[self.feat] == 0:
				#HAVING TROUBLE WITH THE ARRAYS :(
				#left_X = np.append(left_X,x)
				#left_Y = np.append(left_Y,Y[i])
				left_X += [x]
				left_Y += [Y[i]]
			else:
				#right_X = np.append(right_X,x)
				#right_Y = np.append(right_Y,Y[i])
				right_X += [x]
				right_Y += [Y[i]]
		#make left and right nodes and recurse
		left_X = np.array(left_X)
		left_Y = np.array(left_Y)
		right_X = np.array(right_X)
		right_Y = np.array(right_Y)
		self.left = Node(self.depth + 1)
		self.right = Node(self.depth + 1)
		self.left.split(left_X, left_Y, max_depth)
		self.right.split(right_X, right_Y, max_depth)

	def toString(self):
		print('Depth',self.depth)
		if self.feat is None:
			print('Label',self.label)
		else:
			print('Feat',self.feat,'?')
			if self.left is not None:
				print('Left')
				self.left.toString()
			if self.right is not None:
				print('Right')
				self.right.toString()
		print()

#Decision Tree class
class DTree:
	def __init__(self):
		self.root = Node(0)
		self.means = None
	def predict(self, x):
		return self.root.search(x)
	def build(self, X, Y, max_depth):
		self.root.split(X, Y, max_depth)
	def toString(self):
		self.root.toString()
	def fixData(self, X):
		numSamples = len(X)
		numFeats = len(X[0])
		for i in range(0, numFeats):
			for j in range(0, numSamples):
				if X[j][i] <= self.means[i]:
					X[j][i] = 0
				else:
					X[j][i] = 1
		return X

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
	forest = []
	numFeats = len(X_train[0])
	maxDT = None
	maxAccuracy = 0
	#build all depth dts
	for i in range(0, numFeats):
		temp = DTree()
		temp.build(X_train, Y_train, i)
		forest += [temp]
	#use DT_test_binary to find accuracy of each dt on validation data
	for i in range(0, numFeats):
		tempAcc = DT_test_binary(X_val, Y_val, forest[i])
		if tempAcc > maxAccuracy:
			maxAccuracy = tempAcc
			maxDT = i
	#return best dt
	return forest[maxDT]

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
	means = findMeans(X)
	dt = DTree()
	dt.means = means
	fixedX = dt.fixData(X)
	dt.build(fixedX,Y,max_depth)
	return dt

def DT_test_real(X,Y,DT):
	fixedX = DT.fixData(X)
	return DT_test_binary(fixedX,Y,DT)

def DT_train_real_best(X_train,Y_train,X_val,Y_val):
	dt = DTree()
	dt.means = findMeans(X_train)
	fixedTrain = dt.fixData(X_train)
	fixedVal = dt.fixData(X_val)
	return DT_train_binary_best(fixedTrain,Y_train,fixedVal,Y_val)

#returns index of the feature that gives maxIG, None if all zeros
def best_IG(X,Y):
	zeros = 0
	ones = 0
	for y in Y:
		if y[0] == 0:
			zeros += 1
		else:
			ones += 1
	total = zeros + ones
	if zeros == 0 or ones == 0:
		return None
	H = -(zeros/total) * math.log((zeros/total),2) -(ones/total) * math.log((ones/total),2)
	maxIG = 0
	index = None
	for feat_idx in range(0,len(X[0])):
		left_zeros = 0
		left_ones = 0
		right_zeros = 0
		right_ones = 0
		for sample_idx,x in enumerate(X):
			if x[feat_idx]==0:
				if Y[sample_idx][0] == 0:
					left_zeros+=1
				else:
					left_ones+=1
			else:
				if Y[sample_idx][0] == 0:
					right_zeros+=1
				else:
					right_ones+=1
		left_total = left_zeros + left_ones
		right_total = right_zeros + right_ones
		if left_total == 0 or right_total == 0:
			continue
		H_left = 0
		H_right = 0
		if left_zeros == 0:
			H_left = -(left_ones/left_total) * math.log((left_ones/left_total),2)
		elif left_ones == 0:
			H_left = -(left_zeros/left_total) * math.log((left_zeros/left_total),2)
		else:
			H_left = -(left_zeros/left_total) * math.log((left_zeros/left_total),2) -(left_ones/left_total) * math.log((left_ones/left_total),2)
		if right_zeros == 0:
			H_right = -(right_ones/right_total) * math.log((right_ones/right_total),2)
		elif right_ones == 0:
			H_right = -(right_zeros/right_total) * math.log((right_zeros/right_total),2)
		else:
			H_right = -(right_zeros/right_total) * math.log((right_zeros/right_total),2) -(right_ones/right_total) * math.log((right_ones/right_total),2)
		IG = H - (left_total/total)*H_left - (right_total/total)*H_right
		if IG > maxIG:
			maxIG = IG
			index = feat_idx
	return index

#Function to calculate means for real valued data
def findMeans(X):
	means = []
	numSamples = len(X)
	numFeats = len(X[0])
	for i in range(0, numFeats):
		mean = 0
		for j in range(0, numSamples):
			mean = mean + X[j][i]
		mean = mean / numSamples
		means = means + [mean]
	return means