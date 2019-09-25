import numpy as np
import decision_trees as dt
fileName = input("Data file name? ")
print(fileName)
if fileName != "":
	file = open(fileName)
	X = np.array(file.readline())
	Y = np.array(file.readline())
	file.close()
else:
	X = np.array([[0,1,0,1],[1,1,1,1],[0,0,0,1]])
	Y = np.array([[1],[1],[0]])
print(X)
print(Y)
max_depth = 2
DT = dt.DT_train_binary(X,Y, max_depth)
DT.toString()
test_acc = dt.DT_test_binary(X,Y,DT)
print(test_acc) #DEBUGGING
