import numpy as np
import decision_trees as dt
# Write-Up 2
#[[1, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1, 1]]
#[[1], [1], [0], [1], [0], [1], [1], [0]]
# DT_make_prediction
#[[0, 1, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 0]]
#[[0], [1], [0]]
# Write-Up 3
#[[4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 1.2], [5, 3.4, 1.6, 0.2], [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.7, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1]]
#[[1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0]]
#test_binX = np.array([[0,1,0,1],[1,1,1,1],[0,0,0,1]])
#test_binY = np.array([[1],[1],[0]])


trainingX1 = np.array([[0,1],[0,0],[1,0],[0,0],[1,1]])
trainingY1 = np.array([[1],[0],[0],[0],[1]])
valX1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
valY1 = np.array([[0], [1], [0], [1]])
testingX1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
testingY1 = np.array([[1], [1], [0], [1]])

trainingX2 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])
trainingY2 = np.array([[0], [1], [0], [0], [1], [0], [1], [1], [1]])
valX2 = np.array([[1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0]])
valY2 = np.array([[0], [0], [1], [0], [1], [1]])
testingX2 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])
testingY2 = np.array([[1], [1], [0], [0], [1], [0], [1], [1], [1]])

DinnerX = np.array([[1, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1, 1]])
DinnerY = np.array([[1], [1], [0], [1], [0], [1], [1], [0]])
firstX = DinnerX[0:5:1]
firstY = DinnerY[0:5:1]
middleX = DinnerX[1:6:1] #SECOND MIDDLE
middleY = DinnerY[1:6:1]
lastX = DinnerX[3:8:1]
lastY = DinnerY[3:8:1]
DinnerTestX = np.array([[0, 1, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 0]])
DinnerTestY = np.array([[0], [1], [0]])



RealX = np.array([[4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 1.2], [5, 3.4, 1.6, 0.2], [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.7, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1]])
RealY = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0]])

print()
print()
print('WRITE UP 1')
max_depth_1 = -1
DT_train_1 = dt.DT_train_binary(trainingX1,trainingY1, max_depth_1)
DT_train_1.toString()
train_acc_1 = dt.DT_test_binary(testingX1,testingY1,DT_train_1)
print('TRAIN1',train_acc_1)

DT_best_1 = dt.DT_train_binary_best(trainingX1,trainingY1,valX1,valY1)
DT_best_1.toString()
best_acc_1 = dt.DT_test_binary(testingX1,testingY1,DT_best_1)
print('BEST1',best_acc_1)

DT_train_2 = dt.DT_train_binary(trainingX2,trainingY2, max_depth_1)
DT_train_2.toString()
train_acc_2 = dt.DT_test_binary(testingX2,testingY2,DT_train_2)
print('TRAIN2',train_acc_2)

DT_best_2 = dt.DT_train_binary_best(trainingX2,trainingY2,valX2,valY2)
DT_best_2.toString()
best_acc_2 = dt.DT_test_binary(testingX2,testingY2,DT_best_2)
print('BEST2',best_acc_2)

print()
print()
print('WRITE UP 2')
print(middleX)
max_depth_2 = 5
DT_first = dt.DT_train_binary(firstX,firstY,max_depth_2)
DT_middle = dt.DT_train_binary(middleX,middleY,max_depth_2)
DT_last = dt.DT_train_binary(lastX,lastY,max_depth_2)

first_acc = dt.DT_test_binary(DinnerTestX,DinnerTestY,DT_first)
print("FIRSTACC",first_acc)
middle_acc = dt.DT_test_binary(DinnerTestX,DinnerTestY,DT_middle)
print("MIDDLEACC",middle_acc)
last_acc = dt.DT_test_binary(DinnerTestX,DinnerTestY,DT_last)
print("LASTACC",last_acc)

DinnerTrees = [DT_first,DT_middle,DT_last]
for tree in DinnerTrees:
	tree.toString()
correct = 0
for i,x in enumerate(DinnerTestX):
	zeroVotes = 0
	oneVotes = 0
	for tree in DinnerTrees:
		if dt.DT_make_prediction(x,tree) == 0:
			zeroVotes+=1
		else:
			oneVotes+=1
	guess = 0
	if oneVotes > zeroVotes:
		guess = 1
	if guess == DinnerTestY[i][0]:
		correct+=1
print('ForrestACC',correct/len(DinnerTestX))

print()
print()
print('WRITE UP 3')
#DTReal = dt.DT_train_real(RealX,RealY,max_depth)
#DTReal.toString()
#DT_test_real(X,Y,DT):
#DT_train_real_best(X_train,Y_train,X_val,Y_val):

#DT_train_binary_best(X_train, Y_train, X_val, Y_val):
