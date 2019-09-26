import numpy as np
import decision_trees as dt

test_binX = np.array([[0,1,0,1],[1,1,1,1],[0,0,0,1]])
test_binY = np.array([[1],[1],[0]])

trainingX1 = np.array([[0,1],[0,0],[1,0],[0,0],[1,1]])
trainingY1 = np.array([[1],[0],[0],[0],[1]])

testingX1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
testingY1 = np.array([[1], [1], [0], [1]])

valX1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
valY1 = np.array([[0], [1], [0], [1]])

trainingX2 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])
trainingY2 = np.array([[0], [1], [0], [0], [1], [0], [1], [1], [1]])

testingX2 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])
testingY2 = np.array([[1], [1], [0], [0], [1], [0], [1], [1], [1]])

valX2 = np.array([[1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0]])
valY2 = np.array([[0], [0], [1], [0], [1], [1]])

max_depth = 2
DT = dt.DT_train_binary(trainingX2,trainingY2, max_depth)

test_acc = dt.DT_test_binary(testingX2,testingY2,DT)
print(test_acc)
