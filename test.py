import numpy as np

def getPosition():
    a = np.array([[2, 5, 7, 8, 9, 89], [6, 7, 5, 4, 6, 4]]).reshape(2, 6, 1)
    
    raw, column,depth = a.shape# get the matrix of a raw and column
    
    _positon = np.argmax(a)# get the index of max in the a
    print(a)
    print(a.shape)
    print(_positon)
    m, n = divmod(_positon, column)
    print("The raw is " ,m)
    print("The column is ",  n)
    print("The max of the a is ", a[m][n][0])

def confidence_thresh(cscs, predicts, threshold=5):
    """
    decide in predicts which to be pruned by threshold through cscs.
    Args:
        cscs: class-specific confidence score  [batch_size, cell_size, cell_size, 1]
        predicts: output coord tensor(convert to numpy)  [batch_size, cell_size, cell_size, 18]
        threshold: conf_thresh, defined in config.py
    Return:
        output: a numpy feature tensor  [batch_size, cell_size, cell_size, 18]
    """
    out_tensor = np.zeros_like(predicts)
    for i in range(5):
        for j in range(5):
            for k in range(6):
                if cscs[k][i][j][0] > threshold:
                    out_tensor[k][i][j] = predicts[k][i][j]
    return out_tensor

a = np.array(np.arange(150)).reshape(6, 5, 5, 1)
pred = np.array(np.arange(450)).reshape(6, 5, 5, 3)

res = confidence_thresh(a, pred)

print(res[0][1][2])
#getPosition()