import numpy as np
import matplotlib.pyplot as plt 

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo: 
        dict = pickle.load(fo, encoding='bytes')
    return dict

X = unpickle("./data/cifar-10-batches-py/data_batch_1")[b'data']
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

plt.imshow(X[0])
plt.show()