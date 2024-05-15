import numpy as np
import deepchem as dc

x = np.random.random((4,5))
y = np.random.random((4,1))
print(x)
print(y)

# Creation of a df with both set of information
dataset = dc.data.NumpyDataset(x, y)
print(dataset)