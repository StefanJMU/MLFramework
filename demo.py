import numpy as np
from autograd import *

data = np.arange(8).reshape((1, 8))
tensor_1 = Tensor(data, name="T0", dtype='float', requires_grad=True)

data_2 = np.arange(8).reshape((1, 8))
tensor_2 = Tensor(data_2, name="T1", dtype='float', requires_grad=True)

tensor_3 = np.e ** tensor_1
print(tensor_3)
loss = sum(tensor_3, axis=(0, 1))
print(loss)
loss.backward()
print(tensor_1.grad)

#tensor_1[0, 0] = 8

#tensor_3 = tprod(tensor_1, tensor_2)
#tensor_4 = prod(tensor_3, axis=1)

#tensor_4.backward()
#print(tensor_2.grad)
#print(np.broadcast_to(data_2, data.shape))

import torch









