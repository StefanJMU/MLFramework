import numpy as np
from autograd import *

data = np.random.rand(10, 5, 5)
tensor_1 = Tensor(data, name="T0", requires_grad=True)

data_2 = np.random.rand(10, 5, 1)
tensor_2 = Tensor(data_2, name="T1", requires_grad=True)

s = tsum(tensor_1, tensor_2)
res = mean(sum(sum(s, axis=2), axis=1), axis=0)
res.backward()
print(tensor_2.grad)





