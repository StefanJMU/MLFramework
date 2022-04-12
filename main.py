import numpy as np
from autograd import Tensor, sum, mean, tsum, tprod, transpose, prod, power, root

data = np.random.rand(10, 5, 5)
tensor_1 = Tensor(data, name="T0", requires_grad=True)

data_2 = np.random.rand(10, 5, 1)
tensor_2 = Tensor(data_2, requires_grad=True)

data_3 = np.array([[1,2,3,4,5,6]], dtype='float')
tensor_3 = Tensor(data_3, requires_grad=True)

data_4 = np.array([[2]], dtype='float')
tensor_4 = Tensor(data_4, requires_grad=True)


res = mean(sum(root(tensor_3, tensor_4), axis=1), axis=0)
res.backward()
#print(tensor_3.grad)
#print(tensor_4.grad)

tensor_x = mean(tensor_1[:, 1, 1], axis=0)
tensor_x.backward()
print(tensor_1.grad)

#res = tsum(tensor_1, tensor_2)
#res = mean(sum(prod(res, axis=2), axis=1), axis=0)
#res.backward()
#print(res.shape)
#print(tensor_1.grad)
#print(tensor_2.grad)
