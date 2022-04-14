import numpy as np
from autograd import *

#data = np.random.rand(10, 5, 5)
#tensor_1 = Tensor(data, name="T0", requires_grad=True)

#data_2 = np.random.rand(10, 5, 1)
#tensor_2 = Tensor(data_2, name="T1", requires_grad=True)

#data_3 = np.array([[1,2,3,4,5,6]], dtype='float')
#tensor_3 = Tensor(data_3, requires_grad=True)

#data_4 = np.array([[2]], dtype='float')
#tensor_4 = Tensor(data_4, requires_grad=True)

#ten = concat([tensor_1, tensor_2], axis=2)
#res = mean(sum(sum(ten, axis=2), axis=1), axis=0)
#res.backward()
#print(tensor_1.grad)
#print(tensor_2.grad)

t1 = Tensor(np.array([[[1,1,1],[1,1,1]]]))
t2 = Tensor(np.array([[[2,2],[2,2]]]))

res = mix(t1, t2, mask=np.array([[1,0,1],[1,0,1]]))
print(res)
