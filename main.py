import numpy as np
from autograd import Tensor, sum, mean

data = np.random.rand(10, 5, 5)
tensor = Tensor(data, requires_grad=True)

res = mean(sum(sum(tensor, axis=2), axis=1), axis=0)
res.backward()
print(tensor.grad)
