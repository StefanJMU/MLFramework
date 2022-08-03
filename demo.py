import numpy as np
from autograd import *

data = np.arange(16).reshape((2, 2, 4))
tensor_1 = Tensor(data, name="T0", dtype='float', requires_grad=True)

data_2 = np.arange(8).reshape((8))
tensor_2 = Tensor(data_2, name="T1", dtype='float', requires_grad=True)

tensor_3 = np.e ** tensor_1






