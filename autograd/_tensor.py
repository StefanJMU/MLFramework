import numpy as np
from ._operation import Slice
from ._computation_graph import ComputationGraph

class Tensor:

    tensor_code = 0

    def __init__(self,
                 data: np.array = None,
                 shape=None,
                 dtype='float',
                 requires_grad: bool = True,
                 computation_graph=None,
                 name: str = None):
        """
            TODO: implement
            construction options
        """
        if name is not None:
            self.name = name
        else:
            self.name = f'#{Tensor.tensor_code}'
            Tensor.tensor_code += 1

        if dtype is None:
            raise ValueError("Cannot construct Tensor of type None")
        if data is not None:
            # TODO: expand batch dimension is shape has length 1
            self.data = np.copy(data)
            self.shape = self.data.shape
        else:
            if shape is None:
                raise ValueError("Tensor misses data or shape.")
            self.data = np.zeros(shape, dtype=dtype)
            self.shape = shape

        #todo: check matching of dtype and typ of tensor

        #maybe shape and type should be saved individually
        if requires_grad and dtype != "float":
            raise ValueError("A Tensor requiring gradients can only be of type float")

        self.requires_grad = requires_grad
        if self.requires_grad:
            self.grad = np.zeros(self.shape[1:], dtype="float")  #ignore batch-dimension
            self.computation = computation_graph

        #keeps track of modifications of the underlying data
        self.earmark = 0

    def __getitem__(self, key):
        return slice(self, key)

    def backward(self, error_signal: np.array = None):
        if self.requires_grad:

            if error_signal is not None:
                if error_signal.shape != self.shape:
                    raise ValueError(f'Gradient has incompatible shape. Expected {self.shape}'
                                     f'Got {error_signal.shape}')
            elif self.shape == ():  # starting the differentiation chain.
                error_signal = np.ones_like(self.data)
            else:
                raise AttributeError(f'Can calculate gradients only for scalar tensors directly.')

            if self.computation is not None:  # composed tensor
                self.computation.backward(error_signal)
            else:  # atomic and gradient requiring tensor
                # reshape into 1D stack of gradients
                error_signal = np.reshape(error_signal, [-1] + list(self.shape[1:]))
                # conflate by summation
                error_signal = np.sum(error_signal, axis=0)
                # update gradient
                self.grad = self.grad + error_signal

    def zero_grad(self):
        self.grad = np.zeros_like(self.grad)

    def __repr__(self):
        return f'{type(self).__name__}: {self.name} \n' \
               f'   dtype: {self.data.dtype} \n' \
               f'   shape: {self.data.shape} \n' \
               f'   autograd: {self.computation.__repr__()} \n' \
               f'   data: {self.data}'


def slice(tensor, key):
    operation = Slice(key)
    res_data = operation.forward(tensor)
    return Tensor(data=res_data,
                  computation_graph=ComputationGraph(operation=operation, tensor_1=tensor),
                  requires_grad=tensor.requires_grad)