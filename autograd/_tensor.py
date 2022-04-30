import numpy as np
from ._operation import Slice, Reshape
from ._computation_graph import ComputationGraph

from typing import List, Union


class Tensor:

    """
        Class modelling tensor objects. The first dimension is always interpreted as a batch dimension

        Attributes
        ----------
            data: np.array
                underlying numpy array
            dtype: str, default='float'
                datatype of underlying numpy data array
            shape: tuple
                shape of the underlying numpy array
            requires_grad: bool, default=False
                flag indicating, whether the tensor requires gradient calculation
            computation_graph: ComputationGraph, default=None
                computation graph for compound tensors, resulting of operations
            name: str, default=None
                name of the tensor. If None, the next tensor_code is used
            grad: np.array
                gradient of the tensor, with respect to backwarded computation-graphs, the tensor participates in.
                The attribute is only present, if the tensor requires gradients.
            jacobian: np.array
                jacobian of the tensor, with respect to all elements of a backwarded tensor, having this tensor in its
                computation graph.
                The attribute is only present, if the tensor requires gradients.
            earmark: int
                the earmark keeps track of modifications of the data underlying the tensor. Used to check validity
                of the computation graphs, in which the tensor is involved
    """

    # tensor counter
    tensor_code = 0

    def __init__(self,
                 data: np.array = None,
                 shape=None,
                 dtype='float',
                 requires_grad: bool = False,
                 computation_graph=None,
                 name: str = None):

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

        if requires_grad and dtype != "float":
            raise ValueError("A Tensor requiring gradients can only be of type float")

        self.requires_grad = requires_grad
        self.computation = computation_graph
        if self.requires_grad:
            self.grad = np.zeros(self.shape[1:], dtype="float")  #ignore batch-dimension
            self._jacobian_pool = []
            self.jacobian = None

        #keeps track of modifications of the underlying data
        self.earmark = 0

    def size(self):
        return self.data.size

    def __getitem__(self, key):
        """
            Calculates the slice-operation on the tensor

            Parameters
            ----------
            key: slice-object
                slice identifier
            Returns
            -------
                Tensor sliced out of this tensor according to key
        """
        return slice(self, key)

    def backward(self, error_signal: np.array = None, **kwargs):
        """
            Calculates the backward operation of the tensor

            Parameters
            ----------
            error_signal: np.array
                error_signals for the elements of the tensor. Required to have a matching shape with the tensor
        """
        if self.requires_grad:

            if error_signal is not None:
                if error_signal.shape != self.shape:
                    raise ValueError(f'Gradient has incompatible shape. Expected {self.shape}'
                                     f'Got {error_signal.shape}')
            elif self.shape == ():  # starting the differentiation chain.
                error_signal = np.ones_like(self.data)
            else:
                raise AttributeError('Can calculate gradients only for scalar tensors directly.'
                                     'Consider the usage of functions masked_backward or jacobian.')

            if self.computation is not None:  # composed tensor
                self.computation.backward(error_signal, **kwargs)
            else:  # atomic and gradient requiring tensor
                # reshape into 1D stack of gradients
                error_signal = np.reshape(error_signal, [-1] + list(self.grad.shape))
                # conflate by summation
                error_signal = np.sum(error_signal, axis=0)
                # update gradient
                self.grad = self.grad + error_signal

                # jacobian calculation
                if 'jacobian' in kwargs:
                    shape_meta = kwargs['jacobian']
                    self._jacobian_pool.append(self.grad)
                    self.zero_grad()
                    if len(self._jacobian_pool) == shape_meta[-1]:
                        self.jacobian = np.reshape(np.stack(self._jacobian_pool, axis=0),
                                                   newshape=shape_meta[0] + self.grad.shape)
                        self._jacobian_pool = []

    def select_backward(self, selection: tuple, **kwargs):
        """
            Calculates derivative for a single element of a tensor

            Parameters
            ---------
            selection : tuple
                tuple localizing the element to be differentiated in the tensor

        """
        if self.requires_grad:
            if len(selection) != len(self.data.shape):
                raise ValueError(f'The selection tuple is required to have {len(self.data.shape)}. Got {len(selection)}')
            selected = slice(self, selection)
            selected.backward(**kwargs)

    def backward_jacobian(self):
        """
            Calculates the jacobian, i.e. the partial derivative with respect to every element in the tensor
        """
        if self.requires_grad:
            flattened = reshape(self, shape=(-1,))
            for i in range(flattened.shape[0]):
                flattened.select_backward((i,), jacobian=(self.shape, flattened.shape[0]))

    def zero_grad(self):
        """
            Zeros the gradient of the tensor
        """
        self.grad = np.zeros_like(self.grad)

    def __repr__(self):
        if self.computation is None:
            print("asdfasdf")
        return f'{type(self).__name__}: {self.name} \n' \
               f'   dtype: {self.data.dtype} \n' \
               f'   shape: {self.data.shape} \n' \
               f'   autograd: {self.computation.__repr__()} \n' \
               f'   data: {self.data}'


class TensorList:

    """
        Wraps a list of Tensors
    """

    def __init__(self, tensors: List[Tensor]):
        self.tensors = tensors
        self.earmark = [tensor.earmark for tensor in tensors]
        self.requires_grad = np.logical_or(*[tensor.requires_grad for tensor in tensors])
        self.name = f'[{",".join([tensor.name for tensor in tensors])}]'

    def backward(self, error_signal: List, **kwargs):
        if len(error_signal) != len(self.tensors):
            raise ValueError(f'TensorList expected {len(self.tensors)} error signals. Got {len(error_signal)}')
        for i in range(len(error_signal)):
            self.tensors[i].backward(error_signal[i], **kwargs)

def unary_interface(func):
    def wrapper(tensor: Union[Tensor, List[Tensor]], *args, **kwargs):
        res_data, operation, name = func(tensor, *args, **kwargs)
        if type(tensor) is list:
            tensor = TensorList(tensor)
        if tensor.requires_grad:
            return Tensor(data=res_data,
                          computation_graph=ComputationGraph(operation=operation, tensor_1=tensor),
                          requires_grad=True,
                          name=name)
        else:
            return Tensor(data=res_data, name=name)
    return wrapper

def binary_interface(func):
    def wrapper(tensor_1: Tensor, tensor_2: Tensor, *args, **kwargs):
        res_data, operation, name = func(tensor_1, tensor_2, *args, **kwargs)
        if tensor_1.requires_grad or tensor_2.requires_grad:
            return Tensor(data=res_data,
                          computation_graph=ComputationGraph(operation=operation, tensor_1=tensor_1, tensor_2=tensor_2),
                          requires_grad=True,
                          name=name)
        else:
            return Tensor(data=res_data, name=name)
    return wrapper

@unary_interface
def slice(tensor, key):
    operation = Slice(key)
    return operation.forward(tensor), operation

@unary_interface
def reshape(tensor: Tensor, shape: tuple):
    operation = Reshape(shape)
    return operation.forward(tensor), operation