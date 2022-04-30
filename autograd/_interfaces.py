
from ._operation import *
from ._tensor import Tensor, TensorList, unary_interface, binary_interface


@binary_interface
def tsum(tensor_1: Tensor, tensor_2: Tensor):
    operation = TensorSum()
    return operation.forward(tensor_1, tensor_2), operation

@binary_interface
def tprod(tensor_1: Tensor, tensor_2: Tensor):
    operation = TensorProduct()
    return operation.forward(tensor_1, tensor_2), operation

@unary_interface
def sum(tensor: Tensor, axis: int):
    operation = Sum(axis)
    return operation.forward(tensor), operation

@unary_interface
def prod(tensor: Tensor, axis: int):
    operation = Prod(axis)
    return operation.forward(tensor), operation

@unary_interface
def mean(tensor: Tensor, axis: int):
    operation = Mean(axis)
    return operation.forward(tensor), operation

@unary_interface
def transpose(tensor: Tensor, axis_permutation):
    operation = Transpose(axis_permutation)
    return operation.forward(tensor), operation

@unary_interface
def reshape(tensor: Tensor, shape: tuple):
    operation = Reshape(shape)
    return operation.forward(tensor), operation

@unary_interface
def square(tensor: Tensor):
    operation = Square()
    return operation.forward(tensor), operation

@binary_interface
def power(tensor_1: Tensor, tensor_2: Tensor):
    operation = Power()
    return operation.forward(tensor_1, tensor_2), operation

@unary_interface
def sqrt(tensor: Tensor):
    operation = Sqrt()
    return operation.forward(tensor), operation

@binary_interface
def root(tensor_1: Tensor, tensor_2: Tensor):
    operation = Root()
    return operation.forward(tensor_1, tensor_2), operation

@binary_interface
def tdiv(tensor_1: Tensor, tensor_2: Tensor):
    operation = TensorDiv()
    return operation.forward(tensor_1, tensor_2), operation

@unary_interface
def concat(tensor_list, axis: int):
    operation = Concatenate(axis)
    return operation.forward(tensor_list), operation

@binary_interface
def mix(tensor_1: Tensor, tensor_2: Tensor, mask: np.array):
    operation = Mix(mask)
    res_data = operation.forward(tensor_1, tensor_2)
    return res_data, operation

@unary_interface
def softmax(tensor_1: Tensor, axis: int):
    operation = Softmax(axis)
    res_data = operation.forward(tensor_1)
    return res_data, operation

