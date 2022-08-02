
from ._operation import *
from ._tensor import Tensor, TensorList, unary_interface, binary_interface


@unary_interface
def sum(tensor: Tensor, axis: int, keepdims=False, name: str = None):
    operation = Sum(axis, keepdims)
    return operation.forward(tensor), operation, name


@unary_interface
def prod(tensor: Tensor, axis: int, keepdims=False, name: str = None):
    operation = Prod(axis, keepdims)
    return operation.forward(tensor), operation, name


@unary_interface
def mean(tensor: Tensor, axis: int, keepdims=False, name: str = None):
    operation = Mean(axis, keepdims)
    return operation.forward(tensor), operation, name


@unary_interface
def transpose(tensor: Tensor, axis_permutation, name: str = None):
    operation = Transpose(axis_permutation)
    return operation.forward(tensor), operation, name


@unary_interface
def reshape(tensor: Tensor, shape: tuple, name: str = None):
    operation = Reshape(shape)
    return operation.forward(tensor), operation, name


@unary_interface
def square(tensor: Tensor, name: str = None):
    operation = Square()
    return operation.forward(tensor), operation, name


@unary_interface
def sqrt(tensor: Tensor, name: str = None):
    operation = Sqrt()
    return operation.forward(tensor), operation, name


@binary_interface
def root(tensor_1: Tensor, tensor_2: Tensor, name: str = None):
    operation = Root()
    return operation.forward(tensor_1, tensor_2), operation, name


@unary_interface
def concat(tensor_list, axis: int, name: str = None):
    operation = Concatenate(axis)
    return operation.forward(tensor_list), operation, name


@binary_interface
def mix(tensor_1: Tensor, tensor_2: Tensor, mask: np.array, name: str = None):
    operation = Mix(mask)
    res_data = operation.forward(tensor_1, tensor_2)
    return res_data, operation, name


@unary_interface
def softmax(tensor_1: Tensor, axis: int, name: str = None):
    operation = Softmax(axis)
    res_data = operation.forward(tensor_1)
    return res_data, operation, name

@unary_interface
def unsqueeze(tensor_1: Tensor, axis: int, name: str = None):
    operation = Unsqueeze(axis)
    return operation.forward(tensor_1), operation, name


@unary_interface
def squeeze(tensor_1: Tensor, axis: int, name: str = None):
    operation = Squeeze(axis)
    return operation.forward(tensor_1), operation, name


@unary_interface
def tile(tensor_1: Tensor, reps, name: str = None):
    operation = Tile(reps)
    return operation.forward(tensor_1), operation, name
