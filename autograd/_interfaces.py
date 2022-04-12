from ._operation import *
from ._computation_graph import ComputationGraph
from ._tensor import Tensor

def tsum(tensor_1: Tensor, tensor_2: Tensor):
    operation = TensorSum()
    res_data = operation.forward(tensor_1, tensor_2)
    return Tensor(data=res_data,
                  computation_graph=ComputationGraph(operation=operation, tensor_1=tensor_1, tensor_2=tensor_2),
                  requires_grad=(tensor_1.requires_grad or tensor_2.requires_grad))

def tprod(tensor_1: Tensor, tensor_2: Tensor):
    operation = TensorProduct()
    res_data = operation.forward(tensor_1, tensor_2)
    return Tensor(data=res_data,
                  computation_graph=ComputationGraph(operation=operation, tensor_1=tensor_1, tensor_2=tensor_2),
                  requires_grad=(tensor_1.requires_grad or tensor_2.requires_grad))

def sum(tensor: Tensor, axis: int):
    operation = Sum(axis)
    res_data = operation.forward(tensor)
    return Tensor(data=res_data,
                  computation_graph=ComputationGraph(operation=operation, tensor_1=tensor),
                  requires_grad=tensor.requires_grad)

def prod(tensor: Tensor, axis: int):
    operation = Prod(axis)
    res_data = operation.forward(tensor)
    return Tensor(data=res_data,
                  computation_graph=ComputationGraph(operation=operation, tensor_1=tensor),
                  requires_grad=tensor.requires_grad)

def mean(tensor: Tensor, axis: int):
    operation = Mean(axis)
    res_data = operation.forward(tensor)
    return Tensor(data=res_data,
                  computation_graph=ComputationGraph(operation=operation, tensor_1=tensor),
                  requires_grad=tensor.requires_grad)

def transpose(tensor: Tensor, axis_permutation):
    operation = Transpose(axis_permutation)
    res_data = operation.forward(tensor)
    return Tensor(data=res_data,
                  computation_graph=ComputationGraph(operation=operation, tensor_1=tensor),
                  requires_grad=tensor.requires_grad)

def reshape(tensor: Tensor, shape: tuple):
    operation = Reshape(shape)
    res_data = operation.forward(tensor)
    return Tensor(data=res_data,
                  computation_graph=ComputationGraph(operation=operation, tensor_1=tensor),
                  requires_grad=tensor.requires_grad)

def square(tensor: Tensor):
    operation = Square()
    res_data = operation.forward(tensor)
    return Tensor(data=res_data,
                  computation_graph=ComputationGraph(operation=operation, tensor_1=tensor),
                  requires_grad=tensor.requires_grad)

def power(tensor_1: Tensor, tensor_2: Tensor):
    operation = Power()
    res_data = operation.forward(tensor_1, tensor_2)
    return Tensor(data=res_data,
                  computation_graph=ComputationGraph(operation=operation, tensor_1=tensor_1, tensor_2=tensor_2),
                  requires_grad=(tensor_1.requires_grad or tensor_2.requires_grad))

def sqrt(tensor: Tensor):
    operation = Sqrt()
    res_data = operation.forward(tensor)
    return Tensor(data=res_data,
                  computation_graph=ComputationGraph(operation=operation, tensor_1=tensor),
                  requires_grad=tensor.requires_grad)

def root(tensor_1: Tensor, tensor_2: Tensor):
    operation = Root()
    res_data = operation.forward(tensor_1, tensor_2)
    return Tensor(data=res_data,
                  computation_graph=ComputationGraph(operation=operation, tensor_1=tensor_1, tensor_2=tensor_2),
                  requires_grad=(tensor_1.requires_grad or tensor_2.requires_grad))