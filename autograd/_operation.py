import numpy as np
from abc import ABCMeta, abstractmethod
from ._tensor import Tensor
from ._computation_graph import ComputationGraph

"""
    TODO: Write an numpy exception wrapping mechanism

"""

class Operation(metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def check_compatibility(self, operand_1: Tensor, operand_2: Tensor = None):
        ...

    @abstractmethod
    def _forward(self, operand_1: Tensor, operand_2: Tensor = None):
        ...

    def forward(self, operand_1: Tensor, operand_2: Tensor = None):
        self.check_compatibility(operand_1, operand_2)
        return self._forward(operand_1, operand_2)

    @abstractmethod
    def backward_op_1(self, error_signal: Tensor, operand_1: Tensor, operand_2: Tensor = None) -> Tensor:
        ...

    def backward_op_2(self, error_signal: Tensor, operand_1: Tensor, operand_2: Tensor = None) -> Tensor:
        return None

class MatrixMatrixMul(Operation):

    def __init__(self):
        super().__init__()

    def check_compatibility(self, operand_1: Tensor, operand_2: Tensor):
        """
            Using numpy´s intrinsic check
        """
        return True

    def forward(self, operand_1: Tensor, operand_2: Tensor):
        try:
            # Use numpy broadcasting
            res = Tensor(np.matmul(operand_1.data, operand_2.data))
        except:  # TODO: update that
            raise ValueError("Incompatible shapes")

    @abstractmethod
    def backward_op_1(self, error_signal: np.array, operand_1: Tensor, operand_2: Tensor):
        return error_signal @ np.transpose(operand_2)

    @abstractmethod
    def backward_op_2(self, error_signal: np.array, operand_1: Tensor, operand_2: Tensor):
        return np.transose(operand_1.data) @ error_signal

class Sum(Operation):

    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis
        self.axis_length = 0

    def check_compatibility(self, operand_1: Tensor, operand_2: Tensor = None):
        if not len(operand_1.data.shape) > self.axis:
            raise ValueError("TODO")

    def _forward(self, operand_1: Tensor, operand_2: Tensor):
        self.axis_length = operand_1.shape[self.axis]
        return np.sum(operand_1.data, axis=self.axis)

    def backward_op_1(self, error_signal: Tensor, operand_1: Tensor, operand_2: Tensor = None) -> Tensor:
        tile_descriptor = np.ones((len(error_signal.shape)+1,), dtype=int)
        tile_descriptor[self.axis] = self.axis_length
        return np.tile(np.expand_dims(error_signal, self.axis), tile_descriptor)

class Mean(Operation):

    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis
        self.axis_length = None

    def check_compatibility(self, operand_1: Tensor, operand_2: Tensor = None):
        if not len(operand_1.data.shape) > self.axis:
            raise ValueError("TODO")

    def _forward(self, operand_1: Tensor, operand_2: Tensor):
        self.axis_length = operand_1.shape[self.axis]
        return np.mean(operand_1.data, axis=self.axis)

    def backward_op_1(self, error_signal: Tensor, operand_1: Tensor, operand_2: Tensor = None):
        tile_descriptor = np.ones((len(error_signal.shape) + 1,), dtype=int)
        tile_descriptor[self.axis] = self.axis_length  # that could be handled more elegantly
        return (1/self.axis_length) * np.tile(np.expand_dims(error_signal, self.axis), tile_descriptor)

class Transpose(Operation):

    def __init__(self, axis_permutation):
        super().__init__()
        # TODO: check for consistency in the axis permutation
        self.axis_permutation = axis_permutation
        self.inverse_permutation = [0] * len(axis_permutation)
        for i in range(len(axis_permutation)):
            self.inverse_permutation[axis_permutation[i]] = i

    def _forward(self, operand_1: Tensor, operand_2: Tensor = None):
        return np.transpose(operand_1.data, self.axis_permutation)

    def backward_op_1(self, error_signal: Tensor, operand_1: Tensor, operand_2: Tensor = None):
        return np.transpose(error_signal, self.inverse_permutation)

class TensorSum(Operation):

    def __init__(self):
        super().__init__()
        self.res_shape = None

    def _forward(self, operand_1: Tensor, operand_2: Tensor = None):
        res = operand_1.data + operand_2.data
        self.res_shape = res.shape

    def fuse_shapes(self, shape_1, shape_2):
        """
            Convention: shape_2 is at least as long as shape_1
        """
        if len(shape_1) < len(shape_2):
            shape = [0] * len(shape_2)
            for i in range(len(shape_1)):
                shape[i] = shape[i]
        else

    def backward_op_1(self, error_signal: Tensor, operand_1: Tensor, operand_2: Tensor = None) -> Tensor:
        """
            Attention: Broadcasting effects influence the gradient
        """
        if operand_1.shape != error_signal.shape:


        return error_signal



def sum(tensor: Tensor, axis: int):
    operation = Sum(axis)
    res_data = operation.forward(tensor)
    return Tensor(data=res_data,
                  computation_graph=ComputationGraph(operation=operation, tensor_1=tensor))

def mean(tensor: Tensor, axis: int):
    operation = Mean(axis)
    res_data = operation.forward(tensor)
    return Tensor(data=res_data,
                  computation_graph=ComputationGraph(operation=operation, tensor_1=tensor))

def transpose(tensor: Tensor, axis_permutation):
    operation = Transpose(axis_permutation)
    res_data = operation.forward(tensor)
    return Tensor(data=res_data,
                  computation_graph=ComputationGraph(operation=operation, tensor_1=tensor))