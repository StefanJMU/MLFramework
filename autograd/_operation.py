import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Union, Literal
from scipy.signal import convolve2d

from numpy.lib.stride_tricks import sliding_window_view

class Operation(metaclass=ABCMeta):

    def __init__(self, name: str=None, *args, **kwargs):
        if name is None:
            self.name = type(self).__name__
        else:
            self.name = name

    def check_compatibility(self, operand_1, operand_2=None):
        return True

    @abstractmethod
    def _forward(self, operand_1, operand_2=None):
        ...

    @abstractmethod
    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        ...

    def forward(self, operand_1, operand_2=None):
        self.check_compatibility(operand_1, operand_2)
        return self._forward(operand_1, operand_2)

    def backward_op_2(self, error_signal, operand_1, operand_2=None):
        return None

    def debroadcast(self, error_signal, shape):
        """
            function debroacasts error_signal to shape
        """
        dims = len(error_signal.shape)
        for i in range(dims - 1, -1, -1):
            if i >= len(shape):
                # Contract the dimension completely
                error_signal = np.sum(error_signal, axis=i)
            else:
                # Contract to previous length in the dimension (detiling)
                n_blocs = error_signal.shape[i] // shape[i]
                error_signal = np.reshape(error_signal, newshape=error_signal.shape[:i]
                                                                 + (n_blocs, -1)
                                                                 + error_signal.shape[i+1:])
                error_signal = np.sum(error_signal, axis=i)
        return error_signal
    
class MatrixMatrixMul(Operation):

    def __init__(self):
        super().__init__()

    def check_compatibility(self, operand_1, operand_2):
        """
            Using numpy´s intrinsic check
        """
        return True

    def forward(self, operand_1, operand_2):
        # todo
        ...

    def backward_op_1(self, error_signal: np.array, operand_1, operand_2):
        return error_signal @ np.transpose(operand_2)

    def backward_op_2(self, error_signal: np.array, operand_1, operand_2):
        return np.transose(operand_1.data) @ error_signal

class Sum(Operation):

    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def check_compatibility(self, operand_1, operand_2=None):
        if not len(operand_1.data.shape) > self.axis:
            raise ValueError("TODO")

    def _forward(self, operand_1, operand_2):
        return np.sum(operand_1.data, axis=self.axis)

    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        expansion_mask = np.ones_like(operand_1.data)
        return np.expand_dims(error_signal, self.axis) * expansion_mask

class Mean(Operation):

    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def check_compatibility(self, operand_1, operand_2=None):
        if not len(operand_1.data.shape) > self.axis:
            raise ValueError("TODO")

    def _forward(self, operand_1, operand_2):
        return np.mean(operand_1.data, axis=self.axis)

    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        expansion_mask = (1/operand_1.shape[self.axis]) * np.ones_like(operand_1.data)
        return np.expand_dims(error_signal, self.axis) * expansion_mask

class Prod(Operation):

    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def _forward(self, operand_1, operand_2=None):
        return np.prod(operand_1.data, axis=self.axis)

    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        axis_product = np.prod(operand_1.data, axis=self.axis, keepdims=True)
        prop_factor = axis_product * np.reciprocal(operand_1.data)
        return np.expand_dims(error_signal, axis=self.axis) * prop_factor

class Square(Operation):

    def __init__(self):
        super().__init__()

    def _forward(self, operand_1, operand_2=None):
        return np.square(operand_1.data)

    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        return error_signal * 2 * operand_1.data

class Power(Operation):

    def __init__(self):
        super().__init__()

    def _forward(self, operand_1, operand_2=None):
        return np.power(operand_1.data, operand_2.data)

    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        gradient = error_signal * operand_2.data * np.power(operand_1.data, operand_2.data - 1)
        return self.debroadcast(gradient, operand_1.shape)

    def backward_op_2(self, error_signal, operand_1, operand_2=None):
        gradient = np.exp(np.log(operand_1.data) * operand_2.data) * np.log(operand_1.data)
        return self.debroadcast(gradient, operand_2.shape)

class Sqrt(Operation):

    def __init__(self):
        super().__init__()

    def _forward(self, operand_1, operand_2=None):
        return np.sqrt(operand_1.data)

    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        return error_signal * 0.5 * np.power(operand_1.data, -0.5)

class Root(Operation):

    def __init__(self):
        super().__init__()

    def _forward(self, operand_1, operand_2=None):
        return np.power(operand_1.data, np.reciprocal(operand_2.data))

    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        op_2_reci = np.reciprocal(operand_2.data)
        gradient = op_2_reci * np.power(operand_1.data, op_2_reci - 1)
        return self.debroadcast(gradient, operand_1.shape)

    def backward_op_2(self, error_signal, operand_1, operand_2=None):
        gradient = -1 * np.exp(np.log(operand_1.data) * np.reciprocal(operand_2.data)) \
                   * np.log(operand_1.data) * np.reciprocal(np.square(operand_2.data))
        return self.debroadcast(gradient, operand_2.shape)

class Transpose(Operation):

    def __init__(self, axis_permutation):
        super().__init__()
        # TODO: check for consistency in the axis permutation
        self.axis_permutation = axis_permutation
        self.inverse_permutation = [0] * len(axis_permutation)
        for i in range(len(axis_permutation)):
            self.inverse_permutation[axis_permutation[i]] = i

    def _forward(self, operand_1, operand_2=None):
        return np.transpose(operand_1.data, self.axis_permutation)

    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        return np.transpose(error_signal, self.inverse_permutation)

class TensorSum(Operation):

    def __init__(self):
        super().__init__()

    def _forward(self, operand_1, operand_2=None):
        return operand_1.data + operand_2.data

    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        if operand_1.shape != error_signal.shape:
            return self.debroadcast(error_signal, operand_1.shape)
        return error_signal

    def backward_op_2(self, error_signal, operand_1, operand_2=None):
        if operand_2.shape != error_signal.shape:
            return self.debroadcast(error_signal, operand_2.shape)
        return error_signal

class TensorProduct(Operation):

    def __init__(self):
        super().__init__()

    def _forward(self, operand_1, operand_2=None):
        return operand_1.data * operand_2.data

    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        op2_data = np.broadcast_to(operand_2.data, error_signal.shape)
        grad_factor = op2_data * error_signal
        return self.debroadcast(grad_factor, operand_1.shape)

    def backward_op_2(self, error_signal, operand_1, operand_2=None):
        op1_data = np.broadcast_to(operand_1.data, error_signal.shape)
        grad_factor = op1_data * error_signal
        return self.debroadcast(grad_factor, operand_2.shape)

class Reshape(Operation):

    def __init__(self, new_shape):
        super().__init__()
        self.new_shape = new_shape

    def _forward(self, operand_1, operand_2=None):
        return np.reshape(operand_1.data, self.new_shape)

    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        return np.reshape(error_signal, operand_1.shape)

class Slice(Operation):

    def __init__(self, key):
        super().__init__()
        self.key = key

    def _forward(self, operand_1, operand_2=None):
        return operand_1.data[self.key]

    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        gradient = np.zeros(operand_1.shape)
        gradient[self.key] = error_signal
        return gradient

class TensorDiv(Operation):

    def __init__(self):
        super().__init__()

    def _forward(self, operand_1, operand_2=None):
        return operand_1.data / operand_2.data

    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        op_2_data = np.broadcast_to(np.reciprocal(operand_2.data), error_signal.shape)
        gradient = error_signal * op_2_data
        return self.debroadcast(gradient, operand_1.shape)

    def backward_op_2(self, error_signal, operand_1, operand_2=None):
        op_1_data = np.broadcast_to(operand_1.data, error_signal.shape)
        grad_factor = error_signal * op_1_data
        gradient = -grad_factor * np.reciprocal(np.square(operand_2.data))
        return self.debroadcast(gradient, operand_2.shape)

class Concatenate(Operation):

    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def _forward(self, tensor_list, operand_2=None):
        # operand_1 is a list of Tensors
        data_list = [tensor.data for tensor in tensor_list]
        concatenated = np.concatenate(data_list, axis=self.axis)
        return concatenated

    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        # operand_1 of type TensorList
        sizes = np.array([operand.shape[self.axis] for operand in operand_1.tensors][:-1])
        split_points = np.cumsum(sizes)
        return np.split(error_signal, split_points, axis=self.axis)

class Mix(Operation):

    def __init__(self, mask: np.array):
        super().__init__()
        self.mask = mask

    def _forward(self, operand_1, operand_2=None):
        if operand_1.shape != self.mask.shape:
            raise ValueError('Mix requires the mask to match the dimensions of operand_1.'
                             f'Expected {self.mask.shape}. Got {operand_1.shape}')
        # TODO: a check for the mask and operand_2 to be conducted for a more meaningful error message
        mixed_data = np.copy(operand_1.data)
        self.mix_positions = np.nonzero(self.mask)
        mixed_data[self.mix_positions] = np.reshape(operand_2.data, newshape=(-1))
        return mixed_data

    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        gradient = np.copy(error_signal)
        gradient[self.mix_positions] = 0
        return gradient

    def backward_op_2(self, error_signal, operand_1, operand_2=None):
        return error_signal[self.mix_positions]

class Softmax(Operation):

    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def _forward(self, operand_1, operand_2=None):
        op_exp = np.exp(operand_1.data)
        aggregated = np.sum(op_exp, axis=self.axis, keepdims=True)
        return op_exp / aggregated

    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        # could theoretically be used from the forward operation
        op_exp = np.exp(operand_1.data)
        aggregated = np.sum(op_exp, axis=self.axis, keepdims=True)
        softmax = op_exp / aggregated
        return error_signal * (softmax * (1 - softmax))

