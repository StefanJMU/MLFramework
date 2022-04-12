import numpy as np
from abc import ABCMeta, abstractmethod

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

    def forward(self, operand_1, operand_2=None):
        self.check_compatibility(operand_1, operand_2)
        return self._forward(operand_1, operand_2)

    @abstractmethod
    def backward_op_1(self, error_signal, operand_1, operand_2=None):
        ...

    def backward_op_2(self, error_signal, operand_1, operand_2=None):
        return None

    def debroadcast(self, error_signal, shape):
        """
            Convention: shape is broadcast to shape of error_signal (possibly by the identity)
            TODO: this function has to be made more elegant and fast
        """
        dims = len(error_signal.shape)
        for i in range(dims - 1, -1, -1):
            if i >= len(shape):
                # Contract the dimension completely
                error_signal = np.sum(error_signal, axis=i)
            else:
                # Contract to previous length in the dimension (detiling)
                n_blocs = error_signal.shape[i] / shape[i]
                blocks = np.split(error_signal, n_blocs, axis=i)
                error_signal = blocks[0]
                for j in range(1, len(blocks)):
                    error_signal = error_signal + blocks[j]
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

    @abstractmethod
    def backward_op_1(self, error_signal: np.array, operand_1, operand_2):
        return error_signal @ np.transpose(operand_2)

    @abstractmethod
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

