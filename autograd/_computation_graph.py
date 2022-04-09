import numpy as np
from ._tensor import Tensor

class ComputationGraph:

    def __init__(self, operation, tensor_1: Tensor, tensor_2: Tensor = None):
        self.operation = operation
        self.tensor_1 = tensor_1  # ref
        self.tensor_2 = tensor_2  # ref
        # store earmark of the versions of the tensors required for backpropagation
        self.required_earmark_1 = tensor_1.earmark
        self.required_earmark_2 = None if tensor_2 is None else tensor_2.earmark

    def backward(self, error_signal: Tensor):
        # Assumption: if any tensor in the computation history of the tensor requires a grad, so will the tensor itself
        if (self.tensor_1.earmark != self.required_earmark_1
                or (self.tensor_2 is not None and self.tensor_2.earmark != self.required_earmark_2)):
           raise AttributeError("Tensors have been modified since computation graph establishment.")

        if self.tensor_1.requires_grad:
            error_operand_1 = self.operation.backward_op_1(error_signal, self.tensor_1, self.tensor_2)
            self.tensor_1.backward(error_operand_1)
        if self.tensor_2 is not None and self.tensor_2.requires_grad:
            error_operand_2 = self.operation.backward_op_2(error_signal, self.tensor_1, self.tensor_2)
            self.tensor_2.backward(error_operand_2)

