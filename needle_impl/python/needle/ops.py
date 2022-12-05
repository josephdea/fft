"""Operator table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy
from .backend_ndarray.ndarray import prod 

from .backend_selection import array_api, NDArray


def unbroadcast(a, target_shape):
    now_shape = list(a.shape)
    
    # first get rid of the extra dimensions
    while len(now_shape) > len(target_shape):
        a = a.sum((0,))
        now_shape = now_shape[1:]
    
    # now unbroadcast
    for i in range(len(now_shape)):
        n = now_shape[i]
        t = target_shape[i]
        
        if n == t:
            continue
        elif n > t:
            a = summation(a, (i,))  # .sum((i,)).compact()
            now_shape[i] = 1
            a = reshape(a, now_shape)  # a.reshape(now_shape).compact()
        elif n < t:
            now_shape[i] = t
            a = broadcast_to(a, now_shape)  # a.broadcast_to(now_shape).compact()
    
    return a


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1],


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad,


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * power_scalar(node.inputs[0], self.scalar - 1),
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * (lhs / rhs) / rhs
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar,
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if axes is None:
            self.axes = (-1, -2)
        else:
            self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.swapaxes(a, *self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes),  # out_grad.transpose(self.axes),
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape),  # out_grad.reshape(node.inputs[0].shape),
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        old_shape = node.inputs[0].shape
        out_grad = unbroadcast(out_grad, old_shape)
        return out_grad,
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        new_shape = list(a.shape)
        if self.axes is None:
            for i in range(len(new_shape)):
                new_shape[i] = 1
        else:
            if isinstance(self.axes, int):
                new_shape[self.axes] = 1
            else:
                for axis in self.axes:
                    new_shape[axis] = 1
        new_shape = tuple(new_shape)
        out_grad = broadcast_to(reshape(out_grad, new_shape), a.shape)  # .reshape(new_shape).broadcast_to(a.shape)
        return out_grad,
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        ans = (a.compact() @ b.compact()).compact()
        return ans
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        a_shape, b_shape = a.shape, b.shape
        a_grad = matmul(out_grad, transpose(b))
        b_grad = matmul(transpose(a), out_grad)
        # return unbroadcast(a_grad, a.shape), unbroadcast(b_grad, b.shape)
        return a_grad, b_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a * (-1)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (-1),
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0],
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0]),
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * Tensor(node.realize_cached_data() > 0, device=out_grad.device),
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        if self.axes is not None:
            broadcast_shape = list(Z.shape)
            for x in self.axes:
                broadcast_shape[x] = 1
        else:
            broadcast_shape = (1,) * len(Z.shape)
        max_z = array_api.max(Z, axis=self.axes, keepdims=True).broadcast_to(broadcast_shape)
        sums = array_api.sum(array_api.exp(Z - max_z.broadcast_to(Z.shape)), axis=self.axes, keepdims=True).broadcast_to(broadcast_shape)
        logged = array_api.log(sums)
        ans = logged + max_z
        return array_api.squeeze(ans, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        broadcast_shape = [1] * len(Z.shape)
        if self.axes is not None:
            r_ptr = 0
            
            for w_ptr in range(len(Z.shape)):
                if w_ptr in self.axes:
                    continue
                broadcast_shape[w_ptr] = node.shape[r_ptr]
                r_ptr += 1
            
        broadcastable_node = reshape(node, broadcast_shape)
        Z = exp(Z - broadcast_to(broadcastable_node, Z.shape))
        sum_Z = summation(Z, self.axes)
        broadcastable_sum_Z = reshape(sum_Z, broadcast_shape)
        softmax = Z / broadcast_to(broadcastable_sum_Z, Z.shape)
        return broadcast_to(reshape(out_grad, broadcast_shape), Z.shape) * softmax,
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (- (node * node) + 1),
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        stacked = array_api.empty([len(args)] + list(args[0].shape),
                                  dtype=args[0].dtype, device=args[0].device)
        axis = self.axis
        
        for i, arg in enumerate(args):
            stacked[i, :] = arg[:]
        
        if axis != 0:
            # permute
            permute_dims = list(range(1, axis + 1)) + [0] + list(range(axis + 1, len(stacked.shape)))
            stacked = stacked.permute(permute_dims)
        
        return stacked
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis),
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        # if axis is 0, just split
        # otherwise, transpose and split.
        
        if self.axis != 0:
            dims = list(range(len(A.shape)))
            axes = [self.axis] + dims[:self.axis] + dims[self.axis + 1:]
            A = A.permute(tuple(axes))
        
        new_shape = A.shape[1:]
        if not new_shape:
            new_shape = A.shape
        return tuple(A[i].compact().reshape(new_shape) for i in range(A.shape[0]))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis),
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes),
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int, fill: int = 0):
        self.axes = axes
        self.dilation = dilation
        self.fill = fill

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a
        
        slices = [slice(None, None, 1) for _ in range(len(a.shape))]
        new_shape = list(a.shape)
        
        for axis in self.axes:
            if axis < len(slices):
                slices[axis] = slice(None, None, self.dilation + 1)
                new_shape[axis] *= self.dilation + 1
        
        result = array_api.full(new_shape, self.fill, a.dtype, a.device)
        result[tuple(slices)] = a
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation),
        ### END YOUR SOLUTION


def dilate(a, axes, dilation, fill=0):
    return Dilate(axes, dilation, fill)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a
        
        slices = [slice(None, None, 1) for _ in range(len(a.shape))]
        for axis in self.axes:
            if axis < len(slices):
                slices[axis] = slice(None, None, self.dilation + 1)
        return a[tuple(slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation, 1),
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, Z, weight):
        ### BEGIN YOUR SOLUTION
        Z = Z.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = Z.shape
        K, _, _, C_out = weight.shape
        Ns, Hs, Ws, Cs = Z.strides
        
        inner_dim = K * K * C_in
        A = Z.compact().as_strided((N, 1 + (H - K) // self.stride, 1 + (W - K) // self.stride, K, K, C_in),
                                   (Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)).compact()
        A = A.reshape((prod(A.shape) // inner_dim, inner_dim)).compact()
        out = A @ weight.compact().reshape((prod(weight.shape) // C_out, C_out)).compact()
        return out.reshape((N, 1 + (H - K) // self.stride, 1 + (W - K) // self.stride, C_out)).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z, weight = node.inputs
        dZ = conv(dilate(out_grad, (1, 2), self.stride - 1), transpose(flip(weight, (0, 1)), (2, 3)), padding=weight.shape[0] - self.padding - 1)
        dw = transpose(transpose(conv(transpose(Z, (0, 3)), transpose(transpose(dilate(out_grad, (1, 2), self.stride - 1), (0, 1)), (1, 2)), padding=self.padding), (0, 1)), (1, 2))
        return dZ, dw
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


class ForwardFourierReal1D(TensorTupleOp):
    def compute(self, real):
        return array_api.forward_fourier_1d(real)

    def gradient(self, out_grad, node):
        out_grad_real, out_grad_imag = tuple_get_item(out_grad, 0), tuple_get_item(out_grad, 1)
        return backward_fourier_complex_1d(out_grad_real, out_grad_imag)


def forward_fourier_real_1d(real):
    return ForwardFourierReal1D()(real)


class ForwardFourierComplex1D(TensorTupleOp):
    def compute(self, real, imag):
        return array_api.forward_fourier_1d(real, imag)

    def gradient(self, out_grad, node):
        out_grad_real, out_grad_imag = tuple_get_item(out_grad, 0), tuple_get_item(out_grad, 1)
        return backward_fourier_complex_1d(out_grad_real, out_grad_imag)


def forward_fourier_complex_1d(real, imag):
    return ForwardFourierComplex1D()(real, imag)


class BackwardFourierReal1D(TensorTupleOp):
    def compute(self, real):
        return array_api.backward_fourier_1d(real)

    def gradient(self, out_grad, node):
        out_grad_real, out_grad_imag = tuple_get_item(out_grad, 0), tuple_get_item(out_grad, 1)
        return forward_fourier_complex_1d(out_grad_real, out_grad_imag)


def backward_fourier_real_1d(real):
    return BackwardFourierReal1D()(real)


class BackwardFourierComplex1D(TensorTupleOp):
    def compute(self, real, imag):
        return array_api.backward_fourier_1d(real, imag)

    def gradient(self, out_grad, node):
        out_grad_real, out_grad_imag = tuple_get_item(out_grad, 0), tuple_get_item(out_grad, 1)
        return forward_fourier_complex_1d(out_grad_real, out_grad_imag)


def backward_fourier_complex_1d(real, imag):
    return BackwardFourierComplex1D()(real, imag)
