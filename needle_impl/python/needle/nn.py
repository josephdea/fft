"""The module."""
import math
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from functools import reduce
from operator import mul


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # input: (n, in_f)
        # output: (n, out_f)
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True))  # (in, out)
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype, requires_grad=True).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X_shape = X.shape

        reshaped = False
        if len(X_shape) == 1:
            X = ops.reshape(X, (1, X.shape[-1]))
            reshaped = True
        elif len(X_shape) != 2:
            X = ops.reshape(X, (reduce(mul, X_shape[:-1]), X_shape[-1]))
            reshaped = True

        # X = X.reshape(X_shape[:-1] + (1, X_shape[-1]))  # (..., 1, in)
        xW = ops.matmul(X, self.weight)  # (..., 1, in) @ (in, out) -> (..., 1, out)
        xW_shape = xW.shape
        if self.bias:
            xW = xW + ops.broadcast_to(self.bias, xW_shape)  # (..., 1, out)

        if reshaped:
            new_shape = list(X_shape)
            new_shape[-1] = self.out_features
            xW = ops.reshape(xW, tuple(new_shape))

        return xW  # .reshape(xW_shape[:-2] + (xW_shape[-1],))
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        new_shape = reduce(mul, X.shape[1:])
        return ops.reshape(X, (X.shape[0], new_shape))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.power_scalar(ops.add_scalar(ops.exp(-x), 1), -1)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        y_one_hot = init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
        op2 = ops.summation(logits * y_one_hot, axes=(1,))
        op1 = ops.logsumexp(logits, axes=(1,))
        ans = ops.summation(op1 - op2)
        return ans / op1.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))

        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x_shape = x.shape
        batches = x_shape[0]

        if self.training:
            mu_x = x.sum((0,)) / batches
            x_sub_mu_x = x - mu_x.reshape((1, self.dim)).broadcast_to(x_shape)
            var_x = (x_sub_mu_x ** 2).sum((0,)) / batches
            stdev_x = ((var_x + self.eps) ** 0.5).reshape((1, self.dim)).broadcast_to(x_shape)

            whitened = x_sub_mu_x / stdev_x

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu_x
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_x
            return (ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x_shape) * whitened) + ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x_shape) 

        whitened = (x - self.running_mean.broadcast_to(x_shape)) / ((self.running_var.broadcast_to(x_shape) + self.eps) ** 0.5)
        return self.weight * whitened + self.bias
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = ops.reshape(ops.transpose(ops.transpose(x, (1, 2)), (2, 3)), (s[0] * s[2] * s[3], s[1]))
        y = ops.reshape(super().forward(_x), (s[0], s[2], s[3], s[1]))
        return ops.transpose(ops.transpose(y, (2, 3)), (1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x.shape must be (batches, dim)
        x_shape = x.shape
        batches = x_shape[0]

        sum_x = ops.summation(x, axes=(1,))
        sum_x = ops.reshape(sum_x, (batches, 1)).broadcast_to(x_shape)
        mu_x = sum_x / self.dim

        x_sub_mu_x = x - mu_x
        var_x = (ops.summation(x_sub_mu_x ** 2, axes=(1,)) / self.dim) + self.eps
        stdev_x = ops.reshape(var_x ** 0.5, (batches, 1)).broadcast_to(x_shape)
        
        return (self.weight.broadcast_to(x_shape) * (x_sub_mu_x / stdev_x)) + self.bias.broadcast_to(x_shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        dropout = init.randb(*x.shape, p=1 - self.p, device=x.device) / (1 - self.p)
        return x * dropout
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(kernel_size * kernel_size * in_channels,
                                                     kernel_size * kernel_size * out_channels,
                                                     shape=(kernel_size, kernel_size, in_channels, out_channels),
                                                     device=device,
                                                     dtype=dtype,
                                                     requires_grad=True))
        hi = 1 / (in_channels * kernel_size ** 2) ** 0.5
        self.bias = Parameter(init.rand(out_channels, low=-hi, high=hi, device=device, dtype=dtype, requires_grad=True)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x: NCHW
        # we need NHWC
        x = ops.transpose(ops.transpose(x, (1, 2)), (2, 3))
        _, h, w, _ = x.shape
        assert h == w
        # padding = math.ceil(((self.stride - 1) * h + self.kernel_size - self.stride) / 2)
        # padding = ((self.stride - 1) * h + self.kernel_size) // 2
        padding = self.kernel_size // 2  # ?????????????????
        convolved = ops.conv(x, self.weight, stride=self.stride, padding=padding)  # NHWC
        if self.bias is not None:
            bias = ops.broadcast_to(ops.reshape(self.bias, (1, 1, 1, self.out_channels)), convolved.shape)
            convolved += bias
        convolved = ops.transpose(ops.transpose(convolved, (2, 3)), (1, 2))  # NCHW
        # if self.bias is not None:
        #     bias = ops.broadcast_to(ops.reshape(self.bias, (1, self.out_channels, 1, 1)), convolved.shape)
        #     convolved += bias
        return convolved
        ### END YOUR SOLUTION


class FFT1D(Module):
    def forward(self, x):
        return ops.forward_fourier_real_1d(x)


class IFFT1D(Module):
    def forward(self, x):
        return ops.backward_fourier_real_1d(x)


class FFT2D(Module):
    def forward(self, x):
        return ops.forward_fourier_real_2d(x)


class IFFT2D(Module):
    def forward(self, x):
        return ops.backward_fourier_real_2d(x)


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        hi = math.sqrt(1 / hidden_size)
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-hi, high=hi, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-hi, high=hi, device=device, dtype=dtype, requires_grad=True))
        self._bias = bias
        self._device = device
        self.bias_ih = Parameter(init.rand(hidden_size, low=-hi, high=hi, device=device, dtype=dtype, requires_grad=True)) if bias else None
        self.bias_hh = Parameter(init.rand(hidden_size, low=-hi, high=hi, device=device, dtype=dtype, requires_grad=True)) if bias else None
        self.activation = ReLU() if nonlinearity == 'relu' else Tanh()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(X.shape[0], self.W_ih.shape[1], device=self._device)
        result = X @ self.W_ih + h @ self.W_hh
        if self._bias:
            result += ops.broadcast_to(ops.reshape(self.bias_ih + self.bias_hh, (1, self.W_ih.shape[1])), (X.shape[0], self.W_ih.shape[1]))
        return self.activation(result)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._device = device
        self.rnn_cells = ([RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)] +
                          [RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype) for _ in range(num_layers - 1)])
        self._num_layers = num_layers
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h0 is None:
            hs = (init.zeros(X.shape[1], self._hidden_size, device=self._device),) * self._num_layers
        else:
            hs = ops.split(h0, 0)
        
        xs = ops.split(X, 0)
        
        output = []
        for x in xs:
            h_n = []
            for h, rnn_cell in zip(hs, self.rnn_cells):
                x = rnn_cell(x, h)
                h_n.append(x)
            output.append(h_n[-1])
            hs = h_n
        return ops.stack(output, 0), ops.stack(h_n, 0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        hi = math.sqrt(1 / hidden_size)
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-hi, high=hi, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-hi, high=hi, device=device, dtype=dtype, requires_grad=True))
        self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-hi, high=hi, device=device, dtype=dtype, requires_grad=True)) if bias else None
        self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-hi, high=hi, device=device, dtype=dtype, requires_grad=True)) if bias else None
        self._bias = bias
        self._device = device
        self._hidden_size = hidden_size
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = (None, None)
        h0, c0 = h
        if h0 is None:
            h0 = init.zeros(X.shape[0], self._hidden_size, device=self._device)
        if c0 is None:
            c0 = init.zeros(X.shape[0], self._hidden_size, device=self._device)
            
        p = X @ self.W_ih + h0 @ self.W_hh
        if self._bias:
            p += ops.broadcast_to(ops.reshape(self.bias_ih + self.bias_hh, (1, 4 * self._hidden_size)), (X.shape[0], 4 * self._hidden_size))
        p = ops.split(p, 1).tuple()
        i, f, g, o = (ops.stack(p[:self._hidden_size], 1),
                      ops.stack(p[self._hidden_size:self._hidden_size * 2], 1),
                      ops.stack(p[self._hidden_size * 2:self._hidden_size * 3], 1),
                      ops.stack(p[self._hidden_size * 3:], 1))

        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = self.tanh(g)
        o = self.sigmoid(o)
        
        c = c0 * f + i * g
        h = self.tanh(c) * o
        return h, c
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._device = device
        self.lstm_cells = ([LSTMCell(input_size, hidden_size, bias, device, dtype)] +
                           [LSTMCell(hidden_size, hidden_size, bias, device, dtype) for _ in range(num_layers - 1)])
        self._num_layers = num_layers
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            hs = ((init.zeros(X.shape[1], self._hidden_size, device=self._device),) * self._num_layers,
                  (init.zeros(X.shape[1], self._hidden_size, device=self._device),) * self._num_layers)
        else:
            hs = (ops.split(h[0], 0), ops.split(h[1], 0))
        
        xs = ops.split(X, 0)
        
        output = []
        for x in xs:
            h_n = ([], [])
            for h, c, lstm_cell in zip(hs[0], hs[1], self.lstm_cells):
                x, c = lstm_cell(x, (h, c))
                h_n[0].append(x)
                h_n[1].append(c)
            output.append(x)
            hs = h_n
        return ops.stack(output, 0), (ops.stack(h_n[0], 0), ops.stack(h_n[1], 0))
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self._device = device
        self._dtype = dtype
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        xs = ops.split(x, 0)
        ts = tuple(init.one_hot(self.num_embeddings, t, device=self._device, dtype=self._dtype) @ self.weight for t in xs)
        result = ops.stack(ts, 0)
        return result
        ### END YOUR SOLUTION
