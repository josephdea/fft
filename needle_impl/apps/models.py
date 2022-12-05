import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):
        super().__init__()
        self.conv_layer = nn.Conv(in_channels, out_channels, kernel_size, stride, device=device, dtype=dtype)
        self.batch_norm = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.batch_norm(self.conv_layer(x)))


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.convbn1 = ConvBN(3, 16, 7, 4, device=device, dtype=dtype)
        self.convbn2 = ConvBN(16, 32, 3, 2, device=device, dtype=dtype)
        self.convbn3 = ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
        self.convbn4 = ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
        self.convbn5 = ConvBN(32, 64, 3, 2, device=device, dtype=dtype)
        self.convbn6 = ConvBN(64, 128, 3, 2, device=device, dtype=dtype)
        self.convbn7 = ConvBN(128, 128, 3, 1, device=device, dtype=dtype)
        self.convbn8 = ConvBN(128, 128, 3, 1, device=device, dtype=dtype)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        y1 = self.convbn1(x)
        y2 = self.convbn2(y1)
        y3 = self.convbn3(y2)
        y4 = self.convbn4(y3)
        y5 = self.convbn5(y2 + y4)
        y6 = self.convbn6(y5)
        y7 = self.convbn7(y6)
        y8 = self.convbn8(y7)
        y9 = self.flatten(y6 + y8)
        y10 = self.linear1(y9)
        y11 = self.relu(y10)
        y12 = self.linear2(y11)
        return y12
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        else:
            raise ValueError('Unsupported sequence model')
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        w = self.embedding(x)
        out, h = self.seq_model(w, h)
        seq_len, bs, hidden_size = out.shape
        out = ndl.ops.reshape(out, (seq_len * bs, hidden_size))
        out = self.linear(out)
        return out, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
