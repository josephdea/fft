import numpy as np
import torch
import needle as ndl


if __name__ == '__main__':
    N = 1
    C_in = 1
    H = 5
    W = 7

    K = 4
    C_out = 1

    _X = np.random.randn(N, H, W, C_in)
    _w = np.random.randn(K, K, C_in, C_out)

    X = ndl.Tensor(_X, device=ndl.cpu())
    w = ndl.Tensor(_w, device=ndl.cpu())

    Xt = torch.Tensor(_X).permute((0, 3, 1, 2)).float()
    wt = torch.Tensor(_w).permute((3, 2, 0, 1)).float()

    padding = 3
    stride = 3

    ct = torch.nn.functional.conv2d(Xt, wt, padding=padding, stride=stride).permute((0, 2, 3, 1))
    c = ndl.ops.conv2d(X, w, padding=padding, stride=stride)
    cf = ndl.ops.conv2d(X, w, method='fft', padding=padding, stride=stride)

    print(c)
    print(ct)
    print(cf)
