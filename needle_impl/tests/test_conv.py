import sys
sys.path.append('./python')
import numpy as np
import pytest
import torch

import needle as ndl
from needle import backend_ndarray as nd


_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]
CONV_SHAPES = [
    ( (3, 14, 14, 8), (3, 3, 8, 16), 1, 0 ),
    ( (3, 14, 14, 8), (3, 3, 8, 16), 1, 1 ),
    ( (3, 16, 16, 8), (3, 3, 8, 16), 1, 2 ),
    ( (3, 16, 16, 8), (3, 3, 8, 14), 1, 0 ),
    ( (3, 16, 16, 2), (3, 3, 2, 14), 1, 0 ),

    ( (3, 14, 14, 8), (3, 3, 8, 16), 2, 0 ),
    ( (3, 14, 14, 8), (3, 3, 8, 16), 2, 1 ),
    ( (3, 16, 16, 8), (3, 3, 8, 16), 2, 2 ),
    ( (3, 16, 16, 8), (3, 3, 8, 14), 2, 0 ),
    ( (3, 16, 16, 2), (3, 3, 2, 14), 2, 0 ),

    ( (3, 16, 16, 24), (3, 3, 24, 14), 1, 0 ),
    ( (3, 14, 14, 8), (5, 5, 8, 16),   1, 0 ),
    ( (3, 17, 17, 8), (5, 5, 8, 16),   1, 0 ),
    ( (3, 17, 17, 1), (5, 5, 1, 16) ,  1, 0),
    ( (3, 17, 17, 16), (5, 5, 16, 1),  1, 0 ),
    ( (3, 17, 17, 16), (1, 1, 16, 1),  1, 0 ),
    ( (1, 14, 14, 2), (3, 3, 2, 2),    1, 0 ),
]  # taken directly from the conv test in HW4


@pytest.mark.parametrize("Z_shape, W_shape, stride, padding", CONV_SHAPES)
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("backward", [True, False], ids=["backward", "forward"])
def test_conv(Z_shape, W_shape, stride, padding, backward, device):
    np.random.seed(1)

    _Z = np.random.randn(*Z_shape) * 5
    _Z = _Z.astype(np.float32)
    Z = ndl.Tensor(_Z, device=device)
    
    _W = np.random.randn(*W_shape) * 5
    _W = _W.astype(np.float32)
    W = ndl.Tensor(_W, device=device)

    y = ndl.ops.conv(Z, W, padding=padding, stride=stride, method='fft')
    y2 = y.sum()
    if backward:
        y2.backward()

    Ztch = torch.Tensor(_Z).float()
    Ztch.requires_grad=True

    Wtch = torch.Tensor(_W).float()
    Wtch.requires_grad=True

    out = torch.nn.functional.conv2d(Ztch.permute(0, 3, 1, 2), Wtch.permute(3, 2, 0, 1), padding=padding, stride=stride)
    out2 = out.sum()
    if backward:
        out2.backward()

    if backward:
        err1 = np.linalg.norm(Ztch.grad.numpy() - Z.grad.numpy())
        err2 = np.linalg.norm(Wtch.grad.numpy() - W.grad.numpy())
    err3 = np.linalg.norm(out2.detach().numpy() - y2.numpy())

    if backward:
        assert err1 < 1e-2, "input grads match"
        assert err2 < 1e-2, "weight grads match"
    assert err3 < 3e-1, "outputs match %s, %s" % (y2, out2)
