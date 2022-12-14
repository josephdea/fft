import sys
sys.path.append('./python')
import itertools
import numpy as np
import pytest
import torch

import needle as ndl
from needle import backend_ndarray as nd


_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]
FFT_SHAPE_PARAMETERS_1D = [1, 2, 3, 4, 5, 7, 8, 15, 16, 25, 32]
FFT_SHAPE_PARAMETERS_2D = list(itertools.product(FFT_SHAPE_PARAMETERS_1D, repeat=2))


@pytest.mark.parametrize("shape", FFT_SHAPE_PARAMETERS_1D)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fft_1d_grad(shape, device):
    np.random.seed(1)

    _A = np.random.randn(shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    
    A_t = torch.Tensor(_A)
    A_t.requires_grad = True
    
    out_real, out_imag = ndl.ops.forward_fourier_real_1d(A)
    sums = ndl.ops.summation(out_real) + ndl.ops.summation(out_imag)
    sums.backward()
    grad = A.grad.numpy()[0]  # take only the real part

    out_t = torch.fft.fft(A_t).resolve_conj()
    out_real_t = torch.real(out_t)
    out_imag_t = torch.imag(out_t)
    sums_t = torch.sum(out_real_t) + torch.sum(out_imag_t)
    sums_t.backward()
    grad_t = A_t.grad.numpy()

    np.testing.assert_allclose(grad_t, grad, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", FFT_SHAPE_PARAMETERS_2D)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fft_2d_grad(shape, device):
    np.random.seed(1)

    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    
    A_t = torch.Tensor(_A)
    A_t.requires_grad = True
    
    out_real, out_imag = ndl.ops.forward_fourier_real_2d(A)
    sums = ndl.ops.summation(out_real) + ndl.ops.summation(out_imag)
    sums.backward()
    grad = A.grad.numpy()[0]  # take only the real part

    out_t = torch.fft.fft2(A_t).resolve_conj()
    out_real_t = torch.real(out_t)
    out_imag_t = torch.imag(out_t)
    sums_t = torch.sum(out_real_t) + torch.sum(out_imag_t)
    sums_t.backward()
    grad_t = A_t.grad.numpy()

    np.testing.assert_allclose(grad_t, grad, atol=1e-4, rtol=1e-4)
