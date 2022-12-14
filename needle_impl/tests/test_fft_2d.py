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


@pytest.mark.parametrize("shape", FFT_SHAPE_PARAMETERS_2D)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fft_2d_forward_real(shape, device):
    np.random.seed(1)

    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    
    A_t = torch.Tensor(_A)
    
    out_real, out_complex = ndl.ops.forward_fourier_real_2d(A)
    out = out_real.numpy() + 1j * out_complex.numpy()
    
    out_t = torch.fft.fft2(A_t).resolve_conj().numpy()
    
    np.testing.assert_allclose(out_t, out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", FFT_SHAPE_PARAMETERS_2D)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fft_2d_forward_complex(shape, device):
    np.random.seed(1)
    
    _A = np.random.randn(*shape).astype(np.float32)  # real
    _B = np.random.randn(*shape).astype(np.float32)  # complex
    
    A = ndl.Tensor(nd.array(_A), device=device)
    B = ndl.Tensor(nd.array(_B), device=device)
    
    A_t = torch.Tensor(_A)
    B_t = torch.Tensor(_B)
    C_t = A_t + 1j * B_t  # complex
    
    out_real, out_complex = ndl.ops.forward_fourier_complex_2d(A, B)
    out = out_real.numpy() + 1j * out_complex.numpy()

    out_t = torch.fft.fft2(C_t).resolve_conj().numpy()
    
    np.testing.assert_allclose(out_t, out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", FFT_SHAPE_PARAMETERS_2D)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fft_2d_backward_real(shape, device):
    np.random.seed(1)

    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    
    A_t = torch.Tensor(_A)
    
    out_real, out_complex = ndl.ops.backward_fourier_real_2d(A)
    out = out_real.numpy() + 1j * out_complex.numpy()
    
    out_t = torch.fft.ifft2(A_t).resolve_conj().numpy()
    
    np.testing.assert_allclose(out_t, out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", FFT_SHAPE_PARAMETERS_2D)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fft_2d_backward_complex(shape, device):
    np.random.seed(1)
    
    _A = np.random.randn(*shape).astype(np.float32)  # real
    _B = np.random.randn(*shape).astype(np.float32)  # complex
    
    A = ndl.Tensor(nd.array(_A), device=device)
    B = ndl.Tensor(nd.array(_B), device=device)
    
    A_t = torch.Tensor(_A)
    B_t = torch.Tensor(_B)
    C_t = A_t + 1j * B_t  # complex
    
    out_real, out_complex = ndl.ops.backward_fourier_complex_2d(A, B)
    out = out_real.numpy() + 1j * out_complex.numpy()

    out_t = torch.fft.ifft2(C_t).resolve_conj().numpy()
    
    np.testing.assert_allclose(out_t, out, atol=1e-5, rtol=1e-5)
