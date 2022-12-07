import numpy as np
import torch
import needle as ndl
import time


class MyModel(ndl.nn.Module):
    def __init__(self):
        super().__init__()
        self.my_fft = ndl.nn.FFT2D()

    def forward(self, x):
        fftr, ffti = self.my_fft(x)
        print('my fft', fftr, ffti)
        sums = ndl.ops.summation(fftr)
        # print('my sum', sums)
        return sums


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        fft = torch.fft.fft2(x)
        # fft = torch.real(fft)
        print('torch fft', torch.real(fft), torch.imag(fft))
        sums = torch.sum(fft)
        # print('torch sums', sums)
        return sums


if __name__ == '__main__':
    np.random.seed(0)

    # _batch_x = np.random.randn(3, 7, 5, 5).astype(np.float32)
    # _batch_y = np.array([0, 2, 1], dtype=np.float32)
    _batch_x = np.arange(3 * 5).reshape(3, 5) + 1 # np.random.randn(3, 7).astype(np.float32)

    _batch_x_real = np.random.randn(3, 4)
    _batch_x_imag = np.random.randn(3, 4)

    _batch_y = np.array([0.5], dtype=np.float32)

    batch_x = ndl.Tensor(_batch_x, device=ndl.cpu())
    batch_x_real = ndl.Tensor(_batch_x_real, device=ndl.cpu())
    batch_x_imag = ndl.Tensor(_batch_x_imag, device=ndl.cpu())
    batch_y = ndl.Tensor(_batch_y, device=ndl.cpu())


    # print(ndl.ops.forward_fourier_complex_1d(batch_x_real, batch_x_imag))
    # print(np.fft.fft(_batch_x_real + (_batch_x_imag * 1.0j)))

    # import sys; sys.exit(0)


    my_model = MyModel()
    # my_loss = ndl.nn.SoftmaxLoss()
    # my_optim = ndl.optim.Adam(my_model.parameters(), lr=0.8, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.001)

    '''
    conv_weight = my_model.my_conv.weight
    conv_bias = my_model.my_conv.bias
    linear_weight = my_model.my_linear.weight
    linear_bias = my_model.my_linear.bias
    '''

    torch_model = TorchModel()
    # torch_loss = torch.nn.CrossEntropyLoss()

    '''
    torch_model.torch_conv.weight = torch.nn.Parameter(torch.Tensor(conv_weight.numpy().transpose(3, 2, 0, 1)))
    torch_model.torch_conv.bias = torch.nn.Parameter(torch.Tensor(conv_bias.numpy().flatten()))
    torch_model.torch_linear.weight = torch.nn.Parameter(torch.Tensor(linear_weight.numpy().T))
    torch_model.torch_linear.bias = torch.nn.Parameter(torch.Tensor(linear_bias.numpy().flatten()))
    '''

    mine = my_model(batch_x)
    mine.backward()
    batch_x = ndl.Tensor(_batch_x, device=ndl.cuda())
    mine = my_model(batch_x)
    mine.backward()
    u = torch.Tensor(_batch_x)
    u.requires_grad = True
    torchs = torch_model(u)
    torchs.backward()

    print(batch_x.grad.numpy())
    print(u.grad)

    import sys; sys.exit(0)

    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=0.8, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.001)
    
    for it in range(1, 3):
        my_optim.reset_grad()
        batch_pred_y = my_model(batch_x)
        # print('conv weight at', it, ':', my_model.my_conv.weight)
        # print('conv bias at', it, ':', my_model.my_conv.bias)
        # print('linear weight at', it, ':', my_model.my_linear.weight)
        # print('linear bias at', it, ':', my_model.my_linear.bias)
        # print('batch_pred_y at', it, ':', batch_pred_y)
        # loss = my_loss(batch_pred_y, batch_y)
        loss = batch_pred_y
        # print('loss at', it, ':', loss)
        loss.backward()
        my_optim.step()

    print('\n------------\n')

    for it in range(1, 3):
        torch_optim.zero_grad()
        batch_pred_y = torch_model(torch.Tensor(_batch_x))
        # print('conv weight at', it, ':', torch_model.torch_conv.weight.detach().permute(2, 3, 1, 0))
        # print('conv bias at', it, ':', torch_model.torch_conv.bias)
        # print('linear weight at', it, ':', torch_model.torch_linear.weight.T)
        # print('linear bias at', it, ':', torch_model.torch_linear.bias)
        # print('batch_pred_y at', it, ':', batch_pred_y)
        # loss = torch_loss(batch_pred_y, torch.Tensor(_batch_y).long())
        loss = batch_pred_y
        # print('loss at', it, ':', loss)
        loss.backward()
        torch_optim.step()
