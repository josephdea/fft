"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for w in self.params:
            effective_grad = self.weight_decay * w.data + w.grad.detach()  # np.asarray(w.grad.detach().cached_data, dtype=w.data.dtype)
            self.u[w] = self.momentum * self.u.get(w, 0.0) + (1 - self.momentum) * effective_grad
            w.data = w.data + (-self.lr) * self.u[w]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = dict()
        self.v = dict()
        self.last_val = dict()

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for w in self.params:
            assert id(w) not in self.last_val or self.last_val.get(id(w), 0).data != w.data
            assert (id(w) not in self.last_val and w not in self.m and w not in self.v) or (id(w) in self.last_val and w in self.m and w in self.v)
            self.last_val[id(w)] = w

            effective_grad = w.data * self.weight_decay + w.grad.detach()
            
            self.m[w] = self.beta1 * self.m.get(w, 0.0) + (1 - self.beta1) * effective_grad
            self.v[w] = self.beta2 * self.v.get(w, 0.0) + (1 - self.beta2) * (effective_grad ** 2)
            
            m_corrected = self.m[w] / (1 - self.beta1 ** self.t)
            v_corrected = self.v[w] / (1 - self.beta2 ** self.t)
            
            w.data = w.data + ((-self.lr) * m_corrected / (v_corrected ** 0.5 + self.eps))
        ### END YOUR SOLUTION
