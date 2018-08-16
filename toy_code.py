# https://github.com/pytorch/tutorials/blob/master/advanced_source/numpy_extensions_tutorial.py
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-tensors-and-autograd
# https://zhihu.com/question/66988664

import torch
from torch import nn

import numpy
from torch.autograd import Function

class DiffLossFunctionNp(Function):
    @staticmethod
    def forward(ctx, fea_in, fea_out):
        mat_in = fea_in.cpu().detach().numpy()
        mat_out = fea_out.cpu().detach().numpy()
        diff = (mat_in - mat_out) ** 2 /2.0
        count = diff.shape[0] * diff.shape[1]
        loss = numpy.sum(diff) / count
        ctx.save_for_backward(fea_in, fea_out)
        return torch.tensor(loss).to(torch.device('cuda:0'))
    @staticmethod
    def backward(ctx, grad_output):
        fea_in, fea_out = ctx.saved_tensors
        grad_mat_out = fea_out.cpu().detach().numpy()
        grad_mat_in = fea_in.cpu().detach().numpy()
        count = grad_mat_out.shape[0] * grad_mat_out.shape[1]
        grad_out = grad_output.cpu().detach().numpy()
        grad = grad_out * (grad_mat_out - grad_mat_in) / (count*1.0)
        return torch.tensor(-grad).to(torch.device('cuda:0')), torch.tensor(grad).to(torch.device('cuda:0'))

class DiffLossCustomNp(nn.Module):
    def __init__(self):
        super(DiffLossCustomNp, self).__init__()
    def forward(self, fea_in, fea_out):
        return DiffLossFunctionNp.apply(fea_in, fea_out)

class DiffLossFunction(Function):
    @staticmethod
    def forward(ctx, fea_in, fea_out):
        diff = (fea_in - fea_out) * (fea_in - fea_out) /2.0
        count = diff.shape[0] * diff.shape[1]
        loss = torch.sum(diff) / count
        ctx.save_for_backward(fea_in, fea_out)
        return loss
    @staticmethod
    def backward(ctx, grad_output):
        fea_in, fea_out = ctx.saved_tensors
        grad_mat_out = fea_out
        grad_mat_in = fea_in
        count = grad_mat_in.shape[0] * grad_mat_in.shape[1]
        grad = grad_output * (grad_mat_out - grad_mat_in) / (count*1.0)
        return -grad, grad

class DiffLossCustom(nn.Module):
    def __init__(self):
        super(DiffLossCustom, self).__init__()
    def forward(self, fea_in, fea_out):
        return DiffLossFunction.apply(fea_in, fea_out)

class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()
    def forward(self, fea_in, fea_out):
        diff = torch.pow(fea_in - fea_out, 2) /2.0
        return torch.mean(diff)

from torch.autograd.gradcheck import gradcheck
f_in  = torch.rand(3,3, dtype=torch.double, requires_grad=True, device=torch.device('cuda:0'))
f_out = torch.rand(3,3, dtype=torch.double, requires_grad=True, device=torch.device('cuda:0'))
input = (f_in, f_out)
grad = torch.ones(3,3, dtype=torch.double, device=torch.device('cuda:0'))

m1 = DiffLoss()
f1_in = torch.rand(3,3, dtype=torch.double, requires_grad=True, device=torch.device('cuda:0'))
f1_in.data = f_in.clone()
f1_out = torch.rand(3,3, dtype=torch.double, requires_grad=True, device=torch.device('cuda:0'))
f1_out.data = f_out.clone()
o1 = m1(f1_in, f1_out)
o1.backward(grad[0][0])
print 'out1', o1
print 'f1_in_grad', f1_in.grad
print 'f1_out_grad', f1_out.grad
m2 = DiffLossCustomNp()
f2_in = torch.rand(3,3, dtype=torch.double, requires_grad=True, device=torch.device('cuda:0'))
f2_in.data = f_in.clone()
f2_out = torch.rand(3,3, dtype=torch.double, requires_grad=True, device=torch.device('cuda:0'))
f2_out.data = f_out.clone()
o2 = m2(f2_in, f2_out)
o2.backward(grad[0][0])
print 'out2', o2
print 'f2_in_grad', f2_in.grad
print 'f2_out_grad', f2_out.grad
test = gradcheck(DiffLoss(), input, eps=1e-2, raise_exception=True)
print test
test = gradcheck(DiffLossCustomNp(), input, eps=1e-2, raise_exception=True)
print test
