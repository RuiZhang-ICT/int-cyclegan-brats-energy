import torch
from torch import nn

import numpy as np
from torch.autograd import Function

class EnergyLossTorch(nn.Module):
    def __init__(self):
        super(EnergyLossTorch, self).__init__()
    def forward(self, fea_out, fea_in):
        cell_range = 16
        batch = fea_out.shape[0]
        mat_index = (((fea_in +1) *127) / cell_range).int()
        mat_feature = fea_out - fea_in
        bins_num = 256 / cell_range
        bins_idx = torch.zeros(batch, bins_num, device=torch.device('cuda:0'))
        for bs in range(batch):
            fea_map = mat_feature[bs,0]
            for idx in range(bins_num):
                num = torch.sum(mat_index[bs,0]==idx)
                if num >0:
                    bins_idx[bs, idx] = torch.mean(fea_map[mat_index[bs,0]==idx])
        diff_abs = torch.abs(bins_idx[:,1:] - bins_idx[:,:-1])
        diff_tv = torch.mean(diff_abs)
        return diff_tv

class EnergyFunction(Function):
    @staticmethod
    def forward(ctx, fea_out, fea_in):
        cell_range = 16
        mat_in = fea_in.cpu().detach().numpy()
        mat_index = np.uint8((mat_in +1) *127) / cell_range
        mat_out = fea_out.cpu().detach().numpy()
        mat_feature = mat_out - mat_in
        bins_num = 256 / cell_range
        bins_idx = np.zeros(bins_num)
        for idx in range(bins_num):
            num = np.sum(mat_index==idx)
            if num >0: 
                bins_idx[idx] = np.mean(mat_feature[mat_index==idx])
        ctx.save_for_backward(fea_in, fea_out)
        return torch.tensor(bins_idx)
    @staticmethod
    def backward(ctx, grad_output):
        cell_range = 16
        fea_in, fea_out = ctx.saved_tensors
        mat_in = fea_in.cpu().detach().numpy()
        mat_out = fea_out.cpu().detach().numpy() 
        mat_index = np.uint8((mat_in +1) *127) / cell_range
        grad_feature = np.zeros(mat_out.shape)
        grad_out = grad_output.cpu().detach().numpy()
        bins_num = 256 / cell_range
        for idx in range(bins_num):
            num = np.sum(mat_index==idx)
            if num >0:
                grad_feature[mat_index==idx] = grad_out[idx] *1.0 / num
        return torch.tensor(grad_feature), None

class EnergyLoss(nn.Module):
     def __init__(self):
         super(EnergyLoss, self).__init__()
     def forward(self, fea_out, fea_in):
         return EnergyFunction.apply(fea_out, fea_in)

mod = EnergyLossTorch()
m_in = torch.rand(1,1,3,3, dtype=torch.double, device=torch.device('cuda:0')) 
m_out = torch.rand(1,1,3,3, dtype=torch.double, requires_grad=True, device=torch.device('cuda:0'))
m_in.data = (m_in.data +1) /2.0
m_out.data = (m_out.data +1) /2.0
out = mod(m_out, m_in)
print 'out', out
grad = torch.tensor(np.array(range(0,15)), dtype=torch.float, device=torch.device('cuda:0'))
out.backward(grad)
print 'grad', m_out.grad, m_in.grad

inputfea = (m_out, m_in)
from torch.autograd.gradcheck import gradcheck
test = gradcheck(mod, inputfea, eps=1e-2, raise_exception=True)
print test
