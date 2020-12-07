# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import roc_auc_score, mean_squared_error
import logging
from ...utils import (
    unpack_archive_with_buffer,
    save_multiple_parts_file,
    create_save_path,
    drop_nan_by_y_index,
)
from ...log import get_module_logger, TimeInspector

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP




class TabNet_Model(Model):
    def __init__(self, d_feat=158, final_out_dim=64, batch_size = 8192, n_d=64, n_a=64, n_shared=2, n_ind=2,
     n_steps=5, n_epochs=100,relax=1.2, vbs=2048, seed = 710, optimizer='adam', GPU='1', pretrain_loss = 'custom', ps = 0.3, lr = 0.01):
        # set hyper-parameters.
        self.d_feat = d_feat
        self.final_out_dim = final_out_dim
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = optimizer.lower()
        self.pretrain_loss = pretrain_loss
        self.seed = seed
        self.ps = ps
        self.n_epochs = n_epochs
        self.logger = get_module_logger("TabNet")
        self.device = "cuda:%s" % (GPU) if torch.cuda.is_available() else "cpu"

        self.logger.info(
                "TabNet:"
                "\nbatch_size : {}"
                "\nvirtual bs : {}".format(
                    self.batch_size,
                    vbs
                )
        )
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.tabnet_model = TabNet(inp_dim = self.d_feat, final_out_dim = self.final_out_dim, vbs = vbs,device = self.device).to(self.device)
        self.tabnet_decoder = TabNet_Decoder(self.final_out_dim, self.d_feat, n_shared, n_ind, vbs, n_steps, self.device).to(self.device)
  
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(list(self.tabnet_model.parameters())+list(self.tabnet_decoder.parameters()), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(list(self.tabnet_model.parameters())+list(self.tabnet_decoder.parameters()), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
    

    def pretrain(self, dataset = DatasetH):
        [df_train] = dataset.prepare(
            ["pretrain"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        df_train.fillna(df_train.mean(), inplace = True)
        x_train = df_train["feature"]
        for epoch_idx in range(self.n_epochs):
            self.logger.info('epoch: %s' % (epoch_idx))
            self.logger.info("training...")
            self.pretrain_epoch(x_train)
            self.logger.info("evaluating...")
            train_loss = self.pretrain_test_epoch(x_train)
            self.logger.info("train %.6f" % (train_loss))
    
    def fit():
        pass

    def predict():
        pass

    def pretrain_epoch(self, x_train):
        
        train_set = torch.from_numpy(x_train.values)
        train_set[torch.isnan(train_set)] = 0
        indices = np.arange(len(train_set))
        np.random.shuffle(indices)

        self.tabnet_model.train()
        self.tabnet_decoder.train()

        for i in range(len(indices))[:: self.batch_size]:

            if len(indices) - i < self.batch_size:
                break
            
            S_mask =  torch.bernoulli(torch.empty(self.batch_size, self.d_feat).fill_(self.ps))
            x_train_values = train_set[indices[i : i + self.batch_size]] * (1-S_mask)
            y_train_values = train_set[indices[i : i + self.batch_size]] * (S_mask)

            S_mask = S_mask.to(self.device)
            feature = x_train_values.float().to(self.device)
            label = y_train_values.float().to(self.device)
            priors = 1-S_mask
            (vec, sparse_loss) = self.tabnet_model(feature, priors)
            f = self.tabnet_decoder(vec)
            loss = self.loss_fn(label, f, S_mask) 

            self.train_optimizer.zero_grad()
            loss.backward()
            self.train_optimizer.step()

    def pretrain_test_epoch(self, x_train):

        train_set = torch.from_numpy(x_train.values)
        indices = np.arange(len(train_set))

        self.tabnet_model.eval()
        self.tabnet_decoder.eval()

        losses = []

        for i in range(len(indices))[:: self.batch_size]:

            if len(indices) - i < self.batch_size:
                break

            S_mask =  torch.bernoulli(torch.empty(self.batch_size, self.d_feat).fill_(self.ps))
            x_train_values = train_set[indices[i : i + self.batch_size]] * (1-S_mask)
            y_train_values = train_set[indices[i : i + self.batch_size]] * (S_mask)

            feature = x_train_values.float().to(self.device)
            label = y_train_values.float().to(self.device)
            S_mask = S_mask.to(self.device)
            priors = 1-S_mask
            (vec, sparse_loss) = self.tabnet_model(feature, priors)
            f = self.tabnet_decoder(vec)
            
            loss = self.loss_fn(label, f, S_mask)
            losses.append(loss.item())

        return np.mean(losses)


    def loss_fn(self, f_hat, f, S):
        down_mean = torch.mean(f, dim=0)
        down = torch.sqrt(torch.sum(torch.square(f-down_mean), dim = 0))
        up = (f_hat - f)*S
        return torch.sum(torch.square(up/down))

class DecoderStep(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs, device):
        super().__init__()
        self.fea_tran = FeatureTransformer(inp_dim, out_dim, shared, n_ind, vbs, device)
        self.fc = nn.Linear(out_dim, out_dim).to(device)
    
    def forward(self, x):
        x = self.fea_tran(x)
        return self.fc(x)



class TabNet_Decoder(nn.Module):
    def __init__(self, inp_dim, out_dim, n_shared, n_ind, vbs, n_steps, device):
        """
        TabNet decoder that is used in pre-training
        """
        self.out_dim = out_dim

        super().__init__()
        if n_shared > 0:
            self.shared = nn.ModuleList()
            for x in range(n_shared):
                self.shared.append(None) # preset the linear function we will use
        else:
            self.shared=None
        self.n_steps = n_steps
        self.steps = nn.ModuleList()
        for x in range(n_steps):
            self.steps.append(DecoderStep(inp_dim, out_dim, self.shared, n_ind, vbs, device))

    def forward(self, x):
        out = torch.zeros(x.size(0), self.out_dim).to(x.device)
        for step in self.steps:
            out += step(x)
        return out



class TabNet(nn.Module):
    def __init__(self, inp_dim=6, final_out_dim=6, n_d=64, n_a=64, n_shared=2, n_ind=2, n_steps=5, relax=1.2, vbs=1024, device = 'cpu'):
        """
        TabNet AKA the original encoder

        Args:
            n_d: dimension of the features used to calculate the final results
            n_a: dimension of the features input to the attention transformer of the next step
            n_shared: numbr of shared steps in feature transfomer(optional)
            n_ind: number of independent steps in feature transformer
            n_steps: number of steps of pass through tabbet
            relax coefficient:
            virtual batch size:
        """
        super().__init__()

        # set the number of shared step in feature transformer
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * (n_d + n_a)))
            for x in range(n_shared - 1):
                self.shared.append(nn.Linear(n_d + n_a, 2 * (n_d + n_a))) # preset the linear function we will use
        else:
            self.shared=None

        self.first_step = FeatureTransformer(inp_dim, n_d + n_a, self.shared, n_ind, vbs, device) 
        self.steps = nn.ModuleList()
        for x in range(n_steps-1):
            self.steps.append(DecisionStep(inp_dim, n_d, n_a, self.shared, n_ind, relax, vbs, device))
        self.fc = nn.Linear(n_d, final_out_dim)
        self.bn = nn.BatchNorm1d(inp_dim, momentum=0.01)
        self.n_d = n_d
    
    def forward(self, x, priors):
        assert not torch.isnan(x).any()
        x = self.bn(x)
        x_a = self.first_step(x)[:, self.n_d:]
        sparse_loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0), self.n_d).to(x.device)
        for step in self.steps:
            x_te, l = step(x, x_a, priors)
            out += F.relu(x_te[:, :self.n_d]) #split the feautre from feat_transformer
            x_a = x_te[:, self.n_d:]
            sparse_loss += l
        return self.fc(out), sparse_loss


class GBN(nn.Module):
    """
    Ghost Batch Normalization
    an efficient way of doing batch normalization

    Args:
        vbs: virtual batch size
    """
    def __init__(self, inp, vbs=1024, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp, momentum=momentum)
        self.vbs = vbs
    
    def forward(self, x):
        chunk = torch.chunk(x, x.size(0)//self.vbs,0)
        res = [self.bn(y) for y in chunk]
        return torch.cat(res, 0)


class GLU(nn.Module):
    """
    GLU block that extracts only the most essential information

    Args:
        vbs: virtual batch size
    """
    def __init__(self, inp_dim, out_dim, fc=None, vbs=1024):
        super().__init__()
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(inp_dim, out_dim*2)
        self.bn = GBN(out_dim * 2, vbs=vbs) 
        self.od = out_dim
   
    def forward(self, x):
        x = self.bn(self.fc(x))
        return torch.mul(x[:, :self.od], torch.sigmoid(x[:, self.od:]))


class AttentionTransformer(nn.Module):
    """
    Args:
        relax: relax coefficient. The greater it is, we can
        use the same features more. When it is set to 1
        we can use every feature only once
    """
    def __init__(self, d_a, inp_dim, relax, vbs=1024):
        super().__init__()
        self.fc = nn.Linear(d_a, inp_dim)
        self.bn = GBN(inp_dim, vbs=vbs)
        self.r = relax

    #a:feature from previous decision step
    def forward(self, a, priors):  
        a = self.bn(self.fc(a)) 
        mask = sparsemax(a * priors) 
        priors = priors * (self.r - mask)  #updating the prior
        return mask


class FeatureTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs, device):
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared:
            self.shared.append(GLU(inp_dim, out_dim, shared[0], vbs=vbs))
            first= False    
            for fc in shared[1:]:
                self.shared.append(GLU(out_dim, out_dim, fc, vbs=vbs))
        else:
            self.shared = None
        self.independ = nn.ModuleList()
        if first:
            self.independ.append(GLU(inp,out_dim,vbs=vbs))
        for x in range(first, n_ind):
            self.independ.append(GLU(out_dim,out_dim,vbs=vbs))
        self.scale = torch.sqrt(torch.tensor([.5], device=device))
    
    def forward(self, x):
        if self.shared:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x * self.scale
        for glu in self.independ:
            x = torch.add(x, glu(x))
            x = x * self.scale
        return x



class DecisionStep(nn.Module):
    """
    One step for the TabNet 
    """
    def __init__(self, inp_dim, n_d, n_a, shared, n_ind, relax, vbs, device):
        super().__init__()
        self.atten_tran =  AttentionTransformer(n_a, inp_dim, relax,vbs)
        self.fea_tran = FeatureTransformer(inp_dim, n_d + n_a, shared, n_ind, vbs, device)

    def forward(self, x, a, priors):
        mask = self.atten_tran(a, priors)
        sparse_loss = ((-1)*mask*torch.log(mask+1e-10)).mean()
        x = self.fea_tran(x * mask)
        return x ,sparse_loss

def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)

class SparsemaxFunction(Function):
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        """sparsemax: normalizing sparse transform (a la softmax)

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
        input : torch.Tensor
            any shape
        dim : int
            dimension along which to apply sparsemax

        Returns
        -------
        output : torch.Tensor
            same shape as input

        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = SparsemaxFunction._threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        """Sparsemax building block: compute the threshold

        Parameters
        ----------
        input: torch.Tensor
            any dimension
        dim : int
            dimension along which to apply the sparsemax

        Returns
        -------
        tau : torch.Tensor
            the threshold value
        support_size : torch.Tensor

        """

        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size


sparsemax = SparsemaxFunction.apply

