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

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP

from tabnet import TabNet_Decoder
from tabnet import TabNet



class TabNet(Model):
    def __init__(d_feat=6, final_out_dim=1, batch_size = 1024, n_d=64, n_a=64, n_shared=2, n_ind=2,
     n_steps=5, n_epochs=100,relax=1.2, vbs=128, seed = 710, optimizer='adam', GPU='2', pretrain_loss = 'custom', ps = 0.3):
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
        self.device = "cuda:%d" % (GPU) if torch.cuda.is_available() else "cpu"

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.tabnet_model = TabNet(inp_dim = self.d_feat, final_out_dim = self.final_out_dim, device = self.device)
        self.tabnet_decoder = TabNet_Decoder(inp_dim = self.d_feat, out_dim = )

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.gru_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.gru_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
    

    def pretrain(self, dataset = DatasetH):
        df_train, df_valid = dataset.prepare(
            ["pretrain"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        x_train = df_train["feature"]
        print(x_train.shape)
        # for epoch_idx in range(self.n_epochs):
        #     pretrain_epoch(x_train)
    



    def pretrain_epoch(self, x_train):
        S_mask =  torch.bernoulli(torch.empty(self.batch_size, self.d_feat).fill_(self.ps))
        x_train_values = x_train.values*(1-S_mask)
        y_train_values = x_train.values*(S_mask)

        self.tabnet_model.train()
        self.tabnet_decoder.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:

            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

            vec = self.tabnet_model(feature)
            f = self.tabnet_decoder(vec)

            loss = self.loss_fn(label, f, S_mask)

            self.train_optimizer.zero_grad()
            loss.backward()
            self.train_optimizer.step()


    def loss_fn(self, f_hat, f, S):
        loss = torch.tensor(0).to(self.device)
        sum_b = torch.sum(f, dim=0)/S.size(0)

        for j in range(S.size(1)):
            dominator = torch.tensor(0).to(self.device)
            for b1 in range(S.size(0)):
                dominator += torch.square(f[b1][j] - sum_b[j])
            dominator = torch.sqrt(dominator)
            numerator = torch.tensor(0).to(self.device)
            for b2 in range(S.size(0)):
                numerator = S[b2][j] * (f_hat[b2][j] - f[b2][j])
                loss += torch.square(numerator/dominator)
        return loss





