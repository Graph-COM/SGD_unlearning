import numpy as np
from tqdm import tqdm
from opacus.optimizers.optimizer import DPOptimizer
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def stochastic_gradient_descent_algorithm(init_point, dim_w, X, y, lam, sigma, device, burn_in = 10000,
                                           len_list = 1, step=0.1, M = 1, m = 0, projection = 0, batch_size = 0, batch_idx = None,
                                           num_class = 0):
    if batch_idx is None or num_class == 0:
        print('there is no valid batch idx assignment, please check!')
        print('or there is no valid num class')
        return
    # when only has binary class
    if num_class == 2:
        # randomly sample from N(0, C_lsi)
        if init_point == None:
            if m == 0:
                print('m not assigned, please check!')
                return
            var = (2 * sigma**2) / m
            std = torch.sqrt(torch.tensor(var))
            w0 = torch.normal(mean=0, std=std, size=(dim_w,)).reshape(-1).to(device)
        else:
            w0 = init_point.to(device)
        wi = w0
        samples = []
        if batch_size == 0:
            for i in range(len_list + burn_in):
                z = torch.sigmoid(y * X.mv(wi))
                per_sample_grad = X * ((z-1) * y).unsqueeze(-1)
                row_norms = torch.norm(per_sample_grad,dim=1)
                clipped_grad = per_sample_grad * ( M / row_norms).view(-1,1)
                clipped_grad[row_norms <= M] = per_sample_grad[row_norms <= M]
                clipped_grad = clipped_grad + lam * wi.repeat(X.size(0),1)
                grad = clipped_grad.mean(0)
                wi = wi.detach() - step * grad + np.sqrt(2 * step * sigma**2) * torch.randn(dim_w).to(device)
                if projection != 0:
                    w_norm = torch.norm(wi, p=2)
                    if w_norm > projection:
                        wi = (wi / w_norm) * projection
                samples.append(wi.detach().cpu().numpy())
            return samples[burn_in:]
        else:
            # batch stochastic sgd
            # first sample a batch of y and X
            num_batch = int(len(batch_idx) / batch_size)
            for i in range(len_list + burn_in):
                for step_idx in range(num_batch):
                    X_batch = X[batch_idx[step_idx*batch_size:(step_idx + 1)*batch_size]]
                    y_batch = y[batch_idx[step_idx*batch_size:(step_idx + 1)*batch_size]]
                    z = torch.sigmoid(y_batch * X_batch.mv(wi))
                    per_sample_grad = X_batch * ((z-1) * y_batch).unsqueeze(-1)
                    row_norms = torch.norm(per_sample_grad,dim=1)
                    clipped_grad = per_sample_grad * ( M / row_norms).view(-1,1)
                    clipped_grad[row_norms <= M] = per_sample_grad[row_norms <= M]
                    clipped_grad = clipped_grad + lam * wi.repeat(X_batch.size(0),1)
                    grad = clipped_grad.mean(0)
                    wi = wi.detach() - step * grad + np.sqrt(2 * step * sigma**2) * torch.randn(dim_w).to(device)
                    if projection != 0:
                        w_norm = torch.norm(wi, p=2)
                        if w_norm > projection:
                            wi = (wi / w_norm) * projection
                samples.append(wi.detach().cpu().numpy())
            return samples[burn_in:]
    # when do multi class
    elif num_class == 10:
        if init_point == None:
            if m == 0:
                print('m not assigned, please check!')
            var = (2 * sigma**2) / m
            std = torch.sqrt(torch.tensor(var))
            w0 = torch.normal(mean=0, std=std, size=(dim_w, num_class)).to(device)
        else:
            w0 = init_point.to(device)
        wi = w0
        samples = []
        if batch_size == 0:
            for i in range(len_list + burn_in):
                z = torch.sigmoid(y * torch.matmul(X.unsqueeze(1), wi.unsqueeze(0)).squeeze(1))
                per_sample_grad = (X.unsqueeze(1).expand(-1, 10, -1) * ((z-1) * y).unsqueeze(-1)).transpose(1, 2)
                per_sample_grad = per_sample_grad.reshape(-1, dim_w * num_class)
                row_norms = torch.norm(per_sample_grad, dim=1)
                clipped_grad = per_sample_grad * ( M / row_norms).view(-1,1)
                clipped_grad[row_norms <= M] = per_sample_grad[row_norms <= M]
                clipped_grad = clipped_grad.reshape(-1, dim_w, num_class)
                clipped_grad = clipped_grad + lam * wi.repeat(X.size(0),1, 1)
                grad = clipped_grad.mean(0)
                wi = wi.detach() - step * grad + np.sqrt(2 * step * sigma**2) * torch.randn(dim_w, num_class).to(device)
                if projection != 0:
                    w_norm = torch.norm(wi, p=2)
                    if w_norm > projection:
                        wi = (wi / w_norm) * projection
                samples.append(wi.detach().cpu().numpy())
            return samples[burn_in:]
        else:
            # batch stochastic sgd for langevin
            # first sample a batch of y and X
            num_batch = int(len(batch_idx) / batch_size)
            for i in range(len_list + burn_in):
                for step_idx in range(num_batch):
                    X_batch = X[batch_idx[step_idx*batch_size:(step_idx + 1)*batch_size]]
                    y_batch = y[batch_idx[step_idx*batch_size:(step_idx + 1)*batch_size]]
                    z = torch.sigmoid(y_batch * torch.matmul(X_batch.unsqueeze(1), wi.unsqueeze(0)).squeeze(1))
                    per_sample_grad = (X_batch.unsqueeze(1).expand(-1, 10, -1) * ((z-1) * y_batch).unsqueeze(-1)).transpose(1, 2)
                    per_sample_grad = per_sample_grad.reshape(-1, dim_w * num_class)
                    row_norms = torch.norm(per_sample_grad,dim=1)
                    clipped_grad = per_sample_grad * ( M / row_norms).view(-1,1)
                    clipped_grad[row_norms <= M] = per_sample_grad[row_norms <= M]
                    clipped_grad = clipped_grad.reshape(-1, dim_w, num_class)
                    clipped_grad = clipped_grad + lam * wi.repeat(X_batch.size(0),1, 1)
                    grad = clipped_grad.mean(0)
                    wi = wi.detach() - step * grad + np.sqrt(2 * step * sigma**2) * torch.randn(dim_w, num_class).to(device)
                    if projection != 0:
                        w_norm = torch.norm(wi, p=2)
                        if w_norm > projection:
                            wi = (wi / w_norm) * projection
                samples.append(wi.detach().cpu().numpy())
            return samples[burn_in:]