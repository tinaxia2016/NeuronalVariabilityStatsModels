# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:44:24 2023

a modified affine model for fitting single-trial neuronal population response fom one brain area to drifting gratings 
with varying orientations and contrast levels

this model is designed to capture trial-to-trial variability in the neuronal population

@author: xiaji
"""

import torch
import numpy as np
from sklearn.decomposition import FactorAnalysis


dev = "cpu"
device = torch.device(dev) 

class affine_model_contrast_specific_coeff():
    
    '''
    This is the generalized affine model used in Xia et al 2023
    '''
    # affine model with different coefficients for different contrast levels
    
    def __init__(self,x, n, n_ori, n_contrast, n_trial, n_compo, beta_p_init, psi_p_init):
        self.n = n
        self.n_stim = n_ori*n_contrast
        self.n_ori = n_ori
        self.n_contrast = n_contrast
        self.n_trial = n_trial
        self.x = x
        self.n_compo = n_compo
        self.SMALL = 1e-5
        
        #initialize params
        d_p = np.zeros((n, self.n_stim))
        alpha_p = np.zeros((n, n_compo, n_contrast))
        beta_p = np.tile(beta_p_init[:,:,None],(1,1,n_contrast))
        psi_p = np.maximum(psi_p_init, self.SMALL)
       

        for stim_i in range(self.n_stim):
            x_tmp = x[:, stim_i*n_trial:(stim_i+1)*n_trial]
            d_p[:, stim_i] = np.mean(x_tmp, axis=1)


        d_p = torch.from_numpy(d_p).to(device)
        
        alpha_p = torch.from_numpy(alpha_p).to(device)
        alpha_p.requires_grad=True

        beta_p = torch.from_numpy(beta_p).to(device)
        beta_p.requires_grad=True

        psi_p = torch.from_numpy(psi_p).to(device)
        psi_p.requires_grad=True

        self.d_p = d_p
        self.alpha_p = alpha_p
        self.beta_p = beta_p
        self.psi_p = psi_p


    def loss_nll(self, x, n_trial):
        x_var = torch.from_numpy(x)
        x_var = x_var.to(device)
        
        NLL = 0
        for contrast_i in range(self.n_contrast):
            for ori_i in range(self.n_ori):
                
                stim_i = contrast_i*self.n_ori + ori_i
                
                d_s = self.d_p[:, stim_i]
                psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL))

    
                A = self.alpha_p[:,:,contrast_i]*d_s[:,None] + self.beta_p[:,:,contrast_i]
                cov = A@A.T + torch.diag(psi_s)
    
                to_learn = torch.distributions.multivariate_normal.MultivariateNormal(loc=d_s, covariance_matrix= cov)
                NLL += -torch.mean(to_learn.log_prob(x_var[:, stim_i*n_trial:(stim_i+1)*n_trial].T))

        return NLL/self.n_stim

    def recon_data(self, x, n_trial):
        x_var = torch.from_numpy(x)
        x_var = x_var.to(device)
        
        E_z = torch.zeros([self.n_compo, self.n_stim*n_trial]).double()
        x_recon = torch.zeros([self.n, self.n_stim*n_trial])

        for contrast_i in range(self.n_contrast):
            for ori_i in range(self.n_ori):
                
                stim_i = contrast_i*self.n_ori + ori_i
                
                d_s = self.d_p[:, stim_i]
                psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL))

                x_s = x_var[:, stim_i*n_trial:(stim_i+1)*n_trial]
    
                A = self.alpha_p[:,:,contrast_i]*d_s[:, None] + self.beta_p[:,:, contrast_i]
    
                G = torch.linalg.inv(torch.eye(self.n_compo) + A.T@torch.diag(1/psi_s)@A)
                E_z[:, stim_i*n_trial: (stim_i+1)*n_trial] = G@A.T@torch.diag(1/psi_s)@(x_s - d_s[:, None])
                x_recon[:, stim_i*n_trial: (stim_i+1)*n_trial] = d_s[:, None] + A@E_z[:, stim_i*n_trial: (stim_i+1)*n_trial]

        return x_recon, E_z

    def train(self, lr0, x_test, n_trial_test):
        optimizer = torch.optim.Adam([self.alpha_p, self.beta_p, self.psi_p], lr0)
        decayRate = 0.96
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

        NLL_old = self.loss_nll(x_test, n_trial_test)
        
        for t in range(20001):
            optimizer.zero_grad()
            NLL = self.loss_nll(self.x, self.n_trial)
            NLL_test = self.loss_nll(x_test, n_trial_test)

            NLL.backward()
            optimizer.step()
            if (t-1) % 500 == 0:
                print(f"Iteration: {t}, Loss: {NLL.item():0.2f}, test Loss:  {NLL_test.item():0.2f}")
                if NLL_test > (NLL_old-1e-5):
                    print(f"Stop: Iteration: {t}, old test Loss: {NLL_old.item():0.5f}, new test Loss: {NLL_test.item():0.5f}")
                    break
                else:
                    NLL_old = NLL_test
                    lr_scheduler.step()
                    print('learning rate: ', lr_scheduler.get_last_lr())

        return self.d_p, self.alpha_p, self.beta_p, self.psi_p


