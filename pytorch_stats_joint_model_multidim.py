# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:03:39 2022

joint statistical models for fitting single-trial neuronal population response fom two brain areas to multiple stimuli
these models are designed to capture trial-to-trial variability in the neuronal population from two brain areas


@author: xiaji
"""

import torch
import numpy as np
from sklearn.utils.extmath import fast_logdet
from math import log
    
dev = "cpu"
device = torch.device(dev) 


def get_precision2(W, Phi):
    # precision is the inverse of covariance (W^T W + Phi)^-1
    # W is (n_compo, D)
    # Phi is (D,D)
    inv_Phi = np.linalg.pinv(Phi)
    n_comp = W.shape[0]
    A = np.linalg.pinv(np.eye(n_comp) + W@inv_Phi@W.T)
    precision = inv_Phi - inv_Phi@W.T@A@W@inv_Phi
    
    return precision


def ll_heterogeneous_Gaussian(X, precision, mean):
    #X is shape (K, D) 
    #############
    #code copied from scikit-learn factor analysis "score_samples"

    Xr = X - mean
    n_features = X.shape[1]
    log_like = -0.5 * (Xr * (np.dot(Xr, precision))).sum(axis=1)
    log_like -= 0.5 * (n_features * log(2.0 * np.pi) - fast_logdet(precision))
    
    return np.mean(log_like)


def CCA(x1, n1, x2, n2, n_compo, x1_test=0, x2_test=0, crossval = False):
    SMALL = 1e-5
    ## calculate residual 
    res1 = x1 - np.mean(x1, 1)[:, np.newaxis]
    res2 = x2 - np.mean(x2, 1)[:, np.newaxis]
    
    if not crossval:
        x1_test = np.zeros_like(x1)
        x2_test = np.zeros_like(x2)
    
    res1_test = x1_test - np.mean(x1_test, 1)[:, np.newaxis]
    res2_test = x2_test - np.mean(x2_test, 1)[:, np.newaxis]
    
    X_test = np.concatenate((x1_test, x2_test), axis=0)
    X = np.concatenate((x1, x2), axis=0)
    mu = np.mean(X, axis=1)
    
    cov = np.cov(np.concatenate((res1, res2), axis=0))
    cov12 = cov[:n1, n1:]
    cov11 = cov[:n1, :n1]
    cov22 = cov[n1:, n1:]
    
    #calculate cov11^(-1/2) 
    u, s, vt = np.linalg.svd(cov11)
    cov11_neg = np.linalg.pinv(u@np.diag(s**0.5)@vt)
    
    u, s, vt = np.linalg.svd(cov22)
    cov22_neg = np.linalg.pinv(u@np.diag(s**0.5)@vt)
    
    
    v1, p, v2t = np.linalg.svd(cov11_neg@cov12@cov22_neg)
    
    u1 = cov11_neg@v1[:,:n_compo]
    u2 = cov22_neg@(v2t[:n_compo,:].T)
    
    W1 = cov11@u1@np.diag(p[:n_compo]**0.5)
    W2 = cov22@u2@np.diag(p[:n_compo]**0.5)
    
    noise1 = cov11 - W1@W1.T + np.eye(n1)*SMALL
    noise2 = cov22 - W2@W2.T + np.eye(n2)*SMALL
    
    W = np.concatenate((W1, W2), axis=0)
    noise = np.zeros((n1+n2, n1+n2))
    
    noise[:n1, :n1] = noise1
    noise[n1:, n1:] = noise2
    
    res = np.concatenate((res1, res2), axis=0)
    
    G = np.linalg.pinv(np.eye(n_compo) + W.T@np.linalg.pinv(noise)@W)
    E_z = G@W.T@np.linalg.pinv(noise)@res
    
    res_test = np.concatenate((res1_test, res2_test), axis=0)
    E_z_test = G@W.T@np.linalg.pinv(noise)@res_test
    
    precision = get_precision2(W.T, noise)
    ll = ll_heterogeneous_Gaussian(X.T, precision, mu[np.newaxis,:])
    ll_test = ll_heterogeneous_Gaussian(X_test.T, precision, mu[np.newaxis,:])
        
    x_fit = W@E_z + mu[:, np.newaxis]
    x_test_fit = W@E_z_test + mu[:, np.newaxis]
    
    return W, noise1, noise2, [E_z, E_z_test], [x_fit, x_test_fit], [ll,ll_test]


################################################################################

class additive_varp_joint_model():
    '''
    This is the additive joint model used in Xia et al 2023 
    '''
    
    def __init__(self,x, n1, n2, n_stim, n_trial, n_compo, h_p_init, psi1_p_init, psi2_p_init):
        self.n1 = n1
        self.n2 = n2
        self.n = n1+n2
        self.n_stim = n_stim
        self.n_trial = n_trial
        self.x = x
        self.n_compo = n_compo
        self.SMALL = 1e-5
        
        n = n1+n2
        
        #initialize params
        d_p = np.zeros((n, n_stim))
        h_p = h_p_init
        
        for stim_i in range(n_stim):
            x_tmp = x[:, stim_i*n_trial:(stim_i+1)*n_trial]
            d_p[:, stim_i] = np.mean(x_tmp, axis=1)
            
        d_p = torch.from_numpy(d_p).to(device)
        
        h_p = torch.from_numpy(h_p).to(device)
        h_p.requires_grad=True
        
        
        psi1_p = torch.from_numpy(psi1_p_init)
        psi1_p.requires_grad = True
        
        psi2_p = torch.from_numpy(psi2_p_init)
        psi2_p.requires_grad = True

        self.d_p = d_p
        self.h_p = h_p
        self.psi1_p = psi1_p
        self.psi2_p = psi2_p


    def loss_nll(self, x, n_trial):
        
        x_var = torch.from_numpy(x).to(device)
        
        NLL = 0
        for stim_i in range(self.n_stim):
            
            d_s = self.d_p[:, stim_i]
            
            A = self.h_p
            cov_tmp = A@A.T
            
            
            psi_s = torch.zeros(self.n, self.n)
            ### evaluate whether psi is positive-definite
            if torch.linalg.eigvalsh(self.psi1_p[:,:,stim_i]).min()<0:
                L,Q = torch.linalg.eigh(self.psi1_p[:,:,stim_i])
                L_regularized = torch.maximum(L, torch.tensor(self.SMALL))
                psi_s[:self.n1, :self.n1] = Q @ torch.diag(L_regularized) @ Q.T.conj()
            else:
                psi_s[:self.n1, :self.n1] = self.psi1_p[:,:, stim_i]
            
            
            if torch.linalg.eigvalsh(self.psi2_p[:,:,stim_i]).min()<0:
                L,Q = torch.linalg.eigh(self.psi2_p[:,:,stim_i])
                L_regularized = torch.maximum(L, torch.tensor(self.SMALL))
                psi_s[self.n1:, self.n1:] = Q @ torch.diag(L_regularized) @ Q.T.conj()
            else:
                psi_s[self.n1:, self.n1:] = self.psi2_p[:,:, stim_i]
            ###
            
            cov = cov_tmp + psi_s  +  torch.eye(self.n)*self.SMALL
            
            to_learn = torch.distributions.multivariate_normal.MultivariateNormal(loc=d_s, covariance_matrix= cov)
            NLL += -torch.mean(to_learn.log_prob(x_var[:, stim_i*n_trial:(stim_i+1)*n_trial].T))

        return NLL/self.n_stim

    def recon_data(self, x, n_trial):
        x_var = torch.from_numpy(x)
        x_var = x_var.to(device)
        
        E_z = torch.zeros([np.n_compo, self.n_stim*n_trial]).double()
        x_recon = torch.zeros([self.n, self.n_stim*n_trial])

        for stim_i in range(self.n_stim):
            d_s = self.d_p[:, stim_i]
            
            res_s = x_var[:,stim_i*n_trial:(stim_i+1)*n_trial] - d_s[:, None]            
            
            A = self.h_p
            
            psi_s = torch.zeros(self.n, self.n)
            psi_s[:self.n1, :self.n1] = self.psi1_p[:,:, stim_i]
            psi_s[self.n1:, self.n1:] = self.psi2_p[:,:, stim_i]
           
            G = torch.linalg.inv(torch.eye(self.n_compo) + A.T@torch.linalg.pinv(psi_s)@A)
            
            E_z[:, stim_i*n_trial: (stim_i+1)*n_trial] = G@A.T@torch.linalg.pinv(psi_s)@res_s
            
            x_recon[:, stim_i*n_trial: (stim_i+1)*n_trial] = d_s[:, None] + A@E_z[:, stim_i*n_trial: (stim_i+1)*n_trial]

        return x_recon, E_z

    def train(self, lr0, x_test, n_trial_test):
        optimizer = torch.optim.Adam([self.h_p, self.psi1_p, self.psi2_p], lr0)
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

        return self.d_p, self.h_p
    
#########################################################################
class multiplicative_joint_model():
    
    def __init__(self,x, n1, n2, n_stim, n_trial,  n_compo, alpha_p_init, psi1_p_init, psi2_p_init):
        self.n1 = n1
        self.n2 = n2
        self.n = n1+n2
        self.n_stim = n_stim
        self.n_trial = n_trial
        self.x = x
        self.n_compo = n_compo
        self.SMALL = 1e-5
        
        n = n1+n2
        
        #initialize params
        d_p = np.zeros((n, n_stim))
        
        for stim_i in range(n_stim):
            x_tmp = x[:, stim_i*n_trial:(stim_i+1)*n_trial]
            d_p[:, stim_i] = np.mean(x_tmp, axis=1)

            
        d_p = torch.from_numpy(d_p).to(device)
        
        alpha_p = torch.from_numpy(alpha_p_init).to(device)
        alpha_p.requires_grad=True
        
        psi1_p = torch.from_numpy(psi1_p_init)
        psi1_p.requires_grad = True
        
        psi2_p = torch.from_numpy(psi2_p_init)
        psi2_p.requires_grad = True


        self.d_p = d_p
        self.alpha_p = alpha_p
        
        self.psi1_p = psi1_p
        self.psi2_p = psi2_p


    def loss_nll(self, x, n_trial):
        x_var = torch.from_numpy(x)
        x_var = x_var.to(device)
        
        NLL = 0
        
        for stim_i in range(self.n_stim):
        
            d_s = self.d_p[:, stim_i]

            A = self.alpha_p*d_s[:,None] 
            cov_tmp = A@A.T
            
            psi_s = torch.zeros(self.n, self.n)
            ### evaluate whether psi is positive-definite
            if torch.linalg.eigvalsh(self.psi1_p[:,:,stim_i]).min()<0:
                L,Q = torch.linalg.eigh(self.psi1_p[:,:,stim_i])
                L_regularized = torch.maximum(L, torch.tensor(self.SMALL))
                psi_s[:self.n1, :self.n1] = Q @ torch.diag(L_regularized) @ Q.T.conj()
            else:
                psi_s[:self.n1, :self.n1] = self.psi1_p[:,:, stim_i]
            
            
            if torch.linalg.eigvalsh(self.psi2_p[:,:,stim_i]).min()<0:
                L,Q = torch.linalg.eigh(self.psi2_p[:,:,stim_i])
                L_regularized = torch.maximum(L, torch.tensor(self.SMALL))
                psi_s[self.n1:, self.n1:] = Q @ torch.diag(L_regularized) @ Q.T.conj()
            else:
                psi_s[self.n1:, self.n1:] = self.psi2_p[:,:, stim_i]
            ###
            
            cov = cov_tmp + psi_s  +  torch.eye(self.n)*self.SMALL

            to_learn = torch.distributions.multivariate_normal.MultivariateNormal(loc=d_s, covariance_matrix= cov)
            NLL += -torch.mean(to_learn.log_prob(x_var[:, stim_i*n_trial:(stim_i+1)*n_trial].T))

        return NLL/self.n_stim

    def recon_data(self, x, n_trial):
        x_var = torch.from_numpy(x)
        x_var = x_var.to(device)
        
        E_z = torch.zeros([self.n_compo, self.n_stim*n_trial]).double()
        x_recon = torch.zeros([self.n, self.n_stim*n_trial])

        for stim_i in range(self.n_stim):
            d_s = self.d_p[:, stim_i]
            
            psi_s = torch.zeros(self.n, self.n)
            psi_s[:self.n1, :self.n1] = self.psi_p1[:,:, stim_i]
            psi_s[self.n1:, self.n1:] = self.psi_p2[:,:, stim_i]
            
            x_s = x_var[:, stim_i*n_trial:(stim_i+1)*n_trial]

            A = self.alpha_p*d_s[:, None] 

            G = torch.linalg.inv(torch.eye(self.n_compo) + A.T@torch.linalg.pinv(psi_s)@A)
            E_z[:, stim_i*n_trial: (stim_i+1)*n_trial] = G@A.T@torch.linalg.pinv(psi_s)@(x_s - d_s[:, None])
            x_recon[:, stim_i*n_trial: (stim_i+1)*n_trial] = d_s[:, None] + A@E_z[:, stim_i*n_trial: (stim_i+1)*n_trial]

        return x_recon, E_z

    def train(self, lr0, x_test, n_trial_test):
        optimizer = torch.optim.Adam([self.alpha_p, self.psi1_p, self.psi2_p], lr0)
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

        return self.d_p, self.alpha_p
#####################################################################################

class affine_joint_model():
    
    def __init__(self,x, n1, n2, n_stim, n_trial, n_compo, alpha_p_init, beta_p_init, psi1_p_init, psi2_p_init):
        self.n1 = n1
        self.n2 = n2
        self.n = n1+n2
        self.n_stim = n_stim
        self.n_trial = n_trial
        self.x = x
        self.n_compo = n_compo
        self.SMALL = 1e-5
        
        n = n1+n2
        
        #initialize params
        d_p = np.zeros((n, n_stim))

                
        psi1_p = torch.from_numpy(psi1_p_init)
        psi1_p.requires_grad = True
        
        psi2_p = torch.from_numpy(psi2_p_init)
        psi2_p.requires_grad = True
        

        for stim_i in range(n_stim):
            x_tmp = x[:, stim_i*n_trial:(stim_i+1)*n_trial]
            d_p[:, stim_i] = np.mean(x_tmp, axis=1)

            
        d_p = torch.from_numpy(d_p).to(device)
        
        alpha_p = torch.from_numpy(alpha_p_init).to(device)
        alpha_p.requires_grad=True
        
        beta_p = torch.from_numpy(beta_p_init).to(device)
        beta_p.requires_grad=True

        self.d_p = d_p
        self.alpha_p = alpha_p
        self.beta_p = beta_p
        
        self.psi1_p = psi1_p
        self.psi2_p = psi2_p

    def loss_nll(self, x, n_trial):
        x_var = torch.from_numpy(x)
        x_var = x_var.to(device)

        NLL = 0
        for stim_i in range(self.n_stim):

            d_s = self.d_p[:, stim_i]

            A = self.alpha_p*d_s[:,None] + self.beta_p
            cov_tmp = A@A.T
            
            psi_s = torch.zeros(self.n, self.n)
            ### evaluate whether psi is positive-definite
            if torch.linalg.eigvalsh(self.psi1_p[:,:,stim_i]).min()<0:
                L,Q = torch.linalg.eigh(self.psi1_p[:,:,stim_i])
                L_regularized = torch.maximum(L, torch.tensor(self.SMALL))
                psi_s[:self.n1, :self.n1] = Q @ torch.diag(L_regularized) @ Q.T.conj()
            else:
                psi_s[:self.n1, :self.n1] = self.psi1_p[:,:, stim_i]
            
            
            if torch.linalg.eigvalsh(self.psi2_p[:,:,stim_i]).min()<0:
                L,Q = torch.linalg.eigh(self.psi2_p[:,:,stim_i])
                L_regularized = torch.maximum(L, torch.tensor(self.SMALL))
                psi_s[self.n1:, self.n1:] = Q @ torch.diag(L_regularized) @ Q.T.conj()
            else:
                psi_s[self.n1:, self.n1:] = self.psi2_p[:,:, stim_i]
            ###
            
            cov = cov_tmp + psi_s  +  torch.eye(self.n)*self.SMALL

            to_learn = torch.distributions.multivariate_normal.MultivariateNormal(loc=d_s, covariance_matrix= cov)
            NLL += -torch.mean(to_learn.log_prob(x_var[:, stim_i*n_trial:(stim_i+1)*n_trial].T))

        return NLL/self.n_stim

    def recon_data(self, x, n_trial):
        x_var = torch.from_numpy(x)
        x_var = x_var.to(device)
        
        E_z = torch.zeros([self.n_compo, self.n_stim*n_trial]).double()
        x_recon = torch.zeros([self.n, self.n_stim*n_trial])

        for stim_i in range(self.n_stim):
            d_s = self.d_p[:, stim_i]
            
            psi_s = torch.zeros(self.n, self.n)
            psi_s[:self.n1, :self.n1] = self.psi_p1[:,:, stim_i]
            psi_s[self.n1:, self.n1:] = self.psi_p2[:,:, stim_i]
            
            psi_s[torch.eye(self.n)==1] = torch.maximum(psi_s[torch.eye(self.n)==1], torch.tensor(self.SMALL))
            
            x_s = x_var[:, stim_i*n_trial:(stim_i+1)*n_trial]

            A = self.alpha_p*d_s[:, None] + self.beta_p

            G = torch.linalg.inv(torch.eye(self.n_compo) + A.T@torch.linalg.pinv(psi_s)@A)
            
            E_z[:, stim_i*n_trial: (stim_i+1)*n_trial] = G@A.T@torch.linalg.pinv(psi_s)@(x_s - d_s[:, None])
            x_recon[:, stim_i*n_trial: (stim_i+1)*n_trial] = d_s[:, None] + A@E_z[:, stim_i*n_trial: (stim_i+1)*n_trial]

        return x_recon, E_z

    def train(self, lr0, x_test, n_trial_test):
        optimizer = torch.optim.Adam([self.alpha_p, self.beta_p, self.psi1_p, self.psi2_p], lr0)
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

        return self.d_p, self.alpha_p, self.beta_p
    
    
#####################################################################################
class generalized_joint_model():
    
    def __init__(self,x, n1,n2, n_stim, n_trial, n_compo, x_test, n_trial_test):
        
        n = n1+n2
        #initialize params
        d_p = np.zeros((n, n_stim))
        
        psi_p = np.zeros((n,n, n_stim))
        F_p = np.zeros((n, n_compo, n_stim))
        
        z = np.zeros((n_stim*n_trial, n_compo))
        z_test = np.zeros((n_stim*n_trial_test, n_compo))
        
        x_recon = np.zeros_like(x, dtype=float)
        x_test_recon = np.zeros_like(x_test, dtype=float)
        
        ll = 0
        ll_test = 0
        for stim_i in range(n_stim):
            x_tmp = x[:, stim_i*n_trial:(stim_i+1)*n_trial]
            d_p[:, stim_i] = np.mean(x_tmp, axis=1)

            res_x = x_tmp - d_p[:, stim_i:stim_i+1]
            x_test_tmp = x_test[:, stim_i*n_trial_test:(stim_i+1)*n_trial_test]
            res_x_test = x_test_tmp - d_p[:, stim_i:stim_i+1]
            
            
            W, noise1, noise2, [E_z, E_z_test], [x_fit, x_test_fit], [ll_stim,ll_stim_test] = CCA(res_x[:n1, :], n1, res_x[n1:,:], n2, n_compo,
                                                                                                  res_x_test[:n1, :], res_x_test[n1:,:], crossval=True)

            F_p[:, :, stim_i] = W.copy()
            psi_p[:n1, :n1, stim_i] = noise1
            psi_p[n1:, n1:, stim_i] = noise2
            
            ll += ll_stim
            ll_test += ll_stim_test
            
            z[stim_i*n_trial:(stim_i+1)*n_trial, :] = E_z.T
            z_test[stim_i*n_trial_test:(stim_i+1)*n_trial_test, :] = E_z_test.T
            
            x_recon[:, stim_i*n_trial:(stim_i+1)*n_trial] = x_fit + d_p[:,stim_i:stim_i+1]
            x_test_recon[:, stim_i*n_trial_test:(stim_i+1)*n_trial_test] = x_test_fit + d_p[:,stim_i:stim_i+1]
            
        self.d_p = d_p
        self.psi_p = psi_p
        self.F_p = F_p
        self.z = z
        self.z_test = z_test
        
        self.NLL = -ll/n_stim
        self.NLL_test = -ll_test/n_stim 
        
        print('test NLL: ', self.NLL_test, 'train NLL: ', self.NLL)
        
        self.x_recon = x_recon
        self.x_test_recon = x_test_recon
###################################################
class additive_joint_model():
    '''
    This is NOT the additive joint model used in Xia et al 2023, this model assumes stimulus-independent area-private variability  
    '''
    def __init__(self,x, n1,n2, n_stim, n_trial, n_compo, x_test, n_trial_test):
        
        n = n1+n2
        #initialize params
        d_p = np.zeros((n, n_stim))
        
        psi_p = np.zeros((n, n))
        F_p = np.zeros((n, n_compo))
        
        z = np.zeros((n_stim*n_trial, n_compo))
        z_test = np.zeros((n_stim*n_trial_test, n_compo))
        
        x_recon = np.zeros_like(x, dtype=float)
        x_test_recon = np.zeros_like(x_test, dtype=float)
        
        res_x  = np.zeros_like(x, dtype=float)
        res_x_test = np.zeros_like(x_test, dtype=float)
        
        for stim_i in range(n_stim):
            x_tmp = x[:, stim_i*n_trial:(stim_i+1)*n_trial]
            d_p[:, stim_i] = np.mean(x_tmp, axis=1)
            res_x[:, stim_i*n_trial:(stim_i+1)*n_trial] = x_tmp - d_p[:, stim_i:stim_i+1] 
            
            x_test_tmp = x_test[:, stim_i*n_trial_test:(stim_i+1)*n_trial_test]
            res_x_test[:, stim_i*n_trial_test:(stim_i+1)*n_trial_test] = x_test_tmp - d_p[:, stim_i:stim_i+1]
            
            
        W, noise1, noise2, [E_z, E_z_test], [x_fit, x_test_fit], [ll,ll_test] = CCA(res_x[:n1, :], n1, res_x[n1:,:], n2, n_compo,
                                                                                              res_x_test[:n1, :], res_x_test[n1:,:], crossval=True)

        F_p = W.copy()
        psi_p[:n1, :n1] = noise1
        psi_p[n1:, n1:] = noise2


        z = E_z.T
        z_test = E_z_test.T

        for stim_i in range(n_stim):
            x_recon[:, stim_i*n_trial:(stim_i+1)*n_trial] = d_p[:, stim_i:stim_i+1] + x_fit[:, stim_i*n_trial:(stim_i+1)*n_trial]
            x_test_recon[:, stim_i*n_trial_test:(stim_i+1)*n_trial_test] = d_p[:, stim_i:stim_i+1] + x_test_fit[:, stim_i*n_trial_test:(stim_i+1)*n_trial_test]
            
        self.d_p = d_p
        self.psi_p = psi_p
        self.F_p = F_p
        self.z = z
        self.z_test = z_test
        
        self.NLL = -ll
        self.NLL_test = -ll_test 
        
        print('test NLL: ', self.NLL_test, 'train NLL: ', self.NLL)
        
        self.x_recon = x_recon
        self.x_test_recon = x_test_recon

