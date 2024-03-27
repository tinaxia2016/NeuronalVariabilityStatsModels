# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:45:52 2022

statistical models for fitting single-trial neuronal population response fom one brain area to multiple stimuli
these models are designed to capture trial-to-trial variability in the neuronal population

@author: xiaji
"""

import torch
import numpy as np
from sklearn.decomposition import FactorAnalysis


dev = "cpu"
device = torch.device(dev) 

####################################
class multiplicative_model():

    def __init__(self,x, n, n_stim, n_trial, n_compo, alpha_p_init, psi_p_init):
        self.n = n
        self.n_stim = n_stim
        self.n_trial = n_trial
        self.x = x
        self.n_compo = n_compo
        self.SMALL = 1e-5
        
        
        #initialize params
        d_p = np.zeros((n, n_stim))
        alpha_p = alpha_p_init
        psi_p = np.maximum(psi_p_init, self.SMALL)
       

        for stim_i in range(n_stim):
            x_tmp = x[:, stim_i*n_trial:(stim_i+1)*n_trial]
            d_p[:, stim_i] = np.mean(x_tmp, axis=1)


        d_p = torch.from_numpy(d_p).to(device)
        
        alpha_p = torch.from_numpy(alpha_p).to(device)
        alpha_p.requires_grad=True

        psi_p = torch.from_numpy(psi_p).to(device)
        psi_p.requires_grad=True

        self.d_p = d_p
        self.alpha_p = alpha_p
        self.psi_p = psi_p


    def loss_nll(self, x, n_trial):
        #calculating negative log likelihood of the model given data x 
        x_var = torch.from_numpy(x)
        x_var = x_var.to(device)
        
        NLL = 0
        for stim_i in range(self.n_stim):

            d_s = self.d_p[:, stim_i]
            psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL))

            A = self.alpha_p*d_s[:,None]
            cov = A@A.T + torch.diag(psi_s)

            to_learn = torch.distributions.multivariate_normal.MultivariateNormal(loc=d_s, covariance_matrix= cov)
            NLL += -torch.mean(to_learn.log_prob(x_var[:, stim_i*n_trial:(stim_i+1)*n_trial].T))

        return NLL/self.n_stim

    def recon_data(self, x, n_trial):
        #x_recon: reconstructed data x from the multiplicative model
        #E_z: the posterior (P(z|x)) expectation of latent variable z 
        
        x_var = torch.from_numpy(x)
        x_var = x_var.to(device)
        
        E_z = torch.zeros([self.n_compo, self.n_stim*n_trial]).double()
        x_recon = torch.zeros([self.n, self.n_stim*n_trial])

        for stim_i in range(self.n_stim):
            d_s = self.d_p[:, stim_i]
            psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL))
            x_s = x_var[:, stim_i*n_trial:(stim_i+1)*n_trial]

            A = self.alpha_p*d_s[:, None]

            G = torch.linalg.inv(torch.eye(self.n_compo) + A.T@torch.diag(1/psi_s)@A)
            E_z[:, stim_i*n_trial: (stim_i+1)*n_trial] = G@A.T@torch.diag(1/psi_s)@(x_s - d_s[:, None])
            x_recon[:, stim_i*n_trial: (stim_i+1)*n_trial] = d_s[:, None] + A@E_z[:, stim_i*n_trial: (stim_i+1)*n_trial]

        return x_recon, E_z

    def train(self, lr0, x_test, n_trial_test):
        #train parameters with gradient descent to minimize the negative log likelihood
        
        optimizer = torch.optim.Adam([self.alpha_p, self.psi_p], lr0)
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

        return self.d_p, self.alpha_p, self.psi_p



#################################
class affine_model():
    
    def __init__(self,x, n, n_stim, n_trial, n_compo, alpha_p_init, beta_p_init, psi_p_init):
        self.n = n
        self.n_stim = n_stim
        self.n_trial = n_trial
        self.x = x
        self.n_compo = n_compo
        self.SMALL = 1e-5
        
        #initialize params
        d_p = np.zeros((n, n_stim))
        alpha_p = alpha_p_init
        beta_p = beta_p_init
        psi_p = np.maximum(psi_p_init, self.SMALL)
       

        for stim_i in range(n_stim):
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
        for stim_i in range(self.n_stim):

            d_s = self.d_p[:, stim_i]
            psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL))

            A = self.alpha_p*d_s[:,None] + self.beta_p
            cov = A@A.T + torch.diag(psi_s)

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
            psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL))
            x_s = x_var[:, stim_i*n_trial:(stim_i+1)*n_trial]

            A = self.alpha_p*d_s[:, None] + self.beta_p

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
    

#####################################################
class additive_varp_model():
    '''
    This is the additive model used in Xia et al 2023
    '''
    def __init__(self,x, n, n_stim, n_trial, n_compo, h_p_init, psi_p_init):
        self.n = n
        self.n_stim = n_stim
        self.n_trial = n_trial
        self.x = x
        self.n_compo = n_compo
        self.SMALL = 1e-5
        
        #initialize params
        d_p = np.zeros((n, n_stim))
        
        for stim_i in range(n_stim):
            x_tmp = x[:, stim_i*n_trial:(stim_i+1)*n_trial]
            d_p[:, stim_i] = np.mean(x_tmp, axis=1)

        h_p = h_p_init
        psi_p = np.maximum(psi_p_init, self.SMALL)
            
        d_p = torch.from_numpy(d_p).to(device)
        
        h_p = torch.from_numpy(h_p).to(device)
        h_p.requires_grad=True

        psi_p = torch.from_numpy(psi_p).to(device)
        psi_p.requires_grad=True

        self.d_p = d_p
        self.h_p = h_p
        self.psi_p = psi_p


    def loss_nll(self, x, n_trial):
        x_var = torch.from_numpy(x)
        x_var = x_var.to(device)
        
        NLL = 0
        for stim_i in range(self.n_stim):

            d_s = self.d_p[:, stim_i]
            psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL))

            A = self.h_p 
            cov = A@A.T + torch.diag(psi_s)

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
            psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL))
            x_s = x_var[:, stim_i*n_trial:(stim_i+1)*n_trial]

            A = self.h_p

            G = torch.linalg.inv(torch.eye(self.n_compo) + A.T@torch.diag(1/psi_s)@A)
            E_z[:, stim_i*n_trial: (stim_i+1)*n_trial] = G@A.T@torch.diag(1/psi_s)@(x_s - d_s[:, None])
            x_recon[:, stim_i*n_trial: (stim_i+1)*n_trial] = d_s[:, None] + A@E_z[:, stim_i*n_trial: (stim_i+1)*n_trial]

        return x_recon, E_z

    def train(self, lr0, x_test, n_trial_test):
        optimizer = torch.optim.Adam([self.h_p, self.psi_p], lr0)
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

        return self.d_p, self.h_p, self.psi_p
##########################################################
class generalized_model():
    
    def __init__(self,x, n, n_stim, n_trial, x_test, n_trial_test, n_compo):

        #initialize params
        d_p = np.zeros((n, n_stim))
        
        psi_p = np.ones((n, n_stim))
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
            fa = FactorAnalysis()
            fa.n_components = n_compo
            fa.fit(res_x.T)

            F_p[:,:, stim_i] = fa.components_.T 
            psi_p[:,stim_i] = fa.noise_variance_
            
            ll += fa.score(res_x.T)
            
            x_test_tmp = x_test[:, stim_i*n_trial_test:(stim_i+1)*n_trial_test]
            res_x_test = x_test_tmp - d_p[:, stim_i:stim_i+1]
            
            ll_test += fa.score(res_x_test.T)
            
            z[stim_i*n_trial:(stim_i+1)*n_trial, :] = fa.transform(res_x.T)
            z_test[stim_i*n_trial_test:(stim_i+1)*n_trial_test, :] = fa.transform(res_x_test.T)
            
            x_recon[:, stim_i*n_trial:(stim_i+1)*n_trial] = d_p[:, stim_i:stim_i+1] + F_p[:,:, stim_i] @ (z[stim_i*n_trial:(stim_i+1)*n_trial, :].T)
            x_test_recon[:, stim_i*n_trial_test:(stim_i+1)*n_trial_test] = d_p[:, stim_i:stim_i+1] + F_p[:,:,stim_i] @ (z_test[stim_i*n_trial_test:(stim_i+1)*n_trial_test, :].T)
            
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
class additive_model():
    '''
    This is NOT the additive model used in Xia et al. 
    This additive model assumes stimulus-independent private variability for each neuron  
    '''
    
    def __init__(self,x, n, n_stim, n_trial, x_test, n_trial_test, n_compo):

        #initialize params
        d_p = np.zeros((n, n_stim))
        
        z = np.zeros((n_stim*n_trial, n_compo))
        z_test = np.zeros((n_stim*n_trial_test, n_compo))
        
        x_recon = np.zeros_like(x, dtype=float)
        x_test_recon = np.zeros_like(x_test, dtype=float)
        
        ll = 0
        ll_test = 0
        
        res_x  = np.zeros_like(x)
        res_x_test = np.zeros_like(x_test)
        
        for stim_i in range(n_stim):
            x_tmp = x[:, stim_i*n_trial:(stim_i+1)*n_trial]
            d_p[:, stim_i] = np.mean(x_tmp, axis=1)
            res_x[:, stim_i*n_trial:(stim_i+1)*n_trial] = x_tmp - d_p[:, stim_i:stim_i+1] 
            
            x_test_tmp = x_test[:, stim_i*n_trial_test:(stim_i+1)*n_trial_test]
            res_x_test[:, stim_i*n_trial_test:(stim_i+1)*n_trial_test] = x_test_tmp - d_p[:, stim_i:stim_i+1]
                   
        fa = FactorAnalysis()
        fa.n_components = n_compo
        fa.fit(res_x.T)

        h_p = fa.components_.T #h_p is n x n_compo
        psi_p = fa.noise_variance_

        ll = fa.score(res_x.T)
        ll_test = fa.score(res_x_test.T)

        z = fa.transform(res_x.T) # z is n_samples x n_compo
        z_test = fa.transform(res_x_test.T)
        
        for stim_i in range(n_stim):
            x_recon[:, stim_i*n_trial:(stim_i+1)*n_trial] = d_p[:, stim_i:stim_i+1] + h_p @ (z[stim_i*n_trial:(stim_i+1)*n_trial, :].T)
            x_test_recon[:, stim_i*n_trial_test:(stim_i+1)*n_trial_test] = d_p[:, stim_i:stim_i+1] + h_p @ (z_test[stim_i*n_trial_test:(stim_i+1)*n_trial_test, :].T)

        self.d_p = d_p
        self.psi_p = psi_p
        self.h_p = h_p
        self.z = z
        self.z_test = z_test
        
        self.NLL = -ll
        self.NLL_test = -ll_test 
        
        print('test NLL: ', self.NLL_test, 'train NLL: ', self.NLL)
        
        self.x_recon = x_recon
        self.x_test_recon = x_test_recon


class exponent_model():
    '''
    this model is not used in Xia et al 2023
    '''
    def __init__(self,x, n, n_stim, n_trial, n_compo, expo_p_init, beta_p_init, psi_p_init):
        self.n = n
        self.n_stim = n_stim
        self.n_trial = n_trial
        self.x = x
        self.n_compo = n_compo
        self.SMALL = 1e-5
        
        #initialize params
        d_p = np.zeros((n, n_stim))
        alpha_p = np.zeros((n,n_compo))
        beta_p = beta_p_init
        psi_p = np.maximum(psi_p_init, self.SMALL)
        expo_p = expo_p_init
       

        for stim_i in range(n_stim):
            x_tmp = x[:, stim_i*n_trial:(stim_i+1)*n_trial]
            d_p[:, stim_i] = np.mean(x_tmp, axis=1)


        d_p = torch.from_numpy(d_p).to(device)
        
        alpha_p = torch.from_numpy(alpha_p).to(device)
        alpha_p.requires_grad=True

        beta_p = torch.from_numpy(beta_p).to(device)
        beta_p.requires_grad=True

        psi_p = torch.from_numpy(psi_p).to(device)
        psi_p.requires_grad=True
        
        
        expo_p = torch.tensor(expo_p).to(device)
        expo_p.requires_grad=True
        

        self.d_p = d_p
        self.alpha_p = alpha_p
        self.beta_p = beta_p
        self.psi_p = psi_p
        self.expo_p = expo_p



    def loss_nll(self, x, n_trial):
        x_var = torch.from_numpy(x)
        x_var = x_var.to(device)
        
        NLL = 0
        for stim_i in range(self.n_stim):

            d_s = self.d_p[:, stim_i]
            psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL))
            
            A = self.alpha_p*(d_s[:,None]**self.expo_p) + self.beta_p
            cov = A@A.T + torch.diag(psi_s)

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
            psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL))
            x_s = x_var[:, stim_i*n_trial:(stim_i+1)*n_trial]

            A = self.alpha_p*(d_s[:, None]**self.expo_p) + self.beta_p

            G = torch.linalg.inv(torch.eye(self.n_compo) + A.T@torch.diag(1/psi_s)@A)
            E_z[:, stim_i*n_trial: (stim_i+1)*n_trial] = G@A.T@torch.diag(1/psi_s)@(x_s - d_s[:, None])
            x_recon[:, stim_i*n_trial: (stim_i+1)*n_trial] = d_s[:, None] + A@E_z[:, stim_i*n_trial: (stim_i+1)*n_trial]

        return x_recon, E_z

    def train(self, lr0, x_test, n_trial_test):
        optimizer = torch.optim.Adam([self.alpha_p, self.beta_p, self.psi_p, self.expo_p], lr0)
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

        return self.d_p, self.alpha_p, self.beta_p, self.psi_p, self.expo_p
