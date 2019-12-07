# Script containing the class, which builds the CVAE with the learnable priors # 

import torch
import torch.nn as nn
from tensorflow.examples.tutorials.mnist import input_data



class CVAE_learnable_prior(nn.Module):

    def __init__(self, arch_dict):

        super(CVAE_learnable_prior, self).__init__() # 
        self.arch_dict = arch_dict
        self.build_encoder()
        self.build_decoder()
        self.prior_network()
        

    def build_encoder(self):
        self.layer1 = nn.Sequential(
            nn.Linear(self.arch_dict['in_out_size'], self.arch_dict['enc_hidden1'], bias=True),
            nn.Softplus(),
            nn.Linear(self.arch_dict['enc_hidden1'], self.arch_dict['enc_hidden2'], bias=True), # 
            nn.Softplus(),
            )
        self.mu = nn.Linear(self.arch_dict['enc_hidden2'], self.arch_dict['latent_code'])
        self.log_var = nn.Sequential( nn.Linear(self.arch_dict['enc_hidden2'], self.arch_dict['latent_code']),
                                    nn.Softplus())
    
    def prior_network(self):
        self.prior_net_mu = nn.Linear(self.arch_dict['label_size'], self.arch_dict['latent_code'])
        self.prior_net_log_var = nn.Sequential(
                                    nn.Linear(self.arch_dict['label_size'], self.arch_dict['latent_code']),
                                    nn.Softplus())

    def reparametrize (self, mu_values, log_var_values):
        epsilon =  torch.randn_like(log_var_values) # 
        latent_outputs = mu_values + epsilon* (log_var_values.exp()).sqrt()
        return latent_outputs
    
    def build_decoder(self):
        self.dec_layer = nn.Sequential( # 
            nn.Linear(self.arch_dict['latent_code'], self.arch_dict['enc_hidden1'], bias=True),
            nn.Softplus(),
            nn.Linear(self.arch_dict['enc_hidden1'], self.arch_dict['enc_hidden2'], bias=True), # 
            nn.Softplus(),
            nn.Linear(self.arch_dict['enc_hidden2'], self.arch_dict['in_out_size'], bias=True), # 
            )
            
    def decode(self, code):
        reconstructions = self.dec_layer(code)
        return reconstructions


    def forward(self, x, label):
        enc_hid_out = self.layer1(x)
        mu_q = self.mu(enc_hid_out)
        log_var_q = self.log_var(enc_hid_out)
        code_output = self.reparametrize(mu_q, log_var_q)

        mu_prior = self.prior_net_mu(label)
        log_var_prior = self.prior_net_log_var(label)

        final_outputs = self.decode(code_output)
        return final_outputs, mu_q, log_var_q, code_output, mu_prior, log_var_prior