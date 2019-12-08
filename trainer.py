# this class contains everything, which is relevant for the training of the model #

import torch
import torch.nn as nn
import os

from model import CVAE_learnable_prior
import utils


class Trainer(object):
    def __init__(self, args, mnist):
        self.mnist = mnist

        self.train_mode = args.train_mode
        self.training_steps = args.training_steps
        self.batch_size = args.batch_size
        self.beta_factor = args.KL_weight
        self.dec_type = args.dec_type

        self.arch_dict = args.arch_dict
        self.lr = args.learning_rate
        self.model = CVAE_learnable_prior(arch_dict=self.arch_dict)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.generate_digit = args.generate_digit

        # TODO: check whether the folder saved_models exists! --> otherwise create it
        self.save_path = "saved_models/CVAE_{}_Steps_beta_{}_lat_dim_{}_MNIST.pt".format(self.training_steps, self.beta_factor, self.arch_dict["latent_code"])


    def compute_loss(self, outputs, inputs, mu_values_p, mu_values_q, log_var_values_p, log_var_values_q, beta_factor, dec_type="Bernoulli"):

        if dec_type=="Bernoulli":
            log_likelihood = torch.sum(nn.functional.binary_cross_entropy(input=torch.sigmoid(outputs).view(-1,784),target=inputs.view(-1,784), reduction='none'), 1).mean()
        if dec_type=="Gaussian":
            log_likelihood = (outputs - inputs).pow(2).mean() # If a Gaussian Decoder is used, the KL-Term should be downweighted 

        # KL-Divergence between two Normal Distributions with diagonal covariance matrices (both are learned, since prior is learnable)
        KL_Divergence = -0.5 * torch.sum(1 + log_var_values_q - log_var_values_p - (( log_var_values_q.exp() + (mu_values_p - mu_values_q).pow(2)).div( log_var_values_p.exp()) ) ,1).mean() #/#, 1)

        overall_loss =  log_likelihood+ beta_factor*KL_Divergence  
        return overall_loss, KL_Divergence

    def train(self, training_steps, batch_size, beta_factor, dec_type):
        cum_loss = 0
        std_mean = 0
        for steps in range(0,training_steps):

            inputs, labels = self.mnist.train.next_batch(batch_size)
            
            inputs = torch.Tensor(inputs).cuda() 
            labels = torch.Tensor(labels).cuda()   
            self.optimizer.zero_grad()
            # 
            final_outputs, mu, log_var, code_output, mu_prior, log_var_prior = self.model(inputs, labels) # hier wird die forward-methode aufgerufen...
            #print(final_outputs[:10,:])
            loss, KL_Div = self.compute_loss(outputs=final_outputs, inputs=inputs, mu_values_p=mu_prior, mu_values_q=mu, log_var_values_p=log_var_prior, log_var_values_q=log_var, beta_factor=beta_factor, dec_type=dec_type)
            
            #print(loss.item())
            cum_loss = cum_loss + loss.item()
            std_mean = std_mean + log_var.mean().exp().sqrt().item()


            if steps%500==0 and steps>0:
                #writer.add_scalar('Loss/train', cum_loss/500, steps)
                #writer.add_scalar('Posterior_Std/train', std_mean/500, steps)

                print(cum_loss/500)
                print(std_mean/500)
                cum_loss = 0
                std_mean = 0

            loss.backward()
            self.optimizer.step()
        #writer.close()
        return self.model, final_outputs, mu, log_var, code_output, mu_prior, log_var_prior 

    def generate_digit_samples(self, path_to_model_checkpoint, num_of_pics, digit):

        loaded_model = utils.load_model(path_to_model_checkpoint=path_to_model_checkpoint)

        if not os.path.exists("results_pics/"):
            os.makedirs("results_pics/")
        utils.generate_images(loaded_model=loaded_model, num_of_pics=num_of_pics, digit=digit)
