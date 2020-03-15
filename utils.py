# This script contains useful helper-methods # 
import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os

#from trainer import Trainer


def load_model(path_to_model_checkpoint):

    model = torch.load(path_to_model_checkpoint)
    model.cuda()
    return model

def generate_images(latent_dim, loaded_model, num_of_pics, digit):
    with torch.no_grad():

        epsilon = torch.randn(num_of_pics, latent_dim).cuda()
        labels = np.zeros([num_of_pics,10]) #

        if type(digit) == int:
            labels[:,digit] = 1

        elif digit=="mixed":
            indices = np.random.randint(0,9, size=(num_of_pics))
            labels[range(0,num_of_pics),tuple((indices))] = 1

        mu_prior = loaded_model.prior_net_mu(torch.Tensor(labels).cuda())  #
        log_var_prior = loaded_model.prior_net_log_var(torch.Tensor(labels).cuda()) 

        
        repam_latent_outputs = mu_prior + epsilon* (log_var_prior.exp()).sqrt() 
        samples = loaded_model.decode(repam_latent_outputs) # generates new data
        
        save_image(torch.sigmoid(samples).view(num_of_pics, 1, 28, 28),   # save the sampled images
                   "results_pics/"+ "CVAE_MNIST_{}".format(digit) + '.png')