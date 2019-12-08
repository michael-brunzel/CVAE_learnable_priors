# This script contains useful helper-methods # 
import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os


def load_model(path_to_model_checkpoint, model_class=None, arch_dict=None, path_to_arch_dict=None):

    #model = model_class(arch_dict=arch_dict)
    #model.cuda()
    model = torch.load(path_to_model_checkpoint)
    model.cuda()
    return model

def generate_images(loaded_model, num_of_pics, digit):
    with torch.no_grad():

        epsilon = torch.randn(num_of_pics, 128).cuda()
        labels = np.zeros([num_of_pics,10]) #mnist.train.next_batch(1) # one samples means, that only one number is generated, since the prior parameters of this number are inferred
        labels[:,digit] = 1

        mu_prior = loaded_model.prior_net_mu(torch.Tensor(labels).cuda())  #.forward(None, torch.Tensor(labels).cuda())
        log_var_prior = loaded_model.prior_net_log_var(torch.Tensor(labels).cuda()) 

        repam_latent_outputs = mu_prior + epsilon* (log_var_prior.exp()).sqrt()
        samples = loaded_model.decode(repam_latent_outputs) # generates new data

        save_image(torch.sigmoid(samples).view(num_of_pics, 1, 28, 28),   # final_outputs entsprechen den rekonstruierten Bildern und sampels den neu generierten Bildern
                    "results_pics/"+ "CVAE_MNIST_{}".format(digit) + '.png')