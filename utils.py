# This script contains useful helper-methods # 
import torch
import torch.nn as nn
from torchvision.utils import save_image


def load_model(path_to_model_checkpoint, model_class, arch_dict, path_to_arch_dict=None):

    model = model_class(arch_dict=arch_dict)
    model.cuda()
    model = torch.load(path_to_model_checkpoint)
    return model

def generate_images(loaded_model, num_of_pics):
    with torch.no_grad():

        epsilon = torch.randn(num_of_pics, 128).cuda()
        inputs,labels = mnist.train.next_batch(1) # one samples means, that only one number is generated, since the prior parameters of this number are inferred

        _, _, _, _, mu_prior, log_var_prior = model.forward(None, torch.Tensor(labels).cuda())
        repam_latent_outputs = mu_prior + epsilon* (log_var_prior.exp()).sqrt()

        samples = model.decode(repam_latent_outputs) # generates new data
        save_image(torch.sigmoid(samples).view(num_of_pics, 1, 28, 28),   # final_outputs entsprechen den rekonstruierten Bildern und sampels den neu generierten Bildern
                    "results_pics/"+ str("CVAE_MNIST_sigmoid_beta_new3") + '.png')