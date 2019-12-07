# Main-method, which initializes the model and the training #

import torch
import torch.nn as nn
from torchvision.utils import save_image
from tensorflow.examples.tutorials.mnist import input_data

from trainer import Trainer


class args(object): # --> a way of accessing dict-values like attributes
    def __init__(self, args_dict):
        self.__dict__ = args_dict

def main():

    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    arguments = args(args_dict=args_dict)
    trainer = Trainer(args=arguments, mnist=mnist)
    trainer.model.cuda()
    model, final_outputs, mu, log_var, code_output, mu_prior, log_var_prior = trainer.train(training_steps=trainer.training_steps,batch_size=trainer.batch_size,
                    beta_factor=trainer.beta_factor, dec_type=trainer.dec_type)
    torch.save(model, trainer.save_path)
    

    

if __name__ == "__main__":

    # the args-dictionary should contain all necessary arguments for the Trainer-Class
    arch_dict = {'in_out_size':784,
                'enc_hidden1': 500,
                'enc_hidden2': 500,
                'latent_code': 128,
                'dec_hidden1': 500,
                'dec_hidden2': 500,
                'label_size': 10} # --> Input- und Hidden-Schicht muss um Größe des Label-Vektor (bei MNIST:10) erweitern werden
    
    args_dict = {'training_steps': 10000,
                'batch_size': 100,
                'KL_weight': 5,
                'dec_type':"Bernoulli",
                'arch_dict': arch_dict,
                'learning_rate': 0.001,
                }

    main()    

