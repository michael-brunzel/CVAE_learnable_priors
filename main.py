# Main-method, which initializes the model and the training #

import torch
import torch.nn as nn
import os
from torchvision.utils import save_image
from tensorflow.examples.tutorials.mnist import input_data

import trainer
from trainer import Trainer

import importlib
#import utils
#importlib.reload(utils) #


class args(object): 
    def __init__(self, args_dict):
        self.__dict__ = args_dict

def main():

    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    arguments = args(args_dict=args_dict)
    train_inst = Trainer(args=arguments, mnist=mnist)
    train_inst.model.cuda()

    if train_inst.train_mode:
        model, final_outputs, mu, log_var, code_output, mu_prior, log_var_prior = train_inst.train(training_steps=train_inst.training_steps,batch_size=train_inst.batch_size,
                        beta_factor=train_inst.beta_factor, dec_type=train_inst.dec_type)
        if not os.path.exists("saved_models/"):
            os.makedirs("saved_models/")
        torch.save(model, train_inst.save_path)

    if train_inst.generate_digit:
        train_inst.generate_digit_samples(train_inst.save_path, num_of_pics=100, digit=train_inst.digit)
    

    

if __name__ == "__main__":

    # the args-dictionary should contain all necessary arguments for the Trainer-Class
    arch_dict = {'in_out_size':784,
                'enc_hidden1': 500,
                'enc_hidden2': 500,
                'latent_code': 256,
                'dec_hidden1': 500,
                'dec_hidden2': 500,
                'label_size': 10} # 
    
    args_dict = {'train_mode': False,
                'training_steps': 10000,
                'batch_size': 100,
                'KL_weight': 5,
                'dec_type':"Bernoulli",
                'arch_dict': arch_dict,
                'learning_rate': 0.001,
                'digit': 2,
                'generate_digit': True
                }

    main()  

