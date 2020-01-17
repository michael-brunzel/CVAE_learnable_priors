## **Conditional Variational Autoencoder (CVAE) with learnable priors**

### **Loss Function of a Variational Autoencoder**

<img src="pics/VAE_Loss.PNG" width="500">
<!--[ELBO](pics/VAE_Loss.PNG)-->

### **Loss Function of CVAE with learnable priors, *whereby c denotes a one-hot label***

<img src="pics/CVAE_learnable_prior_Loss.PNG" width="500">
<!--[ELBO_modified](pics/CVAE_learnable_prior_Loss.PNG)-->

### **Examples for conditionally sampled digits (*default settings in main.py were used*)**

![Digit 9](pics/CVAE_MNIST_9.png)
![Digit 8](pics/CVAE_MNIST_8.png)
![Digit 3](pics/CVAE_MNIST_3.png)

### One example for mixed sampling from all digit-classes ###
![Digit mixed](pics/CVAE_MNIST_mixed.png)



