# Pytorch-GAN-MNIST-FashionMNIST-USPS
Pytorch implementation of GAN for MNIST, FashionMNIST and USPS dataset.

###Generator
FC(10)&#x2192;FC(1024)&#x2192;FC(1024)&#x2192;FC(IMG)
###Discriminator
FC(IMG)&#x2192;FC(256)&#x2192;FC(256)&#x2192;FC(1)

<br>
Change the DB variable to change the dataset.

For using the saved model to generate images, set LOAD_MODEL to True and EPOCHS to 0.


## Generated Samples
#### MNIST
<img src="/results/MNIST.png" width="900"></img>
#### FashionMNIST
<img src="/results/FashionMNIST.png" width="900"></img>
#### USPS
<img src="/results/USPS.png" width="900"></img>
