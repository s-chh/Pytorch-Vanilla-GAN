# Pytorch-Vanilla-GAN
Pytorch implementation of Vanilla-GAN for MNIST, FashionMNIST and USPS dataset.

### Generator
FC(10)&#x2192;FC(1024)&#x2192;FC(1024)&#x2192;FC(Image_Size)
### Discriminator
FC(Image_Size)&#x2192;FC(256)&#x2192;FC(256)&#x2192;FC(1)

<br>
Change the DB variable to change the dataset.

For using the saved model to generate images, set LOAD_MODEL to True and EPOCHS to 0.


## Generated Samples
#### MNIST
<img src="/Results/MNIST.png" width="500"></img>
#### FashionMNIST
<img src="/Results/FashionMNIST.png" width="500"></img>
#### USPS
<img src="/Results/USPS.png" width="500"></img>
