from torch import optim
import os
import torchvision.utils as vutils
import numpy as np
import math
import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F


# Arguments
BATCH_SIZE = 256
EPOCHS = 500
Z_DIM = 10
LOAD_MODEL = False
CHANNELS = 1
DB = 'MNIST' # MNIST | FashionMNIST | USPS 


if DB == 'MNIST' or DB == 'FashionMNIST':
    IMAGE_SIZE = 28
elif DB == 'USPS':
    IMAGE_SIZE = 16
else:
    print("Incorrect dataset")
    exit(0)


# Directories for storing model and output samples
model_path = os.path.join('./model', DB)
if not os.path.exists(model_path):
    os.makedirs(model_path)
samples_path = os.path.join('./samples', DB)
if not os.path.exists(samples_path):
    os.makedirs(samples_path)


# Method for storing generated images
def generate_imgs(z, epoch=0):
    gen.eval()
    fake_imgs = gen(z)
    fake_imgs_ = vutils.make_grid(fake_imgs, normalize=True, nrow=math.ceil(BATCH_SIZE ** 0.5))
    vutils.save_image(fake_imgs_, os.path.join(samples_path, 'sample_' + str(epoch) + '.png'))


# Data loaders
mean = np.array([0.5])
std = np.array([0.5])
transform = transforms.Compose([transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

if DB=='MNIST':
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
elif DB=='FashionMNIST':
    dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
elif DB=='USPS':
    dataset = datasets.USPS('./data', train=True, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)


# Networks
class Generator(nn.Module):
    def __init__(self, z_dim=10, image_size=28, channels=1, h_dim=1024):
        super(Generator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, image_size * image_size * channels)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        x = x.reshape([-1, self.channels, self.image_size, self.image_size])
        return x


class Discriminator(nn.Module):
    def __init__(self, image_size=28, channels=1, h_dim=256):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(image_size * image_size * channels, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 1)

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc3(x)
        x = torch.sigmoid(x).squeeze()
        return x


gen = Generator(z_dim=Z_DIM, image_size=IMAGE_SIZE, channels=CHANNELS)
dis = Discriminator(image_size=IMAGE_SIZE, channels=CHANNELS)

# Load previous model   
if LOAD_MODEL:
    gen.load_state_dict(torch.load(os.path.join(model_path, 'gen.pkl')))
    dis.load_state_dict(torch.load(os.path.join(model_path, 'dis.pkl')))

# Model Summary
print("------------------Generator------------------")
print(gen)
print("------------------Discriminator------------------")
print(dis)

# Define Optimizers
g_opt = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=2e-5)
d_opt = optim.Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=2e-5)

# Loss functions
loss_fn = nn.BCELoss()

# Fix images for viz
fixed_z = torch.randn(BATCH_SIZE, Z_DIM)

# Labels
real_label = torch.ones(BATCH_SIZE)
fake_label = torch.zeros(BATCH_SIZE)

# GPU Compatibility
is_cuda = torch.cuda.is_available()
if is_cuda:
    gen, dis = gen.cuda(), dis.cuda()
    real_label, fake_label = real_label.cuda(), fake_label.cuda()
    fixed_z = fixed_z.cuda()

total_iters = 0
max_iter = len(data_loader)

# Training
for epoch in range(EPOCHS):
    gen.train()
    dis.train()

    for i, data in enumerate(data_loader):

        total_iters += 1

        # Loading data
        x_real, _ = data
        z_fake = torch.randn(BATCH_SIZE, Z_DIM)

        if is_cuda:
            x_real = x_real.cuda()
            z_fake = z_fake.cuda()

        # Generate fake data
        x_fake = gen(z_fake)

        # Train Discriminator
        fake_out = dis(x_fake.detach())
        real_out = dis(x_real.detach())
        d_loss = (loss_fn(fake_out, fake_label) + loss_fn(real_out, real_label)) / 2

        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # Train Generator
        fake_out = dis(x_fake)
        g_loss = loss_fn(fake_out, real_label)

        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        if i % 50 == 0:
            print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS)
                  + "\titer: " + str(i) + "/" + str(max_iter)
                  + "\ttotal_iters: " + str(total_iters)
                  + "\td_loss:" + str(round(d_loss.item(), 4))
                  + "\tg_loss:" + str(round(g_loss.item(), 4))
                  )

    if (epoch+1) % 10 == 0:
        torch.save(gen.state_dict(), os.path.join(model_path, 'gen.pkl'))
        torch.save(dis.state_dict(), os.path.join(model_path, 'dis.pkl'))

        generate_imgs(fixed_z, epoch=epoch + 1)

generate_imgs(fixed_z)
