import numpy as np
import pandas as pd
import os, math, sys
import glob, itertools
import argparse, random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg19
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid

import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split

from models import *
from dataset import *

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--decay_epoch', type=int, default=4)
    parser.add_argument('--n_cpu', type=int, default=4)
    parser.add_argument('--hr_height', type=int, default=256)
    parser.add_argument('--hr_width', type=int, default=256)
    parser.add_argument('--channels', type=int, default=3)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_args()
    
    # number of epochs of training
    n_epochs = args.n_epochs
    # name of the dataset
    dataset_path = args.dataset_path
    # size of the batches
    batch_size = args.batch_size
    # learning rate
    lr = args.lr
    # epoch from which to start lr decay
    decay_epoch = args.decay_epoch
    # number of cpu threads to use during batch generation
    n_cpu = args.n_cpu
    # high res. image height
    hr_height = args.hr_height
    # high res. image width
    hr_width = args.hr_width
    # number of image channels
    channels = args.channels
    # adam: decay of first order momentum of gradient
    b1 = 0.5
    # adam: decay of second order momentum of gradient
    b2 = 0.999

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    cuda = torch.cuda.is_available()
    hr_shape = (hr_height, hr_width)

    train_paths, test_paths = train_test_split(sorted(glob.glob(dataset_path + "/*.*")), test_size=0.02, random_state=42)
    train_dataloader = DataLoader(ImageDataset(train_paths, hr_shape=hr_shape), batch_size=batch_size, shuffle=True, num_workers=n_cpu)
    test_dataloader = DataLoader(ImageDataset(test_paths, hr_shape=hr_shape), batch_size=int(batch_size*0.75), shuffle=True, num_workers=n_cpu)

    # Initialize generator and discriminator
    generator = GeneratorResNet()
    discriminator = Discriminator(input_shape=(channels, *hr_shape))
    feature_extractor = FeatureExtractor()

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        feature_extractor = feature_extractor.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_content = criterion_content.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    train_gen_losses, train_disc_losses, train_counter = [], [], []
    test_gen_losses, test_disc_losses = [], []
    test_counter = [idx*len(train_dataloader.dataset) for idx in range(1, n_epochs+1)]

    for epoch in range(n_epochs):

        ### Training
        gen_loss, disc_loss = 0, 0
        #tqdm_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch} ', total=int(len(train_dataloader)))
        for batch_idx, imgs in enumerate(train_dataloader):
            generator.train(); discriminator.train()
            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            print(imgs_lr.shape)
            imgs_hr = Variable(imgs["hr"].type(Tensor))
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            
            ### Train Generator
            optimizer_G.zero_grad()
            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)
            # Adversarial loss
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())
            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN
            loss_G.backward()
            optimizer_G.step()

            ### Train Discriminator
            optimizer_D.zero_grad()
            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
            # Total loss
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            gen_loss += loss_G.item()
            train_gen_losses.append(loss_G.item())
            disc_loss += loss_D.item()
            train_disc_losses.append(loss_D.item())
            train_counter.append(batch_idx*batch_size + imgs_lr.size(0) + epoch*len(train_dataloader.dataset))
            #tqdm_bar.set_postfix(gen_loss=gen_loss/(batch_idx+1), disc_loss=disc_loss/(batch_idx+1))
            print('gen_loss', gen_loss/(batch_idx+1), 'disc_loss', disc_loss/(batch_idx+1), 'batch', (batch_idx+1)/(len(train_dataloader)))
    
        # Testing
        print('\nTest')
        gen_loss, disc_loss = 0, 0
        #tqdm_bar = tqdm(test_dataloader, desc=f'Testing Epoch {epoch} ', total=int(len(test_dataloader)))
        for batch_idx, imgs in enumerate(test_dataloader):
            generator.eval(); discriminator.eval()
            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            
            ### Eval Generator
            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)
            # Adversarial loss
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())
            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN

            ### Eval Discriminator
            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            gen_loss += loss_G.item()
            disc_loss += loss_D.item()
            #tqdm_bar.set_postfix(gen_loss=gen_loss/(batch_idx+1), disc_loss=disc_loss/(batch_idx+1))
            
            # Save image grid with upsampled inputs and SRGAN outputs
            if random.uniform(0,1)<0.1:
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
                gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
                imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                img_grid = torch.cat((imgs_hr, imgs_lr, gen_hr), -1)
                save_image(img_grid, f"images/{batch_idx}.png", normalize=False)
                
        print('gen_loss', gen_loss/len(test_dataloader), 'disc_loss', disc_loss/len(test_dataloader), '\n')

        test_gen_losses.append(gen_loss/len(test_dataloader))
        test_disc_losses.append(disc_loss/len(test_dataloader))
        
        # Save model checkpoints
        if np.argmin(test_gen_losses) == len(test_gen_losses)-1:
            torch.save(generator.state_dict(), "saved_models/generator.pth")
            torch.save(discriminator.state_dict(), "saved_models/discriminator.pth")
            
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_counter, y=train_gen_losses, mode='lines', name='Train Generator Loss'))
    fig.add_trace(go.Scatter(x=test_counter, y=test_gen_losses, marker_symbol='star-diamond', 
                            marker_color='orange', marker_line_width=1, marker_size=9, mode='markers', name='Test Generator Loss'))
    fig.update_layout(
        width=1000,
        height=500,
        title="Train vs. Test Generator Loss",
        xaxis_title="Number of training examples seen",
        yaxis_title="Adversarial + Content Loss"),
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_counter, y=train_disc_losses, mode='lines', name='Train Discriminator Loss'))
    fig.add_trace(go.Scatter(x=test_counter, y=test_disc_losses, marker_symbol='star-diamond', 
                            marker_color='orange', marker_line_width=1, marker_size=9, mode='markers', name='Test Discriminator Loss'))
    fig.update_layout(
        width=1000,
        height=500,
        title="Train vs. Test Discriminator Loss",
        xaxis_title="Number of training examples seen",
        yaxis_title="Adversarial Loss"),
    fig.show()