import argparse
import os
import numpy as np
import time
import datetime
import sys

import torch
from torch.autograd import Variable

from models import Create_nets
from datasets import get_dataloaders
from options import TrainOptions
from optimizer import *
from utils import sample_images , LambdaLR
import json



#load the args
args = TrainOptions().parse()
# Calculate output of image discriminator (PatchGAN)
D_out_size = 256//(2**args.n_D_layers) - 2
print(D_out_size)
patch = (1, D_out_size, D_out_size)

# Initialize generator and discriminator
generator, discriminator = Create_nets(args)
# Loss functions
criterion_GAN, criterion_pixelwise = Get_loss_func(args)
# Optimizers
optimizer_G, optimizer_D = Get_optimizers(args, generator, discriminator)
direction = args.which_direction
# Configure dataloaders
train_dataloader,test_dataloader,_ = get_dataloaders(direction)


# ----------
#  Training
# ----------
prev_time = time.time()
#Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
G_loss = []
D_loss = []
for epoch in range(args.epoch_start, args.epoch_num):
    
    for i, batch in enumerate(train_dataloader):

        # Model inputs
        real_A = Variable(batch['A'].type(torch.FloatTensor).cuda())
        real_B = Variable(batch['B'].type(torch.FloatTensor).cuda())

        # Adversarial ground truths
        valid = Variable(torch.FloatTensor(np.ones((real_A.size(0), *patch))).cuda(), requires_grad=False)
        fake = Variable(torch.FloatTensor(np.zeros((real_A.size(0), *patch))).cuda(), requires_grad=False)

        # Update learning rates
        #lr_scheduler_G.step(epoch)
        #lr_scheduler_D.step(epoch)
        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()

        #loss
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A) #changed this to real_B from real_A so that discriminator generates better photoshopped images? Because it currentlyjust regenerated A
        #print("pred_fake: ",pred_fake.size(),"valid: ", valid.size())
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)

        # Total loss
        loss_G = loss_GAN + args.lambda_pixel * loss_pixel # will change args.lambda value to 1000 and then 10000 to see if it will more heavily weight trying to be closer to real_b than be closer to A - update has not changed output
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A) #changed this to real_B from real_A so that discriminator generates better photoshopped images? Because it currently just regenerated A
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------
        # Determine approximate time left
        batches_done = epoch * len(train_dataloader) + i
        batches_left = args.epoch_num * len(train_dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        
        sys.stdout.write("\r[Epoch%d/%d]-[Batch%d/%d]-[Dloss:%f]-[Gloss:%f, loss_pixel:%f, adv:%f] ETA:%s" %
                                                        (epoch+1, args.epoch_num,
                                                        i, len(train_dataloader),
                                                        loss_D.data.cpu(), loss_G.data.cpu(),
                                                        loss_pixel.data.cpu(), loss_GAN.data.cpu(),
                                                        time_left))

        # If at sample interval save image
        if batches_done % args.sample_interval == 0:
            sample_images(generator, test_dataloader, args, epoch, batches_done)
    
    G_loss.append(loss_G.data.cpu().numpy().tolist())
    D_loss.append(loss_D.data.cpu().numpy().tolist())
    if epoch == 50:
        
        dic={'G_loss':G_loss, 'D_loss':D_loss}
        with open("./{}.json".format(args.exp_name), 'w') as json_file:  
            json.dump(dic, json_file)
        break
    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        dirname = '%s/%s'%(args.model_result_dir,args.dataset_name)
        # Save model checkpoints
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(generator.state_dict(), '%s/generator_%d.pt' % (dirname, epoch))
        torch.save(discriminator.state_dict(), '%s/discriminator_%d.pt' % (dirname, epoch))
