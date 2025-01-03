# Training script for the project
# Author: Simon Zhou, last modify Nov. 18, 2022

'''
Change log:
-Simon: file created, write some training code
-Simon: refine training script
-Reacher: train v3
'''

import argparse
import os
import sys

sys.path.append("../")
from tqdm import trange

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16_bn

import meta_config as config
#from model_v5 import *
from model_edge_enhance import *
from our_utils import *
from dataset_loader import *
from loss import *
from val import validate

parser = argparse.ArgumentParser(description='parameters for the training script')
parser.add_argument('--dataset', type=str, default="CT-MRI",
                    help="which dataset to use, available option: CT-MRI, MRI-PET, MRI-SPECT")
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for training')
parser.add_argument('--lr_decay', type=bool, default=False, help='decay learing rate?')
parser.add_argument('--accum_batch', type=int, default=1, help='number of batches for gradient accumulation')
parser.add_argument('--lambda1', type=float, default=0.5, help='weight for image gradient loss')
parser.add_argument('--lambda2', type=float, default=0.5, help='weight for perceptual loss')
# parser.add_argument('--checkpoint', type=str, default='./model', help='Path to checkpoint')
parser.add_argument('--cuda', action='store_true', help='whether to use cuda', default=True)
parser.add_argument('--seed', type=int, default=3407, help='random seed to use')
parser.add_argument('--base_loss', type=str, default='l1_charbonnier',
                    help='which loss function to use for pixel-level (l2 or l1 charbonnier)')
parser.add_argument('--val_every', type=int, default=20,
                    help='run validation for every val_every epochs')

opt = parser.parse_args()

######### whether to use cuda ####################
device = torch.device("cuda:0" if opt.cuda else "cpu")
#################################################

########## seeding ##############
seed_val = opt.seed
random_seed(seed_val, opt.cuda)
################################


model = fullModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

if opt.lr_decay:
    stepLR = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min = 0.0000003)

##### downloading pretrained vgg model ##################
vgg = vgg16_bn(pretrained=True)
########################################################

# gradient accumulation for small batch
NUM_ACCUMULATION_STEPS = opt.accum_batch

NUM_EXP = 3
####### loading dataset ####################################
target_dir = os.path.join(config.data_dir, opt.dataset)
ct, mri = get_common_file(target_dir)
ct_left = ct.copy()
tsize = 0
if "SPECT" in opt.dataset:
    tsize = 50
else:
    tsize = 30
for exp in range(NUM_EXP):
    test_ind = np.random.choice(len(ct_left), size=tsize, replace = False)
    # print(test_ind)
    test = []
    for ind in test_ind:
        test.append(ct_left[ind])
    for fn in test:
        ct_left.remove(fn)
    print(f"ct_left len: {len(ct_left)}")
    if "SPECT" in opt.dataset:
        train_sp, train_mri, test_sp, test_mri = load_data_MRSPECT(ct, target_dir, test) #load_data_MRSPECT
    else:
        train_sp, train_mri, test_sp, test_mri = load_data2(ct, target_dir, test)

    # change this
    fold_path = f"./res/MRISPECT/exp_{exp}_new_abl"
    os.makedirs(fold_path, exist_ok=True)
    model_dir = f"./res/MRISPECT/exp_{exp}_new_abl/pretrained_models"
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(test_sp, os.path.join(fold_path, "sp_test.pt"))
    torch.save(test_mri, os.path.join(fold_path, "mri_test.pt"))
    print(train_sp.shape, train_mri.shape, test_sp.shape, test_mri.shape)
    assert test_sp.shape[0] != 0 or test_mri.shape[0] != 0, "empty test set!"
    train_total = torch.cat((train_sp, train_mri), dim=0).to(device)

    train_loader, val_loader = get_loader(train_sp, train_mri, config.train_val_ratio, opt.batch_size)

    train_loss = []
    val_loss = []
    t = trange(opt.epochs, desc='Training progress...', leave=True)
    lowest_val_loss = int(1e9)
    best_ssim = 0

    for i in t:
        print("\n new epoch {} starts for exp {}!".format(i, exp))
        # clear gradient in model
        model.zero_grad()
        b_loss = 0
        # train model
        model.train()
        for j, batch_idx in enumerate(train_loader):
            # clear gradient in optimizer
            optimizer.zero_grad()
            batch_idx = batch_idx.view(-1).long()
            img = train_total[batch_idx]
            img_out = model(img)
            # compute loss
            loss, _, _, _ = loss_func2(vgg, img_out, img, opt.lambda1, opt.lambda2, config.block_idx, device)
            # back propagate and update weights
            # print("batch reg, grad, percep loss: ", reg_loss.item(), img_grad.item(), percep.item())
            # loss = loss / NUM_ACCUMULATION_STEPS
            loss.backward()

            # if ((j + 1) % NUM_ACCUMULATION_STEPS == 0) or (j + 1 == len(train_loader)):
            optimizer.step()
            b_loss += loss.item()
            # wandb.log({"loss": loss})

        # store loss
        ave_loss = b_loss / len(train_loader)
        train_loss.append(ave_loss)
        print("epoch {}, training loss is: {}".format(i, ave_loss))

        # validation
        val_loss = []
        val_display_img = []
        with torch.no_grad():
            b_loss = 0
            # eval model, unable update weights
            model.eval()
            for k, batch_idx in enumerate(val_loader):
                batch_idx = batch_idx.view(-1).long()
                val_img = train_total[batch_idx]
                val_img_out = model(val_img)
                # display first image to visualize, this can be changed
                loss, _, _, _ = loss_func2(vgg, img_out, img, opt.lambda1, opt.lambda2, config.block_idx, device)
                b_loss += loss.item()

        ave_val_loss = b_loss / len(val_loader)
        val_loss.append(ave_val_loss)
        print("epoch {}, validation loss is: {}".format(i, ave_val_loss))

        # save model
        if ave_val_loss < lowest_val_loss:
            torch.save(model.state_dict(), model_dir + "/model_at_{}.pt".format(i))
            lowest_val_loss = ave_val_loss
            print("model is saved in epoch {}".format(i))

            # Evaluate during training
            # Save the current model
            torch.save(model.state_dict(), model_dir + "/current.pt")
    
            # test_mri = 
            val_psnr, val_ssim, val_nmi, val_mi, val_fsim, val_en = validate(opt.dataset, model_dir + "/current.pt", test_sp.cuda(), test_mri.cuda(), exp, i)
    
            print("PSNR", "SSIM", "NMI", "MI", "FSIM", "Entropy")
            print(val_psnr, val_ssim, val_nmi, val_mi, val_fsim, val_en)
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                print(f"ヾ(◍°∇°◍)ﾉﾞ New best SSIM = {best_ssim}")
                # overwrite
                torch.save(model.state_dict(), model_dir + "/best.pt".format(i))

        if i == opt.epochs - 1:
            torch.save(model.state_dict(), model_dir + "/last.pt".format(i))

        # lr decay update
        if opt.lr_decay:
            stepLR.step()
        
        os.remove(model_dir + "/current.pt")
        torch.cuda.empty_cache()
    ########################################s
