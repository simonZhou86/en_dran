# inference fused image

import os
import argparse
import torch
import torch.nn as nn
import sys
from torchmetrics import PeakSignalNoiseRatio
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from our_utils import *
from eval import psnr, ssim, mutual_information
from evaluation_metrics import fsim, nmi, en
import skimage.io as io
from model_edge_enhance import *
import time
# import sys
# sys.path.append("./model")

parser = argparse.ArgumentParser(description='Inference Fused Image configs')
parser.add_argument('--test_folder', type=str, default='./testset', help='input test image')
parser.add_argument('--model', type=str, default='./res/pretrained_models/model_v5/last.pt', help='which model to use')
parser.add_argument('--save_folder', type=str, default='./res/fused_image', help='input image to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda', default='true')

opt = parser.parse_args()

########### gpu ###############
device = torch.device("cuda:0" if opt.cuda else "cpu")
###############################

######### make dirs ############
# save_dir = os.path.join(opt.save_folder, "model_vf_sfnnMean")
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
###############################

####### loading pretrained model ########
model = fullModel().to(device)
model.load_state_dict(torch.load(opt.model))
model.eval()
#########################################
########### loading test set ###########
test_sp = torch.load(os.path.join(opt.test_folder, 'ct_test.pt')).to(device)
test_mri = torch.load(os.path.join(opt.test_folder, 'mri_test.pt')).to(device)
########################################

def process(out, cb, cr):
    out_img_y = out
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    cb = cb.clip(0,255)
    cr = cr.clip(0,255)
    # print(out_img_y.shape, cb.shape, cr.shape)
    out_img_y = Image.fromarray(np.uint8(out_img_y), mode='L')
    out_img_cb = Image.fromarray(np.uint8(cb), mode = "L")
    out_img_cr = Image.fromarray(np.uint8(cr), mode = "L")
    # out_img_cb = cb#cb.resize(out_img_y.size, Image.BICUBIC)
    # out_img_cr = cr#cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    return out_img

psnr = PeakSignalNoiseRatio()
psnrs = []
ssims = []
nmis = []
mis = []
fsims = []
ens = []
start = time.time()
for slice in range(test_sp.shape[0]):
    mri_slice = test_mri[slice,:,:,:]
    mri_slice_inf = test_mri[slice,:,:,:].unsqueeze(0)
    
    sp_slice = test_sp[slice,0,:,:]
    sp_slice_inf = test_sp[slice,0,:,:].unsqueeze(0).unsqueeze(0)
    cb0 = test_sp[slice,1,:,:]
    cr0 = test_sp[slice,2,:,:]
    sp_slice = sp_slice.detach().cpu().numpy()
    cb0 = cb0.detach().cpu().numpy()
    cr0 = cr0.detach().cpu().numpy()
    out = process(sp_slice, cb0, cr0)
    mri_slice = mri_slice.squeeze(0).squeeze(0).detach().cpu()
    # store original image
    out.save(f"{opt.save_folder}/SPECT_{slice}.jpg")
    io.imsave(f"{opt.save_folder}/mri_{slice}.jpg", (mri_slice.numpy() * 255).astype(np.uint8))

    with torch.no_grad():
        sp_fe = model.fe(sp_slice_inf)
        #print(ct_fe.shape)
        mri_fe = model.fe(mri_slice_inf)

        fused = fusion_strategy(sp_fe, mri_fe, device, "SFNN")
        #fused = torch.maximum(ct_fe, mri_fe)
        final = model.recon(fused)
    # final = model(ct_slice, mri_slice)
    #print(final.squeeze(0).squeeze(0))
    #ct_fe = model.fe(ct_slice)
    #print(ct_fe.shape)
    # mri_fe = model.fe(mri_slice)

    #fused = fusion_strategy(ct_fe, mri_fe, device, "SFNN")
    #fused = torch.maximum(ct_fe, mri_fe)
    #final = model.recon(fused)
    # final = model(ct_slice, mri_slice)
    #print(final.squeeze(0).squeeze(0))
    final = final.squeeze(0).squeeze(0).detach().cpu().clamp(min=0, max=1)
    # print(torch.max(final), torch.min(final))
    out_f = process(final, cb0, cr0)
    out_f.save(f"{opt.save_folder}/fuse_{slice}.jpg")
 
print(f"time used: {(time.time()-start)/len(ens)}")