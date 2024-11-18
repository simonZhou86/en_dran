# Get dataloader for MRI-CT data

# Author: Simon Zhou, last modify Nov. 11, 2022

'''
Change log: 
- Simon: file created, implement dataset loader
'''

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import skimage.io as io
from sklearn.model_selection import KFold
from PIL import Image


def normalize(im):
    im = (im-np.min(im))/(np.max(im)-np.min(im)) if np.max(im) != np.min(im) else im # to 0,1
    return im

# def denormalize(im):
    

class getIndex(Dataset):
	def __init__(self, total_len):
		self.total_len = total_len
	
	def __len__(self):
		return self.total_len
	
	def __getitem__(self, ind):
		return torch.Tensor([ind])


def get_common_file(target_dir):
    '''
    target_dir: target directory of data, for now is MRI-CT data 
    return: ct, mri file names (should be the same name and order)
    '''
    ct = os.path.join(target_dir, "SPECT") # SPECT
    mri = os.path.join(target_dir, "MRI")

    ct_file = []
    mri_file = []

    # get file name for ct images
    for file in sorted(os.listdir(ct)):
        ct_file.append(file)
    
    # get file name for mri images
    for file in sorted(os.listdir(mri)):
        mri_file.append(file)

    diff1 = [file for file in ct_file if file not in mri_file]
    diff2 = [file for file in mri_file if file not in ct_file]

    assert len(diff1) == len(diff2) == 0, "data is somehow not paired"

    return ct_file, mri_file


def load_data(file, target_dir, test_num):
    '''
    file: list of file names (for ct, mri)
    target_dir: file directory
    test_num: number of test data
    return: torch .pt file store ct and mri
    '''

    test_ind = np.random.choice(len(file), size=test_num, replace = False)
    print(test_ind)
    test = []
    for ind in test_ind:
        test.append(file[ind])
    
    #print(test)
    
    HEIGHT = 256
    WIDTH = 256

    # 1 channel image, with shape 256x256
    data_ct = torch.empty(0, 1, HEIGHT, WIDTH)
    data_mri = torch.empty(0, 1, HEIGHT, WIDTH)
    data_ct_t = torch.empty(0, 1, HEIGHT, WIDTH)
    data_mri_t = torch.empty(0, 1, HEIGHT, WIDTH)
    
    for f in file:
        # read data and normalize
        img_ct = io.imread(os.path.join(target_dir, "CT", f)).astype(np.float32) / 255.
        img_mri = io.imread(os.path.join(target_dir, "MRI", f)).astype(np.float32) / 255.
        img_ct = torch.from_numpy(img_ct)
        img_mri = torch.from_numpy(img_mri)
        img_ct = img_ct.unsqueeze(0).unsqueeze(0) # change shape to (1, 1, 256, 256)
        img_mri = img_mri.unsqueeze(0).unsqueeze(0)

        if f not in test:
            data_ct = torch.cat((data_ct, img_ct), dim = 0)
            data_mri = torch.cat((data_mri, img_mri), dim = 0)
        else:
            data_ct_t = torch.cat((data_ct_t, img_ct), dim = 0)
            data_mri_t = torch.cat((data_mri_t, img_mri), dim = 0)
    
    return data_ct, data_mri, data_ct_t, data_mri_t


def load_data2(file, target_dir, test_f):
    '''
    file: list of file names (for ct, mri)
    target_dir: file directory
    test_f: files for test set
    return: torch .pt file store ct and mri
    '''

    #print(test)
    
    HEIGHT = 256
    WIDTH = 256

    # 1 channel image, with shape 256x256
    data_ct = torch.empty(0, 1, HEIGHT, WIDTH)
    data_mri = torch.empty(0, 1, HEIGHT, WIDTH)
    data_ct_t = torch.empty(0, 1, HEIGHT, WIDTH)
    data_mri_t = torch.empty(0, 1, HEIGHT, WIDTH)
    
    for f in file:
        # read data and normalize
        img_ct = io.imread(os.path.join(target_dir, "CT", f)).astype(np.float32)
        img_ct = normalize(img_ct)
        # print(np.min(img_ct), np.max(img_ct))
        
        img_mri = io.imread(os.path.join(target_dir, "MRI", f)).astype(np.float32)
        img_mri = normalize(img_mri)
        # print(np.min(img_mri), np.max(img_mri))
        
        img_ct = torch.from_numpy(img_ct)
        img_mri = torch.from_numpy(img_mri)
        img_ct = img_ct.unsqueeze(0).unsqueeze(0) # change shape to (1, 1, 256, 256)
        img_mri = img_mri.unsqueeze(0).unsqueeze(0)

        if f not in test_f:
            data_ct = torch.cat((data_ct, img_ct), dim = 0)
            data_mri = torch.cat((data_mri, img_mri), dim = 0)
        else:
            data_ct_t = torch.cat((data_ct_t, img_ct), dim = 0)
            data_mri_t = torch.cat((data_mri_t, img_mri), dim = 0)
    
    return data_ct, data_mri, data_ct_t, data_mri_t


def load_data_MRSPECT(file, target_dir, test_f):
    '''
    file: list of file names (for ct, mri)
    target_dir: file directory
    test_f: files for test set
    return: torch .pt file store ct and mri
    '''

    #print(test)
    
    HEIGHT = 256
    WIDTH = 256

    # 1 channel image, with shape 256x256
    data_sp = torch.empty(0, 1, HEIGHT, WIDTH)
    data_mri = torch.empty(0, 1, HEIGHT, WIDTH)
    data_sp_t = torch.empty(0, 3, HEIGHT, WIDTH)
    data_mri_t = torch.empty(0, 1, HEIGHT, WIDTH)
    
    max_mri = -int(1e9)
    min_mri = int(1e9)
    max_sp = -int(1e9)
    min_sp = int(1e9)
    for f in file:
        # read data and normalize
        img_mri = io.imread(os.path.join(target_dir, "MRI", f)).astype(np.float32)
        if np.max(img_mri) > max_mri:
            max_mri =  np.max(img_mri)
        if np.min(img_mri) < min_mri:
            min_mri = np.min(img_mri)
        # print(np.min(img_mri), np.max(img_mri))
        img_mri = normalize(img_mri)
        # print(np.min(img_mri), np.max(img_mri))
        
        img_sp = Image.open(os.path.join(target_dir, "SPECT", f)).convert('YCbCr') #.astype(np.float32)
        if np.max(np.array(img_sp)) > max_sp:
            max_sp = np.max(np.array(img_sp))
        if np.min(np.array(img_sp)) < min_sp:
            min_sp = np.min(np.array(img_sp))        
        # print("sp min max",np.min(np.array(img_sp)), np.max(np.array(img_sp)))
        y0, cb0, cr0 = img_sp.split()
        y0 = np.array(y0).astype(np.float32)
        img_sp = normalize(y0)
        # print(np.min(img_mri), np.max(img_mri))
        cb0 = torch.from_numpy(np.array(cb0))
        cr0 = torch.from_numpy(np.array(cr0))
        img_sp = torch.from_numpy(img_sp)
        img_mri = torch.from_numpy(img_mri)
        img_sp = img_sp.unsqueeze(0).unsqueeze(0) # change shape to (1, 1, 256, 256)
        img_mri = img_mri.unsqueeze(0).unsqueeze(0)

        if f not in test_f:
            data_sp = torch.cat((data_sp, img_sp), dim = 0)
            data_mri = torch.cat((data_mri, img_mri), dim = 0)
        else:
            cb0 = cb0.unsqueeze(0).unsqueeze(0)
            cr0 = cr0.unsqueeze(0).unsqueeze(0)
            all_for_sp_t = torch.cat((img_sp, cb0, cr0), dim = 1)
            data_sp_t = torch.cat((data_sp_t, all_for_sp_t), dim = 0)
            data_mri_t = torch.cat((data_mri_t, img_mri), dim = 0)
        
    print(f"mri max {max_mri}, mri min {min_mri}, sp max {max_sp}, sp min {min_sp}")
    return data_sp, data_mri, data_sp_t, data_mri_t


def get_loader(ct, mri, tv_ratio, bs):
    '''
    ct: ct data
    mri: mri data
    tv_ratio: train & validation ratio
    bs: batch size
    return: Dataloader class for train and val
    '''
    assert ct.shape[0] == mri.shape[0], "two datasets do not have the same length? whats wrong"
    total_len = ct.shape[0] + mri.shape[0]
    n_train = int(tv_ratio * total_len)

    train_set, val_set = random_split(getIndex(total_len), lengths=(n_train, total_len - n_train))
    train_loader = DataLoader(train_set, batch_size=bs, num_workers=0, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=bs, num_workers=0, shuffle=False, drop_last=False)
    return train_loader, val_loader

def get_loader2(ct, mri, tv_ratio, bs):
    '''
    ct: ct data
    mri: mri data
    tv_ratio: train & validation ratio
    bs: batch size
    return: Dataloader class for train and val
    '''
    assert ct.shape[0] == mri.shape[0], "two datasets do not have the same length? whats wrong"
    total_len = ct.shape[0]
    n_train = int(tv_ratio * total_len)

    train_set, val_set = random_split(getIndex(total_len), lengths=(n_train, total_len - n_train))
    train_loader = DataLoader(train_set, batch_size=bs, num_workers=0, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=bs, num_workers=0, shuffle=False, drop_last=False)
    return train_loader, val_loader


def get_cv_dataset(n, fold, shuffle=False):
    X = np.arange(n)
    kfold = KFold(n_splits=fold, shuffle=shuffle, random_state=42)  ## kfold为KFolf类的一个对象
    return_res = []
    for a, b in kfold.split(X):
        fold_train = []
        fold_val = []
        for i in a:
            # store index
            fold_train.append(i)
        for j in b:
            fold_val.append(j)
        return_res.append({"train_data_ind": fold_train, "val_data_ind": fold_val})

    return return_res

# if __name__ == "__main__":
#     target_dir = "./CT-MRI/"
#     ct, mri = get_common_file(target_dir)
#     train_ct, train_mri, test_ct, test_mri = load_data(ct, target_dir, 20)
#     print(train_ct.shape, train_mri.shape, test_ct.shape, test_mri.shape)
#     train_loader, val_loader = get_loader(train_ct, train_mri, 0.8, 16)
#     print(len(train_loader), len(val_loader))