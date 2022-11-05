"""Code for computation of PLCC, SRCC and KRCC between
    PIQ metrics predictions and ground truth scores from MOS databases.
"""
import piq
import tqdm
import torch
import torch.nn.functional as F
import argparse
import functools
import torchvision

import pandas as pd
import numpy as np
from scipy import stats

from typing import List, Callable, Tuple
from pathlib import Path
import pathlib
from skimage.io import imread
from skimage.util import img_as_float32
from scipy.stats import spearmanr, kendalltau
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from torch import nn
from itertools import chain
import os
import sys
root_dir = str(pathlib.Path(__file__).parent / '..')
sys.path.insert(0, root_dir)

from pipelines.landmark_model import LandmarkModel
from facenet_pytorch import InceptionResnetV1



## load images from directory
class DATASET(Dataset):
    def __init__(self, path):
        self.path = path
        self.frames = os.listdir(self.path)
        
    def get_path(self, index):
        path = os.path.join(self.path, self.frames[index])
        return path
    
    def __getitem__(self, index):
        frame = img_as_float32(imread(self.get_path(index)))
        return {'images': frame.transpose(2, 0, 1)}
    
    def __len__(self):
        return len(self.frames)
    
class MetricEvaluater():
    def __init__(self, config, landmark_model=None):
        self.config = config
        self.dataset = DATASET
        self.dataloader = DataLoader
        self.landmark_model = LandmarkModel(config.config.common.checkpoints.landmark_model.dir) if landmark_model is None else landmark_model
        self.inception = InceptionResnetV1(pretrained='vggface2').eval().to('cuda:0')
        self.openface_dir = '/home/server19/minyeong_workspace/OpenFace/build/bin'
        self.cache = {}
        
    def openface_extract_feature_from_dir(self, x, save_dir):
        featurer = os.path.join(self.openface_dir, 'FeatureExtraction')
        out_name = os.path.join(save_dir, os.path.basename(x) + '.csv')
        if os.path.exists(out_name):
            return out_name
        
        cmd = f'{featurer} -fdir {x} -out_dir {save_dir};rm -rf '
        # os.makedir(os.path.join(x, '..', 'openface'))
        os.system(cmd)
        
        aligned_path = os.path.join(save_dir, os.path.basename(x) + '_aligned')
        cmd = f'rm -rf {aligned_path}'
        os.system(cmd)
        
        return out_name

    def load_csv_as_dict(self, name):
        arr = np.loadtxt(name, delimiter=',', dtype=str)
        d = {line[0]: line[1:].astype(np.float32) for line in arr.T}
        return d 
    
    def load_au_c_from_csv(self, fname):
        d = self.load_csv_as_dict(fname)
        d_au = {}
        for k in list(d.keys()):
            if k.startswith('AU') and k.endswith('c'):
                d_au[k] = d[k]
        return d_au
        
    def get_dataloader(self, path, **kwargs):
        dataset = self.dataset(path)
        return self.dataloader(dataset)
    
    def get_paired_frames(self, x, y):
        # x, y: image directory
        files_x = os.listdir(x)
        files_y = os.listdir(y)
        fids_x = list(map(lambda x: int(x.split('.')[0]), files_x))
        fids_y = list(map(lambda x: int(x.split('.')[0]), files_y))
        dict_y = {k: v for (k, v) in zip(fids_y, files_y)}
        # print(f'files_x: {files_x}')
        # print(f'files_y: {files_y}')
        # while True:
        #     continue
        print(f'x: {x}')
        pairs = []
        for fid_x, file_x in zip(fids_x, files_x):
            if fid_x in dict_y:
                file_y = dict_y[fid_x]
                path_x = os.path.join(x, file_x)
                path_y = os.path.join(y, file_y)
                frame_x = img_as_float32(imread(path_x))
                frame_y = img_as_float32(imread(path_y))
                pairs.append([frame_x, frame_y])
                
        print(f'len frames: {len(pairs)}')
        pairs = torch.tensor(np.array(pairs)).permute(0, 1, 4, 2, 3)
        
        return pairs
    
    def get_frames(self, x):
        # x, y: image directory
        if x in self.cache:
            return self.cache[x]
        
        files_x = os.listdir(x)

        frames = []
        
        for file in files_x:
            path_x = os.path.join(x, file)
            frame_x = img_as_float32(imread(path_x))
            frames.append(frame_x)
        
        frames = torch.tensor(np.array(frames)).permute(0, 3, 1, 2)
        self.cache[x] = frames
        
        return frames
    
    def get_frame(self, path):
        if path in self.cache:
            return self.cache[path]
        frame = img_as_float32(imread(path))
        frame = torch.tensor(frame).unsqueeze(0).permute(0, 3, 1, 2)
        self.cache[path] = frame
        return frame
    
    def L1(self, x, y, is_path=True):
        # x: generated, y: ground truth tensor of (B x channel x H x W)
        if is_path:
            pairs = self.get_paired_frames(x, y)
            x = pairs[:, 0]
            y = pairs[:, 1]

        return torch.abs((x - y)).mean()
    
    def FID(self, x, y, is_path=True):
        metric = piq.FID()
        if is_path:
            dl_x = self.get_dataloader(x)
            dl_y = self.get_dataloader(y)
            x = metric.compute_feats(dl_x)
            y = metric.compute_feats(dl_y)
        return metric(x, y)


    def SSIM(self, x, y, is_path=True):
        metric = piq.ssim
        if is_path:
            pairs = self.get_paired_frames(x, y)
            x = pairs[:, 0]
            y = pairs[:, 1]
        return metric(x, y)

    def MS_SSIM(self, x, y, is_path=True):
        metric = piq.multi_scale_ssim
        if is_path:
            pairs = self.get_paired_frames(x, y)
            x = pairs[:, 0]
            y = pairs[:, 1]
        return metric(x, y)
    
    def LPIPS(self, x, y, is_path=True):
        metric = piq.LPIPS()
        if is_path:
            pairs = self.get_paired_frames(x, y)
            x = pairs[:, 0]
            y = pairs[:, 1]
        return metric(x, y)
    
    def PSNR(self, x, y, is_path=True):
        metric = piq.psnr
        if is_path:
            pairs = self.get_paired_frames(x, y)
            x = pairs[:, 0]
            y = pairs[:, 1]
        return metric(x, y)
    
    
    def AKD(self, x, y, is_path=True):
        # x,y : B x C x H x W images
        if is_path:
            pairs = self.get_paired_frames(x, y)
            x = pairs[:, 0]
            y = pairs[:, 1]
        bs = len(x)
        lm_x = self.landmark_model.get_landmarks_batch(x).reshape((bs, -1, 2))
        lm_y = self.landmark_model.get_landmarks_batch(y).reshape((bs, -1, 2))
                
        res = torch.tensor(np.linalg.norm(lm_x - lm_y, axis=-1).mean()).float()
        return res
    
    def calc_dist_uniformity(self, x, func=None, is_path=True):
        if is_path:
            # x: path to samples in .txt format
            samples = np.loadtxt(x)
        else:
            smaples = x
        if func is not None:
            samples = func(samples)
        return stats.kstest(samples, stats.uniform.cdf)
    
    def calc_dist_similarity(self, x, y, is_path=True):
        pass
    
    def AED(self, x, y, is_path=True):
        if is_path:
            x = self.get_frames(x)[[0]]
            y = self.get_frames(y)
            
        bs = len(y)
        chunk_length = min(32, bs)
        
        feat_x = self.inception(x.to('cuda:0')).detach().cpu()
        # print(f'feat_x shape: {feat_x.shape}')
        feat_y = []
        for i in range(0, bs, chunk_length):
            y_mini = y[i : i + chunk_length]
            # print(f'y_mini shape: {y_mini.shape}')
            feat_y_mini = self.inception(y_mini.to('cuda:0')).detach().cpu()
            feat_y.append(feat_y_mini)
            
        feat_y = torch.cat(feat_y, dim=0)

        feat_x = feat_x.repeat(len(feat_y), 1)
        res = torch.norm(feat_x - feat_y, dim=-1).mean().detach().cpu()
        
        return res

    def f1(self, x, y):
        ## x, y are torch boolean tensor
        precision = (x * (x == y)).sum() / x.sum()
        recall = (y * (x == y)).sum() / y.sum()
        TP = (x * (x == y)).sum()
        FP = (x * (x != y)).sum()
        FN = ((x == False) * (x != y)).sum()
        

        precision = TP / (TP + FP)
        recal = TP / (TP + FN)
        f1 = 2 * TP / (2 * TP + FP + FN)

        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def AUCON(self, x, y, is_path=True):
        # x, y are paths to session/frames 
        ## run open face evaluation
        openface_dir_x = os.path.join(x, '..', 'openface')
        openface_dir_y = os.path.join(y, '..', 'openface')
        
        file_x = self.openface_extract_feature_from_dir(x, openface_dir_x)
        file_y = self.openface_extract_feature_from_dir(y, openface_dir_y)
        
        feat_x = self.load_au_c_from_csv(file_x)
        feat_y = self.load_au_c_from_csv(file_y)
        
        f1_scores = {}
        for k in list(feat_x.keys()):
            au_c_x = torch.tensor(feat_x[k]).bool()
            au_c_y = torch.tensor(feat_y[k]).bool()
            f1 = self.f1(au_c_x, au_c_y)
            f1_scores[k] = f1['f1']
        
        vs = []
        for v in list(f1_scores.values()):
            if not torch.isnan(v):
                vs.append(v)
        mean = torch.stack(vs).mean()

        return mean
        
    def CSIM(self, x, y, is_path):
        # x, y: B x C x H x W
        if is_path:
            x = self.get_frames(x)
            y = self.get_frame(y)
        
        # extract feature using Inception / VGG
        # x_feat = 
        # y_feat = 
        
        res = torch.einsum('bd,cd->bc', F.normalize(x_feat), F.normalize(y_feat))
            
        return res
        
    def run(self, metrics=[]):
        pass

