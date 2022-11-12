import os
import torch
import shutil
import sys
import imageio
from skimage import io, img_as_float32, img_as_ubyte
import numpy as np

module_dir = '/home/server19/minyeong_workspace/MDTH/models/fv2v2'
sys.path.append(module_dir)

import utils
from utils.util import extract_mesh_normalize, mesh_tensor_to_landmarkdict, get_mesh_image

data_dir = '/mnt/hdd/minyeong_workspace/Experiment/proc'
input_file = 'exp_check_mp_stability/inputs.txt'
result_dir = 'exp_check_mp_stability'
ref_path = 'exp_check_mp_stability/reference_mesh.pt'
ref = torch.load(ref_path)
shape = (256, 256)

with open(input_file) as f:
    inputs = f.readlines()

inputs = [v.strip() for v in inputs]

for vid in inputs:
    print(f'working video: {vid}')
    source_dir = os.path.join(data_dir, vid)
    vid_name = os.path.basename(source_dir)
    source_dir = os.path.join(source_dir, 'frames')
    save_dir = os.path.join(result_dir, f'{vid_name}')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    predictions = []
    frames = sorted(os.listdir(source_dir))
    
    for frame in frames:
        frame_path = os.path.join(source_dir, frame)
        img = img_as_ubyte(imageio.imread(frame_path))
        mesh_tensor = extract_mesh_normalize(img, ref)['value']
        mesh_pred = get_mesh_image(mesh_tensor, shape).astype(np.uint8)
        pred = np.concatenate([img, mesh_pred], axis=1)
        predictions.append(pred)

    imageio.mimsave(os.path.join(save_dir, 'video.mp4'), predictions, fps=25)

        
