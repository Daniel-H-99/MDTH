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
input_file = 'exp_check_expression/inputs.txt'
result_dir = 'exp_check_expression/'
ref_path = 'reference_mesh.pt'
ref = torch.load(ref_path)
config = '/home/server19/minyeong_workspace/MDTH/models/fv2v2/config/mesh_mesh_fc_stage2_v2.yaml'
config = 


shape = (256, 256)

with open(input_file) as f:
    inputs = f.readlines()

inputs = [v.strip() for v in inputs]

### load models
self.exp_transformer = export.load_exp_transformer(self.config.config.common.checkpoints.exp_transformer.config, self.config.config.common.checkpoints.exp_transformer.model, self.gpus)
self.generator = export.load_generator(self.config.config.common.checkpoints.generator.config, self.config.config.common.checkpoints.generator.model, self.gpus)

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

        
