import os
import shutil
import sys
import imageio
from skimage import io, img_as_float32, img_as_ubyte
import torch

module_dir = '/home/server19/minyeong_workspace/MDTH/models/fv2v2'
sys.path.append(module_dir)

import utils
from utils.util import extract_mesh_normalize, mesh_tensor_to_landmarkdict

path = '/mnt/hdd/minyeong_workspace/Experiment/data/celeb/E_00002_cropped.png' ### suhyeon kim's frontalized face

img = img_as_ubyte(imageio.imread(path))

# mesh_tensor = extract_mesh(img)['raw_value']

# mesh_dict = mesh_tensor_to_landmarkdict(mesh_tensor)

ref_dict = '/home/server19/minyeong_workspace/MDTH/models/fv2v2/reference_mesh.pt'
ref = torch.load(ref_dict)

mesh_dict  = extract_mesh_normalize(img, ref)

save_path = '/home/server19/minyeong_workspace/MDTH/models/fv2v2/rhs_mesh.pt'

torch.save(mesh_dict, save_path)



