import os
from skimage import io, img_as_float32, img_as_ubyte
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from datetime import datetime
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob
import pickle as pkl
import cv2 

import os
from skimage import io, img_as_float32, img_as_ubyte
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
import imageio
import time 
from utils.util import extract_mesh, get_mesh_image, draw_section, draw_mouth_mask, OPENFACE_EYE_IDX, OPENFACE_LIP_IDX, LEFT_EYE_IDX, LEFT_EYEBROW_IDX,LEFT_IRIS_IDX, RIGHT_EYE_IDX, RIGHT_EYEBROW_IDX, RIGHT_IRIS_IDX, IN_LIP_IDX, OUT_LIP_IDX
import torch
from modules.landmark_model import LandmarkModel

def extend_bbox(bbox, frame):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    X = (x1 + x2) // 2
    Y = (y1 + y2) // 2
    L = max(x2 - x1, y2 - y1)
    L = int(L * 1.4)
    # random_move = (2 * random.random() - 1) * L * 0.1
    random_move = 0
    topleft = (int(max(min(Y - L // 2 + random_move, H - L), 0)), int(max(min(X - L // 2 + random_move, W - L), 0)))
    y1_ext, x1_ext = topleft
    y2_ext, x2_ext = y1_ext + L, x1_ext + L
    bbox_ext = (x1_ext, y1_ext, x2_ext, y2_ext)
    return bbox_ext


def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset3(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, cache=None, z_bias=0, landmarkmodel_path=None):
        # self.sections = sections
        self.root_dir = root_dir
        self._root_dir = root_dir
        self.cache = cache
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = True
        self.z_bias = z_bias
        self.landmark_model = LandmarkModel(landmarkmodel_path)
        # self.reference_dict = torch.load('mesh_dict_reference.pt')
        if os.path.exists(os.path.join(root_dir, 'train')):
            # assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            tag = 'train' if is_train else 'test'
            if id_sampling:
                train_videos = os.listdir(os.path.join(self.root_dir, tag))
                train_videos = list(train_videos)
                ids = {}
                for vid in train_videos:
                    id = vid.split('#')[0]
                    ids[id] = ids.get(id, []) + [vid]
                train_videos = ids
            else:
                train_videos = os.listdir(os.path.join(root_dir, tag))
            # test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        # if is_train:
        self.videos = train_videos
        # else:
        #     self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None
            
    def __len__(self):
        return len(self.videos)

    def split_section(self, X):
        res = []
        for i, sec in enumerate(self.sections):
            res.append(X[sec[0]])
        return res

    def extract_openface_mesh(self, image, noise=None):
        mesh = {}
        H, W = image.shape[:2]
        bb, lm = self.landmark_model.get_landmarks_fa(image)
        # lm[:, 1] = H - lm[:, 1]
        lm_normed_3d, U, Ind = self.landmark_model.normalize_mesh(lm, H, W, z_mean=0)
        # U = torch.from_numpy(U['U'])
        normalizer = U['normalizer']
        U = U['U']
        # print(f'normed: {lm_normed_3d}')
        if noise is not None:
            # if np.r
            noise = np.random.randn(3, 3) * noise
            noise = noise.clip(-0.1, 0.1) * 100
            lm_normed_3d[[3] + list(range(17, 22)) + list(range(36, 42))] += noise[[0]]
            lm_normed_3d[[4] + list(range(22, 27)) + list(range(42, 48))] += noise[[1]]
            lm_normed_3d[48:] += noise[[2]]
            # lm_normed_3d = np.concatenate([lm_normed_3d, noise], axis=0)

        lm_3d = np.concatenate([lm_normed_3d, np.ones((len(lm_normed_3d), 1))], axis=1) @ U
        if noise is not None:
            noise_real = noise @ U[:3, :3]
        else:
            noise_real = None

        lm_3d = lm_3d[:, :3]
        # lm_3d[:, 1] = H - lm_3d[:, 1]
        scale = H // 2
        lm_scaled_3d = lm_normed_3d / scale
        mesh["raw_value"] = lm_3d
        mesh["value"] = lm_scaled_3d.astype(np.float32)
        mesh["U"] = U.astype(np.float32)
        mesh["scale"] = scale
        # print(f'landmark: {lm}')
        # print(f'mesh: {mesh}')

        return mesh, noise_real, normalizer
        
    def concat_section(self, sections):
        # sections[]: (num_sections) x -1 x 3
        return np.concatenate(sections, axis=0)

    def get_mouth_image(self, mesh):
        mouth = draw_mouth_mask(mesh[:, :2].astype(np.int32), self.frame_shape)
        mouth = mouth[:, :, :1].astype(np.float32).transpose((2, 0, 1))
        
        return mouth
    
    def get_mesh_image_section(self, mesh, section_config=None):
        # mesh: N0 x 3
        # print(f'mesh type: {mesh.type()}')
        # mouth_mask = (255 * draw_mouth_mask(mesh[:, :2].numpy().astype(np.int32), self.frame_shape)).astype(np.int32)
        # print(f'mouth mask shape {mouth_mask.type()}')
        if section_config is None:
            sections = self.concat_section(self.split_section(mesh))
            secs = draw_section(sections[:, :2].astype(np.int32), self.frame_shape, split=False) # (num_sections) x H x W x 3
        else:
            section_ids = []
            for sec in section_config:
                section_ids.extend(sec)
            sections = mesh[section_ids]
            secs = draw_section(sections[:, :2].astype(np.int32), self.frame_shape, section_config=section_config, split=False) # (num_sections) x H x W x 3
        # print(f'sections shape: {sections.shape}')
        
        # print(f'draw section done')
        secs = secs[:, :, :1].astype(np.float32).transpose((2, 0, 1)) / 255.0
        # secs = [sec[:, :, :1].astype(np.float32).transpose((2, 0, 1)) / 255.0 for sec in secs]
        # print('got mesh image sections')
        return secs
    
    def __getitem__(self, idx):
        while True:
            try:
                idx = (idx + int(datetime.now().timestamp())) % (len(self.videos))
                if self.is_train and self.id_sampling:
                    id = list(self.videos.keys())[idx]
                    name = str(np.random.choice(self.videos[id]))
                    path = str(np.random.choice(glob.glob(os.path.join(self.root_dir, name))))
                else:
                    name = self.videos[idx]
                    path = os.path.join(self.root_dir, name)

                # if self.is_train and os.path.isdir(path):
                frames = os.listdir(path)
                num_frames = len(frames)
                frame_idx = np.sort(np.random.choice(num_frames, replace=False, size=2))
                frame_idx[1] = (frame_idx[1] + 100) % num_frames

                raw_video_array = [io.imread(os.path.join(path, frames[(idx + int(datetime.now().timestamp())) % num_frames])) for idx in frame_idx]
                video_array = np.stack([cv2.resize(img_as_float32(frame), self.frame_shape[:2]) for frame in raw_video_array], axis=0)

                meshes = []

                for i, frame in enumerate(video_array):
                    L = self.frame_shape[0]
                    mesh, noise, normalizer = self.extract_openface_mesh(img_as_ubyte(frame)) # {value (N x 3), R (3 x 3), t(3 x 1), c1}
                    A = np.array([[-1, -1, 0]], dtype='float32') # 3 x 1
                    # mesh = {}

                    mesh_mp = extract_mesh(img_as_ubyte(frame))
                    right_iris = mesh_mp['raw_value'][RIGHT_IRIS_IDX].mean(dim=0) # 3
                    left_iris = mesh_mp['raw_value'][LEFT_IRIS_IDX].mean(dim=0) # 3
                    # print(f'right_iris shape: {right_iris.shape}')
                    # mesh['value'][3] = (normalizer(right_iris[None].numpy().astype(np.float32)) / (L // 2))
                    # mesh['value'][4] = (normalizer(left_iris[None].numpy().astype(np.float32)) / (L // 2))
                    # print(f'right_iris: {mesh["value"][3]}')
                    # print(f'right_iris: {mesh["value"][3]}')
                    # print(f'right_eye: {mesh["value"][36:42].mean(axis=0)}')
                    # mesh_mp['raw_value'][:, 2] = mesh_mp['raw_value'][:, 2]
                    mesh_mp['_raw_value'] = mesh_mp['raw_value'].clone().detach()


                    # if noise is not None:
                    #     mesh_mp['raw_value'][RIGHT_EYEBROW_IDX + RIGHT_EYE_IDX + RIGHT_IRIS_IDX] += torch.tensor(noise[[0]])
                    #     # print(f"right: {mesh_mp['raw_value'][RIGHT_EYE_IDX+RIGHT_EYEBROW_IDX]}")
                    #     # print(f'onise: {noise[0]}')
                    #     # print(f"left: {mesh_mp['raw_value'][LEFT_EYE_IDX+LEFT_EYEBROW_IDX]}")
                    #     # print(f'mesh open right: {mesh["raw_value"][36:42]}')
                    #     mesh_mp['raw_value'][LEFT_EYEBROW_IDX + LEFT_EYE_IDX + LEFT_IRIS_IDX] += torch.tensor(noise[[1]])
                    #     mesh_mp['raw_value'][OUT_LIP_IDX+IN_LIP_IDX] += torch.tensor(noise[[2]])

                    # print(f'value: {mesh["raw_value"][36:42]} ')
                    # print(f'mp value: {mesh_mp["raw_value"][RIGHT_EYE_IDX]}')
                    # mesh['value'] = np.array(mesh['value'], dtype='float32') * 2 / L  + np.squeeze(A, axis=-1)[None]
                    # mesh['R'] = np.array(mesh['R'], dtype='float32')
                    # mesh['c'] = np.array(mesh['c'], dtype='float32')
                    # t = np.array(mesh['t'], dtype='float32')
                    # mesh['t'] = (np.eye(3).astype(np.float32) - mesh['c'] * mesh['R']) @ A + t * 2 / L
                    # # print('checkpoint 1')

                    # mesh['mesh_img'] = (get_mesh_image(mesh['raw_value'], self.frame_shape)[:, :, [0]] / 255.0).transpose((2, 0, 1))
                    MP_SECTIONS_CONFIG = [LEFT_EYEBROW_IDX, LEFT_EYE_IDX, LEFT_IRIS_IDX, RIGHT_EYEBROW_IDX, RIGHT_EYE_IDX, RIGHT_IRIS_IDX, OUT_LIP_IDX, IN_LIP_IDX]
                    MP_SECTIONS =  LEFT_EYEBROW_IDX + LEFT_EYE_IDX + LEFT_IRIS_IDX + RIGHT_EYEBROW_IDX + RIGHT_EYE_IDX + RIGHT_IRIS_IDX + OUT_LIP_IDX + IN_LIP_IDX

                    mesh['MP_SECTIONS'] = MP_SECTIONS
                    mesh['MP_EYE_SECTIONS'] = LEFT_EYEBROW_IDX + LEFT_EYE_IDX + LEFT_IRIS_IDX + RIGHT_EYEBROW_IDX + RIGHT_EYE_IDX + RIGHT_IRIS_IDX
                    mesh['MP_MOUTH_SECTIONS'] = OUT_LIP_IDX + IN_LIP_IDX
                    
                    mesh['mesh_img_sec'] =  np.zeros_like(self.get_mesh_image_section(mesh_mp['raw_value'].numpy(), section_config=MP_SECTIONS_CONFIG))
                    mesh['_mesh_img_sec'] =  np.zeros_like(self.get_mesh_image_section(mesh_mp['_raw_value'].numpy(), section_config=MP_SECTIONS_CONFIG))
                    # mesh['raw_value'] = mesh_mp['raw_value'] * 2 / L + A
                    # mesh['_raw_value'] = mesh_mp['_raw_value'] * 2 / L + A

                    # print('msh img sec got')
                    # print(f'mp sections: {MP_SECTIONS}')
                    # print(f'mesh mp shape: {mesh_mp["raw_value"].shape}')
                    mesh['section_landmarks'] = mesh_mp["raw_value"][MP_SECTIONS] * 2 / L + A
                    # mesh['mouth_img'] = self.get_mouth_image(mesh['raw_value'].numpy())
                    # print(f'mouth image shape: {mesh["mouth_img"].shape}')
                    # mouth_center = mesh['raw_value'][-20:, :2].mean(dim=0) # 2
                    # mouth_corner = (mouth_center - np.array([[L // 4, L // 4]])).clip(min=0)
                    # print(f"mp_raw_value: {mesh_mp['raw_value'][OUT_LIP_IDX] * 2 / L + A}")
                    # print(f"raw_value: {mesh['raw_value'][48:] * 2 / L + A}")
                    # print(f"value: {mesh['value'][48:]}")

                    # mesh['raw_value'] = np.array(mesh_mp['raw_value'], dtype='float32') * 2 / L + A
                    
                    ### use openface raw value
                    mesh['raw_value'] = np.array(mesh['raw_value'], dtype='float32') * 2 / L + A
                    mesh['OPENFACE_EYE_IDX'] = OPENFACE_EYE_IDX
                    mesh['OPENFACE_LIP_IDX'] = OPENFACE_LIP_IDX
                    
                    # print('checkpoint 2')
                    # print(f'data type: {mesh["value"].dtype}')
                    meshes.append(mesh)
                    
                # ### Make intermediate target mesh ###
                # src_mesh = meshes[0]
                # drv_mesh = meshes[1]
                # target_mesh = (1 / src_mesh['c'][np.newaxis, np.newaxis]) * np.einsum('ij,nj->ni', np.linalg.inv(src_mesh['R']), drv_mesh['value'] - src_mesh['t'][np.newaxis, :, 0])
                # drv_mesh['intermediate_value'] = target_mesh
                # target_mesh = L * (target_mesh - np.squeeze(A, axis=-1)[None]) // 2
                # drv_mesh['intermediate_mesh_img_sec'] = self.get_mesh_image_section(target_mesh)
                
                break
            
            except Exception as e:
                print(f'error: {e}')
                continue
            
            
        out = {}
        # if self.is_train:
        source = np.array(video_array[0], dtype='float32')
        driving = np.array(video_array[1], dtype='float32')
        out['driving'] = driving.transpose((2, 0, 1))
        out['source'] = source.transpose((2, 0, 1))
        out['driving_mesh'] = meshes[1]
        out['source_mesh'] = meshes[0]
        out['hit'] = 0
        out['name'] = path
        # out['hopenet_source'] = hopenet_video_array[0]
        # out['hopenet_driving'] = hopenet_video_array[1]
        
        # else:
        #     video = np.array(video_array, dtype='float32')
        #     out['video'] = video.transpose((3, 0, 1, 2))
        #     out['mesh'] = meshes
            
        # out['name'] = video_name

        return out

class FramesDataset4(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, cache=None):
        self.root_dir = root_dir
        self._root_dir = root_dir
        self.cache = cache
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = True
        # self.reference_dict = torch.load('mesh_dict_reference.pt')
        if os.path.exists(os.path.join(root_dir, 'train')):
            # assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            tag = 'train' if is_train else 'test'
            if id_sampling:
                train_videos = os.listdir(os.path.join(self.root_dir, tag))
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, tag))
            # test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        # if is_train:
        self.videos = train_videos
        # else:
        #     self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None
            
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        while True:
            try:
                magic_num = int(datetime.now().timestamp())
                idx = (idx + magic_num) % (len(self.videos))
                id = self.videos[idx]
                chunk = np.random.choice(list(filter(lambda x: '.mp4' not in x and '.wav' not in x, os.listdir((os.path.join(self.root_dir, id))))))
                path = os.path.join(self.root_dir, id, chunk)
                if self.cache is not None:
                    rel_path = os.path.relpath(path, self._root_dir)
                    cache_path = os.path.join(self.cache, rel_path)
                    if not os.path.exists(cache_path):
                        hit = False
                        # print(f'loading to cache: {cache_path}')
                        t = time.time()
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        shutil.copytree(path, cache_path)
                        t = time.time() - t
                        # print(f'loaded to cache ({t}): {cache_path}')
                    else:
                        hit = True
                        # print(f'cache hit: {cache_path}')
                    path = cache_path

                frames_dir = os.path.join(path, 'frames')
                faces_path = os.path.join(path, 'faces.pckl')

                with open(faces_path, 'rb') as f:
                    faces = pkl.load(f)

                num_frames = len(faces)
                
                item_indice = np.random.choice(num_frames, replace=False, size=2)
                item_indice[1] = (item_indice[1] + magic_num) % num_frames

                video_array = []

                bboxes = []
                for item_idx in item_indice:
                    item = faces[item_idx][0]  # frame, bbox, conf
                    fid = item['frame']
                    bbox = item['bbox']
                    frame_path = os.path.join(frames_dir, "{:05d}.jpg".format(fid + 1))
                    frame = io.imread(frame_path)
                    x1, y1, x2, y2 = extend_bbox(bbox, frame)
                    bboxes.append([x1, y1, x2, y2])
                bboxes = np.array(bboxes)
                bbox_united = bboxes.mean(axis=0)
                x1, y1, x2, y2 = bbox_united.astype(int)

                for item_idx in item_indice:
                    item = faces[item_idx][0]  # frame, bbox, conf
                    fid = item['frame']
                    frame_path = os.path.join(frames_dir, "{:05d}.jpg".format(fid + 1))
                    frame = io.imread(frame_path)
                    cropped = frame[y1:y2, x1:x2]
                    cropped_scaled = cv2.resize(img_as_float32(cropped), self.frame_shape[:2])
                    video_array.append(cropped_scaled)

                video_array = np.stack(video_array, axis=0)

                if self.transform is not None:
                    video_array = self.transform(video_array)

                break
            
            except Exception as e:
                print(f'error: {e}')
                continue
            
            
        out = {}
        # if self.is_train:
        source = np.array(video_array[0], dtype='float32')
        driving = np.array(video_array[1], dtype='float32')
        out['driving'] = driving.transpose((2, 0, 1))
        out['source'] = source.transpose((2, 0, 1))
        out['hit'] = hit
        out['name'] = path

        # out['hopenet_source'] = hopenet_video_array[0]
        # out['hopenet_driving'] = hopenet_video_array[1]
        
        
        # else:
        #     video = np.array(video_array, dtype='float32')
        #     out['video'] = video.transpose((3, 0, 1, 2))
        #     out['mesh'] = meshes
            
        # out['name'] = video_name

        return out
    

class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
