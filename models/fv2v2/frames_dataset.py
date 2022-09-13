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
                        print(f'loading to cache: {cache_path}')
                        t = time.time()
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        shutil.copytree(path, cache_path)
                        t = time.time() - t
                        print(f'loaded to cache ({t}): {cache_path}')
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
