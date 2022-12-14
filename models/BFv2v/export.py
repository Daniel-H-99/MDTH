import numpy as np
import cv2
from skimage import io
from matplotlib import pyplot as plt
import face_alignment
import argparse
import os
import torch
import yaml 
import imageio
import numpy as np
import face_alignment
import argparse
import yaml
import time
import cv2
import math
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from skimage.transform import resize
from skimage import img_as_ubyte, img_as_float32
import torch.multiprocessing as mp
import math
###############################################################
from BFv2v.modules.keypoint_detector import HEEstimator
from BFv2v.modules.landmark_model import LandmarkModel
from BFv2v.modules.generator import OcclusionAwareSPADEGenerator
from BFv2v.sync_batchnorm import DataParallelWithCallback
from BFv2v.utils.util import extract_mesh, draw_section, draw_mouth_mask, get_mesh_image, matrix2euler, euler2matrix, RIGHT_IRIS_IDX, LEFT_IRIS_IDX, extract_mesh
from BFv2v.utils.one_euro_filter import OneEuroFilter
from BFv2v.ffhq_align import image_align
###############################################################


def load_landmark_model(config_dir, gpu=[0]):
    landmark_model = LandmarkModel(config_dir, gpu)
    return landmark_model

def load_he_estimator(config, he_estimator_path, gpu=[0]):
    assert config is not None
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                            **config['model_params']['common_params'])

    if torch.cuda.is_available():
        he_estimator.to(gpu[0])

    ckpt = torch.load(he_estimator_path, map_location=f'cuda:{gpu[0]}')
    he_estimator.load_state_dict(ckpt['he_estimator'])
    he_estimator.eval()
    return he_estimator

def load_checkpoints(config, checkpoint_path, gpu=[0]):

    generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params'])
    generator = generator.to(gpu[0])
    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{gpu[0]}')
    generator_dict= generator.state_dict()
    for k,v in checkpoint['generator'].items():
        if k in generator_dict:
            generator_dict[k] = v
    generator.load_state_dict(generator_dict)
    generator.eval()
    generator = DataParallelWithCallback(generator, device_ids=gpu)

    return generator

def load_bfv2v(config, checkpoint_path, gpu=[0]):
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['train_params']['num_kp'] = config['model_params']['common_params']['num_kp']
    config['train_params']['sections'] = config['model_params']['common_params']['sections']

    sections = config['train_params']['sections']
    
    generator = load_checkpoints(config=config, checkpoint_path=checkpoint_path, gpu=gpu)
    generator.ignore_emotion = True
    return generator


def extract_landmark_from_img(path, out_dir, he_estimator, landmark_model):
    output_name = path.split('/')[-1].split('.jpg')[0].split('.png')[0]
    
    im = imageio.imread(path)
    im = im[:, :, :3]
    H, W = im.shape[:2]
    (bb, coords) = landmark_model.get_landmarks_fa(im)
    vertices, a, Ind = landmark_model.normalize_mesh(coords, H, W)

    coords_3d = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1) @ a['U']

    normalizer = a['normalizer']
    mesh_mp = extract_mesh(img_as_ubyte(im))
    right_iris = mesh_mp['raw_value'][RIGHT_IRIS_IDX].mean(dim=0) # 3
    left_iris = mesh_mp['raw_value'][LEFT_IRIS_IDX].mean(dim=0) # 3
    normed_right_iris = normalizer(right_iris[None].numpy().astype(np.float32))
    normed_left_iris = normalizer(left_iris[None].numpy().astype(np.float32))

    del a['normalizer']
    landmark_item = {'2d_landmarks':torch.from_numpy(coords), '3d_landmarks_pose': torch.from_numpy(coords_3d), '3d_landmarks': torch.from_numpy(vertices), 'p': a, 'normed_right_iris': torch.from_numpy(normed_right_iris), 'normed_left_iris': torch.from_numpy(normed_left_iris)}

    with torch.no_grad():
        he = he_estimator(torch.tensor(im).to(landmark_model.gpu[0]).permute(2, 0, 1)[None, [2, 1, 0]] / 255) # {R, t, ...}
    R, t = he['R'][0], he['t'][0]

    landmark_item['he_p'] = {'R': he['R'][0].detach().cpu().numpy(), 't': he['t'][0].detach().cpu().numpy()}

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    torch.save(landmark_item, os.path.join(out_dir, '3d_landmarks.pt'))
    return 

def extract_landmark_from_video(resource, target, he_estimator, landmark_model):

    if not os.path.exists(target):
        os.mkdir(target)

    start = time.time()

    # svVideo = os.path.join(target, 'output.mp4') # create output video file
    sv2DLdMarks = os.path.join(target, '2d_landmarks') # create 2D landmarks file
    sv3DLdMarks = os.path.join(target, '3d_landmarks') # create 3D frontalised landmarks file
    sv3DLdMarks_Pose = os.path.join(target, '3d_landmarks_pose') # create 3D landmarks coupled with pose file
    svNonDetect = os.path.join(target, 'NonDetected') # create non-detected frames file

    landmarks  = []
    nonDetectFr = []

    cap = cv2.VideoCapture(resource)  # load video
    # video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    totalFrame = np.int32(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("Total frames: ", totalFrame)
    print("Frame size: ", size)
    print("fps: ", fps)
    vis = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # create VideoWriter object
    # width, height = 256, 256
    totalIndx = 0

    d = 0
    while(cap.isOpened()):
        frameIndex = np.int32(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print("Processing frame ", frameIndex, "...")
        # capture frame-by-frame
        ret, frame = cap.read()
        if ret==True:
            # operations on the frame
            try:
                # generate face bounding box and track 2D landmarks for current frame
                frame = frame[:, :, :3]
                height, width = frame.shape[:2]
                (bb, frame_landmarks) = landmark_model.get_landmarks_fa(frame)
            except:
                print("Landmarks in frame ", frameIndex, " (", frameIndex/fps, " s) could not be detected.")
                nonDetectFr.append(frameIndex/fps)
                continue

            if he_estimator is not None:
                with torch.no_grad():
                    he = he_estimator(torch.tensor(frame).to(landmark_model.gpu[0]).permute(2, 0, 1)[None, [2, 1, 0]] / 255) # {R, t, ...}
                R, t = he['R'][0], he['t'][0]

            vertices, a, Ind = landmark_model.normalize_mesh(frame_landmarks, height, width)

            coords_3d = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1) @ a['U']

            normalizer = a['normalizer']
            mesh_mp = extract_mesh(img_as_ubyte(frame))
            right_iris = mesh_mp['raw_value'][RIGHT_IRIS_IDX].mean(dim=0) # 3
            left_iris = mesh_mp['raw_value'][LEFT_IRIS_IDX].mean(dim=0) # 3
            # print(f'right_iris shape: {right_iris.shape}')
            normed_right_iris = normalizer(right_iris[None].numpy().astype(np.float32))
            normed_left_iris = normalizer(left_iris[None].numpy().astype(np.float32))

            del a['normalizer']
            
            landmark_item = {
                '2d_landmarks': torch.from_numpy(frame_landmarks),
                '3d_landmarks_pose': torch.from_numpy(coords_3d),
                '3d_landmarks': torch.from_numpy(vertices),
                'p': a,
                'normed_right_iris': torch.from_numpy(normed_right_iris),
                'normed_left_iris': torch.from_numpy(normed_left_iris)
            }

            if he_estimator is not None:
                landmark_item['he_p'] = {'R': he['R'][0].detach().cpu().numpy(), 't': he['t'][0].detach().cpu().numpy()}

            landmarks.append(landmark_item)

            totalIndx = totalIndx + 1

        else:
            break

    torch.save(landmarks, os.path.join(target, '3d_landmarks.pt'))

def adapt_values(origin, values, minimum=None, maximum=None, rel_minimum=None, rel_maximum=None, scale=None, center_align=False, center=None):
    # origin: float
    # values: tensor of size L
    if scale is None:
        scale = 1
    sample_min, sample_max, sample_mean = values.min(), values.max(), values.mean()
    
    if not center_align:
        origin = sample_mean

    if center is not None:
        origin = center

    if rel_maximum is not None:
        if maximum is not None:
            maximum = min(maximum, origin + rel_maximum)
        else:
            maximum = origin + rel_maximum
        
    if rel_minimum is not None:
        if minimum is not None:
            minimum = max(minimum, origin + rel_minimum)
        else:
            minimum = origin + rel_minimum

    if (minimum is not None) and (maximum is not None):
        scale = min(scale, (maximum - minimum) / (sample_max - sample_min).clamp(min=1e-6))

    inter_values = origin + scale * (values - sample_mean)
    inter_min, inter_max = inter_values.min(), inter_values.max()
    adapted_values = inter_values
    
    if minimum is not None:
        print(f'minimum: {minimum}')
        print(f'inter min: {inter_min}')
        clip = max(minimum, inter_min)
        delta = clip - inter_min
        adapted_values = adapted_values + delta
       
    if maximum is not None: 
        clip = min(maximum, inter_min)
        delta = clip - inter_min
        adapted_values = adapted_values + delta
    
    return adapted_values

def filter_values(values):
    MIN_CUTOFF = 1.0
    BETA = 1.0
    num_frames = len(values)
    fps = 30
    times = np.linspace(0, num_frames / fps, num_frames)
    
    filtered_values= []
    
    values = values
    
    for i, x in enumerate(values):
        if i == 0:
            filter_value = OneEuroFilter(times[0], x, min_cutoff=MIN_CUTOFF, beta=BETA)
        else:
            x = filter_value(times[i], x)
        
        filtered_values.append(x)
        
    res = np.array(filtered_values)
    res = res
    return res


def filter_mesh(meshes, source_mesh, SCALE):
    # meshes: list of dict of mesh({R, t, c})
    R_xs = []
    R_ys = []
    R_zs = []
    t_xs = []
    t_ys = []
    t_zs = []
    
    ts = []
    for i, mesh in enumerate(meshes):
        he_R, t = mesh['R'], mesh['t']
        ts.append(t)
        R_x, R_y, R_z = matrix2euler(he_R)
        t_x, t_y, t_z = t
        R_xs.append(R_x)
        R_ys.append(R_y)
        R_zs.append(R_z)
        t_xs.append(t_x)
        t_ys.append(t_y)
        t_zs.append(t_z)
    
    R_xs = torch.tensor(R_xs).float()
    R_ys = torch.tensor(R_ys).float()
    R_zs = torch.tensor(R_zs).float()

    R_x_source, R_y_source, R_z_source = matrix2euler(source_mesh['R'])
    
    R_xs_adapted = adapt_values(R_x_source, R_xs, minimum=(-math.pi / 6), maximum=(math.pi / 6), center_align=True)
    R_ys_adapted = adapt_values(R_y_source, R_ys, rel_minimum=(-math.pi / 6), rel_maximum=(math.pi / 6), center_align=True)
    R_zs_adapted = adapt_values(R_z_source, R_zs, rel_minimum=(-math.pi / 6), rel_maximum=(math.pi / 6), center_align=True)
    
    R_xs_filtered = torch.tensor(filter_values(R_xs_adapted.numpy())).float()
    R_ys_filtered = torch.tensor(filter_values(R_ys_adapted.numpy())).float()
    R_zs_filtered = torch.tensor(filter_values(R_zs_adapted.numpy())).float()
    
    new_Rs = []

    for R_x, R_y, R_z in zip(R_xs_filtered, R_ys_filtered, R_zs_filtered):
        new_R = torch.tensor(euler2matrix([R_x, R_y, R_z])).float()
        new_Rs.append(new_R)

    new_Rs = torch.stack(new_Rs, dim=0).numpy()
    ts = np.stack(ts, axis=0)
    final_Us = torch.tensor(np.concatenate([source_mesh['s'] * SCALE * new_Rs, SCALE * (-source_mesh['s'] * new_Rs @ source_mesh['b'] + ts[:, :, np.newaxis] + 1)], axis=2).transpose(0, 2, 1)).float()
    t_xs = final_Us[:, 3, 0]
    t_ys = final_Us[:, 3, 1]
    t_zs = final_Us[:, 3, 2]

    t_x_source, t_y_source, t_z_source = source_mesh['t']

    t_xs_adapted = adapt_values(t_x_source, t_xs, minimum=32, maximum=224, center_align=True)
    t_ys_adapted = adapt_values(t_y_source, t_ys, minimum=32, maximum=224, center_align=True)
    t_zs_adapted = adapt_values(t_z_source, t_zs, center_align=True)
    
    
    t_xs_filtered = torch.tensor(filter_values(t_xs_adapted.numpy())).float()
    t_ys_filtered = torch.tensor(filter_values(t_ys_adapted.numpy())).float()
    t_zs_filtered = torch.tensor(filter_values(t_zs_adapted.numpy())).float()
    
    for R_x, R_y, R_z, t_x, t_y, t_z, mesh in zip(R_xs_filtered, R_ys_filtered, R_zs_filtered, t_xs_filtered, t_ys_filtered, t_zs_filtered, meshes):
        new_R = torch.tensor(euler2matrix([R_x, R_y, R_z])).float()
        new_t = torch.stack([t_x, t_y, t_z], dim=0).float()
        mesh['R'] = new_R 
        mesh['t'] = new_t

        rot_src = source_mesh['view'].copy()
        trans_src = source_mesh['viewport'].copy()

        r = R.from_matrix(rot_src[:3, :3])

        rot_src[:3, :3] = new_R.numpy().astype(np.float32)
        
        trans_src[:2, 3] = new_t[:2].numpy().astype(np.float32)
        
        final_U = rot_src.T @ source_mesh['proj'].T @ trans_src.T
        mesh['U'] = torch.tensor(final_U).float()

def get_mesh_image_section(mesh, frame_shape, section_indices, sections_indices_splitted):
    # mesh: N0 x 3
    secs = draw_section(mesh[section_indices, :2].numpy().astype(np.int32), frame_shape, section_config=sections_indices_splitted[:6], groups=[0] * 6, split=False) # (num_sections) x H x W x 3
    secs = torch.from_numpy(secs[:, :, :1].astype(np.float32).transpose((2, 0, 1)) / 255.0)

    return secs

def preprocess_dict(d_list, device='cuda:0'):
    res = {}
    d_ref = d_list[0]

    for k, v in d_ref.items():
        if type(v) == torch.Tensor:
            res[k] = torch.stack([d[k] for d in d_list], dim=0).to(device)
        elif type(v) == list:
            tmp = []
            for i in range(len(v)):
                tmp.append(torch.stack([d[k][i]for d in d_list], dim=0).to(device))
            res[k] = tmp
        elif type(v) == dict:
            res[k] = preprocess_dict([d[k] for d in d_list])
        else:
            res[k] = torch.cat([torch.Tensor([d[k]]) for d in d_list], dim=0).to(device)
        
    return res

def make_animation(rank, gpu_list, source_image, source_mesh, driving_meshes, data_per_node, generator, que=None):
    # torch.distributed.init_process_group(backend='nccl',init_method='tcp://127.0.0.1:3456',
    #                                         world_size=len(gpu_list), rank=rank)
    # generator = generator.to(gpu_list[rank])
    # generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[gpu_list[rank]])

    with torch.no_grad():
        predictions = []
        device = f'cuda:{gpu_list[rank]}'
        num_gpus = len(gpu_list)

        bs = 4 * num_gpus

        driving_meshes = driving_meshes[rank * data_per_node:(rank + 1) * data_per_node]

        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).repeat(bs, 1, 1, 1)
        source = source.to(device)

        kp_source = preprocess_dict([source_mesh] * bs, device=device)

        for frame_idx in tqdm(range(0, len(driving_meshes), bs)):
            kp_driving = preprocess_dict(driving_meshes[frame_idx:frame_idx+bs], device=device)
            if len(kp_driving['value']) < bs:
                kp_source = preprocess_dict([source_mesh] * len(kp_driving['value']), device=device)
                source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).repeat(len(kp_driving['value']), 1, 1, 1)
                source = source.to(device)
            kp_norm = kp_driving

            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1]))


    predictions = np.concatenate(predictions, axis=0)
    # predictions = np.ascontiguousarray(np.concatenate(predictions, axis=0)).astype(np.uint8).clip(0, 255))
    # que.put((rank, predictions))

    # torch.distributed.destroy_process_group()
    return predictions

def test_model(opt, generator, gpu_list):
    st = time.time()
    with open(opt.model_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['train_params']['num_kp'] = config['model_params']['common_params']['num_kp']
    config['train_params']['sections'] = config['model_params']['common_params']['sections']

    sections = config['train_params']['sections']
    estimate_jacobian = config['model_params']['common_params']['estimate_jacobian']

    source_image = imageio.imread(opt.source_image)

    section_indices = []
    sections_indices_splitted = []
    for sec in sections:
        section_indices.extend(sec[0])
        sections_indices_splitted.append(sec[0])
    
    fps = opt.fps

    frame_shape = config['dataset_params']['frame_shape']

    if len(source_image.shape) == 2:
        source_image = cv2.cvtColor(source_image, cv2.COLOR_GRAY2RGB)

    source_image = resize(img_as_float32(source_image), frame_shape[:2])[..., :3]

    L = frame_shape[0]
    SCALE = L // 2
    A = np.array([-1, -1, 0], dtype='float32')[:, np.newaxis] # 3 x 1

    driving_meshes = []
    target_meshes = []

    source_landmarks = torch.load(os.path.join(opt.driven_dir, 'src', '3d_landmarks.pt'))
    source_mesh = {}
    source_mesh['value'] = source_landmarks['3d_landmarks'].float() / SCALE
    right_iris = source_landmarks['normed_right_iris'].float() / SCALE
    left_iris = source_landmarks['normed_left_iris'].float() / SCALE
    source_mesh['value'][3] = right_iris
    source_mesh['value'][4] = left_iris
    source_mesh['raw_value'] = source_landmarks['3d_landmarks_pose']
    pose_p = source_landmarks['p']
    source_mesh['proj'] = pose_p['proj'].copy()
    source_mesh['view'] = pose_p['view'].copy()
    source_mesh['viewport'] = pose_p['viewport'].copy()
    source_mesh['R'] = pose_p['view'][:3, :3]
    source_mesh['t'] = pose_p['viewport'][:3, 3].copy()
    source_mesh['U'] = pose_p['U']
    source_mesh['scale'] = SCALE
    source_mesh['he_R'] = source_landmarks['he_p']['R']
    source_mesh['he_t'] = source_landmarks['he_p']['t']

    ### calc bias ###
    s = np.linalg.norm((source_mesh['raw_value'][45] - source_mesh['raw_value'][36]) / SCALE) / np.linalg.norm(source_mesh['value'][45] - source_mesh['value'][36])
    source_mesh['s'] = s
    b = np.linalg.inv(source_mesh['he_R']) @ ((1 / s) * (source_mesh['U'][3, :3] / SCALE - source_mesh['he_t'] - np.array([1, 1, 0]))[:, np.newaxis])
    source_mesh['b'] = b

    raw_mesh = source_mesh['raw_value']
    source_mesh['mesh_img_sec'] = get_mesh_image_section(raw_mesh, frame_shape, section_indices, sections_indices_splitted)

    driving_landmarks = torch.load(os.path.join(opt.driven_dir, 'drv', '3d_landmarks.pt'))
    num_of_frames = len(driving_landmarks)
    driving_meshes = []
    if opt.from_flame:
        driving_landmarks_from_flame = torch.load(os.path.join(opt.driven_dir, 'drv', '3d_landmarks_from_flame.pt')).float() * 1000
        num_of_frames = len(driving_landmarks_from_flame)
        from_flame_bias = -17
    else:
        driving_landmarks_from_flame = [driving_landmark['3d_landmarks'].float() for driving_landmark in driving_landmarks]
        from_flame_bias = 0

    ### eye drive ###
    ROI_EYE_IDX = list(range(17, 27)) + list(range(36, 42)) + list(range(42, 48))
    LEFT_EYE_LANDMARK_IDX = list(range(12, 18))
    LEFT_EYEBROW_LANDMARK_IDX = list(range(2, 7))
    LEFT_IRIS_LANDMARK_IDX = [0]
    RIGHT_EYE_LANDMARK_IDX = list(range(18, 24))
    RIGHT_EYEBROW_LANDMARK_IDX = list(range(7, 12))
    RIGHT_IRIS_LANDMARK_IDX = [1]
    LEFT_PART_IDX = LEFT_IRIS_LANDMARK_IDX + LEFT_EYEBROW_LANDMARK_IDX + LEFT_EYE_LANDMARK_IDX
    RIGHT_PART_IDX = RIGHT_IRIS_LANDMARK_IDX + RIGHT_EYEBROW_LANDMARK_IDX + RIGHT_EYE_LANDMARK_IDX
    LEFT_EYE_ANCHOR_IDX = [15]
    RIGHT_EYE_ANCHOR_IDX = [18]
    UPPER_EYES_IDX = [13, 14, 19, 20] 
    ONLY_EYES_IDX = LEFT_EYE_LANDMARK_IDX
    EYEBROW_LANDMARK_IDX = LEFT_IRIS_LANDMARK_IDX + RIGHT_IRIS_LANDMARK_IDX 
    # + LEFT_EYEBROW_LANDMARK_IDX + RIGHT_EYEBROW_LANDMARK_IDX
    # UPPER_EYES_IDX = ONLY_EYES_IDX

    ROI_EYE_IDX_FLAME = torch.tensor(ROI_EYE_IDX) + from_flame_bias
    drv_eyes = torch.stack([torch.cat([drv_lmk['normed_right_iris'] - torch.tensor([[0, 0, 0]]), drv_lmk['normed_left_iris'] - torch.tensor([[0, 0, 0]]), drv_lmk['3d_landmarks'][ROI_EYE_IDX].float()]) for drv_lmk in driving_landmarks]).float()  # B x n x 3
    drv_eyes_rel = torch.zeros_like(drv_eyes)
    print(f'drv eyes shape: {drv_eyes.shape}')
    print(f'drv eyes anchor shape: {drv_eyes[:, LEFT_EYE_ANCHOR_IDX].shape}')
    drv_eyes_rel[:, LEFT_PART_IDX] = drv_eyes[:, LEFT_PART_IDX] - drv_eyes[:, LEFT_EYE_ANCHOR_IDX]
    drv_eyes_rel[:, RIGHT_PART_IDX] = drv_eyes[:, RIGHT_PART_IDX] - drv_eyes[:, RIGHT_EYE_ANCHOR_IDX]
    mean_drv_eyes_rel = drv_eyes_rel.mean(dim=0)
    mean_drv_eyes = drv_eyes.mean(dim=0)    # n x 3

    src_eyes = torch.cat([source_landmarks['normed_right_iris'], source_landmarks['normed_left_iris'], source_landmarks['3d_landmarks'][ROI_EYE_IDX]]).float()    # n x 3
    src_eyes_rel  = torch.zeros_like(src_eyes)
    src_eyes_rel[LEFT_PART_IDX] = src_eyes[LEFT_PART_IDX] - src_eyes[LEFT_EYE_ANCHOR_IDX]
    src_eyes_rel[RIGHT_PART_IDX] = src_eyes[RIGHT_PART_IDX] - src_eyes[RIGHT_EYE_ANCHOR_IDX]
    eye_size_drv = mean_drv_eyes[ONLY_EYES_IDX].max(dim=0)[0] - mean_drv_eyes[ONLY_EYES_IDX].min(dim=0)[0] # 3
    eye_size_src = src_eyes[ONLY_EYES_IDX].max(dim=0)[0] - src_eyes[ONLY_EYES_IDX].min(dim=0)[0]  # 3
    eye_convert_scale = eye_size_src / eye_size_drv.clamp(min=1e-6)
    src_eye_width = src_eyes[LEFT_EYE_LANDMARK_IDX].max(dim=0)[0] - src_eyes[LEFT_EYE_LANDMARK_IDX].min(dim=0)[0]
    left_eye_center = (src_eyes[LEFT_EYE_LANDMARK_IDX].max(dim=0)[0] + src_eyes[LEFT_EYE_LANDMARK_IDX].min(dim=0)[0]) / 2
    right_eye_center = (src_eyes[RIGHT_EYE_LANDMARK_IDX].max(dim=0)[0] + src_eyes[RIGHT_EYE_LANDMARK_IDX].min(dim=0)[0]) / 2

    delta_eye_drv = drv_eyes_rel - mean_drv_eyes_rel[None]

    delta_pca = torch.load(opt.pca_path)
    # delta_pca = torch.pca_lowrank(delta_eye_drv.flatten(1), q=5, niter=10) # u, s, v
    # torch.save(delta_pca, f'pca_{time.time()}.pt')
    delta_eye_proj_coef = (delta_eye_drv.flatten(1) @ delta_pca[2]) @ torch.diag(delta_pca[1]).inverse()
    # print(f'PCA S: {delta_pca[1]}')

    k = 30
    T = 0.3
    delta_eye_proj_coef_nonlinear = k * (delta_eye_proj_coef ** 3)
    # delta_eye_proj_coef = -k * (torch.exp(-delta_eye_proj_coef / T) - 1)

    ## smooth movement
    filtered_items = []
    for i in range(delta_eye_proj_coef_nonlinear.shape[1]):
        filtered_item = torch.tensor(filter_values(delta_eye_proj_coef_nonlinear[:, i].numpy())).float()
        filtered_items.append(filtered_item)
    filtered_items = torch.stack(filtered_items, dim=1)
    delta_eye_proj_coef_nonlinear = filtered_items

    filtered_items = []
    for i in range(delta_eye_proj_coef.shape[1]):
        filtered_item = torch.tensor(filter_values(delta_eye_proj_coef[:, i].numpy())).float()
        filtered_items.append(filtered_item)
    filtered_items = torch.stack(filtered_items, dim=1)
    delta_eye_proj_coef = filtered_items

    delta_eye_proj_coef_nonlinear = (delta_eye_proj_coef_nonlinear @ torch.diag(delta_pca[1]) @ delta_pca[2].t()).view(delta_eye_drv.shape)
    delta_eye_proj = (delta_eye_proj_coef @ torch.diag(delta_pca[1]) @ delta_pca[2].t()).view(delta_eye_drv.shape)
    
    ### Disable PCA ###
    # delta_eye_proj = delta_eye_drv

    ### Disable Eye Size Scaling ###
    eye_convert_scale[[0, 2]] = torch.ones_like(eye_convert_scale[[0, 2]])

    print(f'eye convert scale: {eye_convert_scale}')
    delta_eye_drv_nonlinear = delta_eye_proj_coef_nonlinear
    delta_eye_drv = delta_eye_proj
    
    eyes_drvn_rel = src_eyes_rel[None] + 0 * delta_eye_drv # B x n x 3
    eyes_drvn_rel[:, UPPER_EYES_IDX] = src_eyes_rel[None][:, UPPER_EYES_IDX] + eye_convert_scale[None, None] * delta_eye_drv_nonlinear[:, UPPER_EYES_IDX]  # B x n x 3
    eyes_drvn_rel[:, EYEBROW_LANDMARK_IDX] = src_eyes_rel[None][:, EYEBROW_LANDMARK_IDX] + 1 * delta_eye_drv[:, EYEBROW_LANDMARK_IDX]  # B x n x 3
    eyes_drvn = torch.zeros_like(eyes_drvn_rel)
    eyes_drvn[:, LEFT_PART_IDX] += src_eyes[None][:, LEFT_EYE_ANCHOR_IDX] + eyes_drvn_rel[:, LEFT_PART_IDX]
    eyes_drvn[:, RIGHT_PART_IDX] += src_eyes[None][:, RIGHT_EYE_ANCHOR_IDX] + eyes_drvn_rel[:, RIGHT_PART_IDX]
    
   ## iris_movement adaption
    left_iris_move = eyes_drvn[:, LEFT_IRIS_LANDMARK_IDX]
    adapted_left_iris_move_x = adapt_values(left_eye_center[0], left_iris_move[:, 0, 0], rel_minimum=(-src_eye_width[0] / 12), rel_maximum=src_eye_width[0] / 12)
    eyes_drvn[:, LEFT_IRIS_LANDMARK_IDX, 0] = adapted_left_iris_move_x.unsqueeze(-1)
    right_iris_move = eyes_drvn[:, RIGHT_IRIS_LANDMARK_IDX]
    adapted_right_iris_move_x = adapt_values(right_eye_center[0], right_iris_move[:, 0, 0], rel_minimum=(-src_eye_width[0] / 24), rel_maximum=src_eye_width[0] / 24)
    eyes_drvn[:, RIGHT_IRIS_LANDMARK_IDX, 0] = adapted_right_iris_move_x.unsqueeze(-1)

    ## iris_movement adaption
    left_iris_move = eyes_drvn[:, LEFT_IRIS_LANDMARK_IDX]
    adapted_left_iris_move_y = adapt_values(left_eye_center[1], left_iris_move[:, 0, 1], rel_minimum=(-src_eye_width[1] / 12), rel_maximum=src_eye_width[1] / 12)
    eyes_drvn[:, LEFT_IRIS_LANDMARK_IDX, 1] = adapted_left_iris_move_y.unsqueeze(-1)
    right_iris_move = eyes_drvn[:, RIGHT_IRIS_LANDMARK_IDX]
    adapted_right_iris_move_y = adapt_values(right_eye_center[1], right_iris_move[:, 0, 1], rel_minimum=(-src_eye_width[1] / 24), rel_maximum=src_eye_width[1] / 24)
    eyes_drvn[:, RIGHT_IRIS_LANDMARK_IDX, 1] = adapted_right_iris_move_y.unsqueeze(-1)


    ## iris_movement adaption
    eyes_drvn[:, LEFT_IRIS_LANDMARK_IDX, 2] =  eyes_drvn[0, LEFT_IRIS_LANDMARK_IDX, 2].squeeze()
    eyes_drvn[:, RIGHT_IRIS_LANDMARK_IDX, 2] = eyes_drvn[0, RIGHT_IRIS_LANDMARK_IDX, 2].squeeze()

    for i, driving_landmark in enumerate(driving_landmarks_from_flame):
        driven_pose_index = min(2 * len(driving_landmarks) - 1  - i % (2 * len(driving_landmarks)), i % (2 * len(driving_landmarks)))
        mesh = {}
        ROI_IDX = list(range(48, 68))
        ROI_IDX = torch.tensor(ROI_IDX)
        ROI_IDX_FLAME = ROI_IDX + from_flame_bias

        target_landmarks = torch.tensor(source_landmarks['3d_landmarks']).float()
        target_landmarks[ROI_IDX] = driving_landmark[ROI_IDX_FLAME]

        ### apply eye movement ###
        target_landmarks[[3, 4] + ROI_EYE_IDX] = eyes_drvn[driven_pose_index]
        # target_landmarks[[3, 4]] = source_mesh['value'][[3, 4]] * SCALE


        # mesh['value'] = source_mesh['value']
        mesh['value'] = target_landmarks.float() / SCALE
        mesh['raw_value'] = torch.tensor(source_landmarks['3d_landmarks_pose'])

        ### manipulate head pose ###
        driving_pose = driving_landmarks[driven_pose_index]['he_p']

        rot_src = pose_p['view'].copy()
        trans_src = pose_p['viewport'].copy()

        r = R.from_matrix(rot_src[:3, :3])
        MAX_ROT = np.pi / 3
        angle_x = 0
        angle_y = 0
        angle_z = 0
        delta_angle = np.array([angle_x, angle_y, angle_z])
        delta_r = R.from_rotvec(delta_angle)
        final_r = delta_r.as_matrix() @ r.as_matrix()

        # use driving R
        final_r = driving_pose['R']
        rot_src[:3, :3] = final_r
        mesh['R'] = final_r

        trans_x = 0
        trans_y = 0
        trans_z = 0
        final_trans = np.array([trans_x, trans_y, trans_z])

        # use driving trans
        final_trans = driving_pose['t'][:3]
        mesh['t'] = final_trans
        mesh['scale'] = SCALE
        
        driving_meshes.append(mesh)


    # use one euro filter for denoising
    filter_mesh(driving_meshes, source_mesh, SCALE)
    target_meshes = []

    ## split inputs
    # NODES = len(gpu_list)
    NODES = 1
    data_per_node = math.ceil(len(driving_meshes) / NODES)

    for mesh in driving_meshes:
        raw_mesh = torch.cat([mesh['value'] * SCALE, torch.ones(mesh['value'].shape[0], 1)], dim=1).matmul(mesh['U']).int()
        mesh['raw_mesh'] = raw_mesh
        mesh['mesh_img_sec'] = get_mesh_image_section(raw_mesh, frame_shape, section_indices, sections_indices_splitted)
        target_meshes.append(raw_mesh[section_indices])

    # que = mp.Manager().Queue()

    # mp.set_start_method('spawn', force=True)
    # processes = []
    # for i, generator in enumerate(list_generator):
    #     p = mp.Process(target=make_animation, args=(source_image, source_mesh, driving_meshes[i * data_per_node:(i + 1) * data_per_node], generator, i, que))
    #     p.start()
    #     processes.append(p)

    # print(f'preprocessing time: {time.time() - st}')
    # torch.multiprocessing.spawn(make_animation, nprocs=NODES, args=(gpu_list, source_image, source_mesh, driving_meshes, data_per_node, generator, que))
   
    
    # preds = {}
    # for i in range(NODES):
    #     r, pred = que.get()
    #     preds[r] = pred
    # predictions = np.concatenate([preds[i] for i in range(NODES)], axis=0)

    # del preds

    predictions = make_animation(0, gpu_list, source_image, source_mesh, driving_meshes, data_per_node, generator)

    # predictions = output['prediction']
    
    # mesh styling
    meshed_frames = []

    for i, frame in enumerate(predictions):
        frame = np.ascontiguousarray(img_as_ubyte(frame))
        if i >= len(target_meshes):
            continue
        mesh = target_meshes[i]
        # frame = draw_section(mesh[:, :2].numpy().astype(np.int32), frame_shape, section_config=sections_indices_splitted, mask=frame)
        meshed_frames.append(frame)

    predictions = meshed_frames

    imageio.mimsave(os.path.join(opt.result_dir, opt.result_video), predictions, fps=fps)