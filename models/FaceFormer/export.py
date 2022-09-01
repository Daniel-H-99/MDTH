import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import tempfile
import pyrender
import trimesh
import pickle as pkl
import os 
import numpy as np
import numpy as np
import scipy.io.wavfile as wav
import librosa
import os,sys,shutil,argparse,copy,pickle
import math,scipy
from scipy.spatial.transform import Rotation as R
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
from subprocess import call
from psbody.mesh import Mesh
from smplx.lbs import lbs, batch_rodrigues, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords
from smplx.utils import to_tensor
###############################################################
from FaceFormer.fit_lmk3d import run_fitting as fit_lmk3d
from FaceFormer.faceformer import Faceformer
from FaceFormer.smpl_webuser.serialization import load_model
from FaceFormer.fitting.landmarks import load_embedding, landmark_error_3d, mesh_points_by_barycentric_coordinates
###############################################################

def load_faceformer(args, checkpoint_path, gpu=[0]):
    args['device'] = gpu[0]
    model = Faceformer(args)
    model.load_state_dict(torch.load(checkpoint_path, map_location=f'cuda:{gpu[0]}'))
    model = model.to(gpu[0])
    model.eval()
    return model

@torch.no_grad()
def test_model(args, model):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    template_file = args.template_path

    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    one_hot_labels = np.eye(len(train_subjects_list))
    iter = train_subjects_list.index(args.condition)
    one_hot = one_hot_labels[iter]
    one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device=args.device)

    template_mesh = Mesh(filename=template_file)

    template = template_mesh.v.reshape((-1))
    template = np.reshape(template,(-1,template.shape[0]))
    template = torch.FloatTensor(template).to(device=args.device)

    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

    prediction = model.predict(audio_feature, template, one_hot)
    prediction = prediction.squeeze() # (seq_len, V*3)
    prediction = prediction.view(len(prediction), -1, 3).detach().cpu().numpy()

    if args.id_landmarks_path is not None:
        id_vertices = torch.tensor(torch.load(args.id_landmarks_path)['3d_landmarks']).float() / 1000
    else:
        assert False
        id_vertices = None

    if args.params_path is not None:
        with open(args.params_path, 'rb') as f:
            params = pkl.load(f)
    else:
        params = None
        assert False


    # if args.add_eye_blink:
    #     prediction, prediction_blink = add_eye_blink(prediction.detach().cpu(), args.generic_model_path, args.num_blinks, args.blink_duration, params=params)

    template_vertices = template.view(-1, 3).cpu().numpy()

    if params is not None:
        flame_model = load_model(args.generic_model_path)
        vertices = np.concatenate([prediction, template_vertices[None]], axis=0)
        predicted_vertices = np.zeros((len(vertices), flame_model.v_template.shape[0], flame_model.v_template.shape[1]))
        for frame_idx in range(len(vertices)):
            flame_model.v_template[:] = vertices[frame_idx]
            if params is not None:
                if 'pose' in params:
                    flame_model.pose[:] = params['pose'][:]
                if 'trans' in params:
                    flame_model.trans[:] = params['trans'][:]
            
            wo_eye_blinked_v = flame_model.r
            predicted_vertices[frame_idx] = wo_eye_blinked_v
        
        prediction = torch.tensor(predicted_vertices[:-1]).float()
        template_vertices = np.array(predicted_vertices[-1]).astype(np.float32)
    else:
        assert False

    np.save(os.path.join(args.result_path, 'vertices.npy'), prediction)

    if args.static_landmark_embedding_path is not None:
        with open(args.static_landmark_embedding_path, 'rb') as f:
            static_embeddings = pickle.load(f, encoding='latin1')

        lmk_faces_idx = torch.tensor(static_embeddings['lmk_face_idx'].astype(np.int64))
        lmk_bary_coords = torch.tensor(static_embeddings['lmk_b_coords'])

        landmarks = [np.array(mesh_points_by_barycentric_coordinates(torch.tensor(pred), to_tensor(template_mesh.f.astype(np.int64)).long(), lmk_faces_idx.long(), lmk_bary_coords.float())) for pred in prediction]
        landmarks = torch.tensor(np.array(landmarks)).float()

        # mix with id
        if id_vertices is not None:
            # print(f'landmarks: {landmarks[0]}')
            # print(f'id landmarks: {id_vertices[17:]}')

            template_landmark = torch.tensor(np.array(mesh_points_by_barycentric_coordinates(torch.tensor(template_vertices), to_tensor(template_mesh.f.astype(np.int64)).long(), lmk_faces_idx.long(), lmk_bary_coords.float()))).float()
            landmarks = id_vertices[17:][None].repeat(len(landmarks), 1, 1) + (landmarks - template_landmark[None])

        torch.save(landmarks, os.path.join(args.result_path, '3d_landmarks_from_flame.pt'))

def add_eye_blink(vertices, flame_model_fname, num_blinks, blink_duration, uv_template_fname='', texture_img_fname='', params=None):
    '''
    Load existing animation sequence in "zero pose" and add eye blinks over time
    :param source_path:         path of the animation sequence (files must be provided in OBJ file format)
    :param out_path:            output path of the altered sequence
    :param flame_model_fname:   path of the FLAME model
    :param num_blinks:          number of blinks within the sequence
    :param blink_duration:      duration of a blink in number of frames
    '''

    # Load sequence files
    vertices = vertices.numpy()
    num_frames = len(vertices)
    if num_frames == 0:
        print('No sequence meshes found')
        return None

    # Load FLAME head model
    model = load_model(flame_model_fname)

    blink_exp_betas = np.array(
        [0.04676158497927314, 0.03758675711005459, -0.8504121184951298, 0.10082324210507627, -0.574142329926028,
         0.6440016589938355, 0.36403779939335984, 0.21642312586261656, 0.6754551784690193, 1.80958618462892,
         0.7790133813372259, -0.24181691256476057, 0.826280685961679, -0.013525679499256753, 1.849393698014113,
         -0.263035686247264, 0.42284248271332153, 0.10550891351425384, 0.6720993875023772, 0.41703592560736436,
         3.308019065485072, 1.3358509602858895, 1.2997143108969278, -1.2463587328652894, -1.4818961382824924,
         -0.6233880069345369, 0.26812528424728455, 0.5154889093160832, 0.6116267181402183, 0.9068826814583771,
         -0.38869613253448576, 1.3311776710005476, -0.5802565274559162, -0.7920775624092143, -1.3278601781150017,
         -1.2066425872386706, 0.34250140710360893, -0.7230686724732668, -0.6859285483325263, -1.524877347586566,
         -1.2639479212965923, -0.019294228307535275, 0.2906175769381998, -1.4082782880837976, 0.9095436721066045,
         1.6007365724960054, 2.0302381182163574, 0.5367600947801505, -0.12233184771794232, -0.506024823810769,
         2.4312326730634783, 0.5622323258974669, 0.19022395712837198, -0.7729758559103581, -1.5624233513002923,
         0.8275863297957926, 1.1661887586553132, 1.2299311381779416, -1.4146929897142397, -0.42980549225554004,
         -1.4282801579740614, 0.26172301287347266, -0.5109318114918897, -0.6399495909195524, -0.733476856285442,
         1.219652074726591, 0.08194907995352405, 0.4420398361785991, -1.184769973221183, 1.5126082924326332,
         0.4442281271081217, -0.005079477284341147, 1.764084274265486, 0.2815940264026848, 0.2898827213634057,
         -0.3686662696397026, 1.9125365942683656, 2.1801452989500274, -2.3915065327980467, 0.5794919897154226,
         -1.777680085517591, 2.9015718628823604, -2.0516886588315777, 0.4146899057365943, -0.29917763685660903,
         -0.5839240983516372, 2.1592457102697007, -0.8747902386178202, -0.5152943072876817, 0.12620001057735733,
         1.3144109838803493, -0.5027032013330108, 1.2160353388774487, 0.7543834001473375, -3.512095548974531,
         -0.9304382646186183, -0.30102930208709433, 0.9332135959962723, -0.52926196689098, 0.23509772959302958])

    step = blink_duration//3
    blink_weights = np.hstack((np.interp(np.arange(step), [0,step], [0,1]), np.ones(step), np.interp(np.arange(step), [0,step], [1,0])))

    frequency = num_frames // (num_blinks+1)
    weights = np.zeros(num_frames)
    for i in range(num_blinks):
        x1 = (i+1)*frequency-blink_duration//2
        x2 = x1+3*step
        if x1 >= 0 and x2 < weights.shape[0]:
            weights[x1:x2] = blink_weights *3

    predicted_vertices = np.zeros((num_frames, model.v_template.shape[0], model.v_template.shape[1]))
    predicted_vertices_blink = np.zeros((num_frames, model.v_template.shape[0], model.v_template.shape[1]))

    for frame_idx in range(num_frames):
        model.v_template[:] = vertices[frame_idx]
        model.betas[300:] = weights[frame_idx]*blink_exp_betas
        if params is not None:
            if 'pose' in params:
                model.pose[:] = params['pose'][:]
            if 'trans' in params:
                model.trans[:] = params['trans'][:]
        
        wo_eye_blinked_v = model.r

        model.v_template[:] = vertices[frame_idx]
        model.betas[300:] = weights[frame_idx]*blink_exp_betas
        if params is not None:
            if 'pose' in params:
                model.pose[:] = params['pose'][:]
            if 'trans' in params:
                model.trans[:] = params['trans'][:]
        
        w_eye_blinked_v = model.r

        predicted_vertices[frame_idx] = wo_eye_blinked_v
        predicted_vertices_blink[frame_idx] = w_eye_blinked_v

    
    return torch.tensor(predicted_vertices).float(), torch.tensor(predicted_vertices_blink).float()
