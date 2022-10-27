import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset3

from modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from modules.discriminator import Discriminator, MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector, HEEstimator, ExpTransformer

import torch

from train import train_transformer
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    
    # torch.multiprocessing.set_start_method('spawn')

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
    parser.add_argument("--mode", default="train", choices=["train",])
    parser.add_argument("--stage", default=1, type=int, choices=[1, 2, -1])
    parser.add_argument("--gen", default="spade", choices=["original", "spade"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    # parser.add_argument("--checkpoint_ref", default='/home/server19/minyeong_workspace/MDTH/models/fv2v2/log/img_mesh_logloss 11_10_22_09.07.39/last.tar', help="path to checkpoint to restore")
    parser.add_argument("--checkpoint_ref_gen", default=None, help="path to checkpoint to restore")
    parser.add_argument("--checkpoint_ref_he", default=None, help="path to checkpoint to restore")
    
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # if opt.mode == 'train':
        # config['train_params']['num_kp'] = config['model_params']['common_params']['num_kp']
        # config['train_params']['sections'] = config['model_params']['common_params']['sections']
        
    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())
        

    if opt.gen == 'original':
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
    elif opt.gen == 'spade':
        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params'])

    # if torch.cuda.is_available():
    #     print('cuda is available')
    #     generator.to(opt.device_ids[0])
    # if opt.verbose:
    #     print(generator)
    generator = None
    
    # discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
    #                                         **config['model_params']['common_params'])
    # if torch.cuda.is_available():
    #     discriminator.to(opt.device_ids[0])
        
    discriminator = None
    # if opt.verbose:
    #     print(discriminator)

    # kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
    #                            **config['model_params']['common_params'])
    # if torch.cuda.is_available():
    #     kp_detector.to(opt.device_ids[0])

    # if opt.verbose:
    #     print(kp_detector)

    # he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
    #                            **config['model_params']['common_params'])

    # if torch.cuda.is_available():
    #     he_estimator.to(opt.device_ids[0])

    he_estimator = None
    
    exp_transformer = ExpTransformer(**config['model_params']['exp_transformer_params'],
                               **config['model_params']['common_params'])
    
    if torch.cuda.is_available():
       exp_transformer.to(opt.device_ids[0])

    dataset = FramesDataset3(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    ckpt_ref = {}
    if opt.checkpoint_ref_gen is not None:
        ckpt_ref['generator'] = opt.checkpoint_ref_gen
    if opt.checkpoint_ref_he is not None:
        ckpt_ref['he_estimator'] = opt.checkpoint_ref_he
        
        
    if opt.mode == 'train':
        print(f"Training with stage {opt.stage}...")
        train_transformer(config, opt.stage, exp_transformer, generator, discriminator, None, he_estimator, opt.checkpoint, ckpt_ref, log_dir, dataset, opt.device_ids)
