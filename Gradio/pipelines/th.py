import sys
import os
import copy
import yaml
import torch
import soundfile as sf
import BFv2v.export as bfv2v
import FaceFormer.export as faceformer
from pathlib import Path
import json
import glob
import librosa
import importlib
import numpy as np
import soundfile as sf
from scipy.io.wavfile import read
#######################################
from common.abstract import AbstractPipeline
from common.misc import *
#######################################

class THPipeline(AbstractPipeline):

    def __init__(self, config, gpus=None):
        super(THPipeline, self).__init__(config, gpus)
        assert type(gpus) == list or type(gpus) == int, 'gpus should be list or integer'
        self.gpus = gpus if type(gpus) == list else [gpus]
        self.print('TH module has been initialized.')
        self.set_twin(twin_id='temporal')
    def initialize(self, task):
        ################################
        torch.cuda.empty_cache()
        ################################
        if task == 'preprocess':
            self.landmark_model = bfv2v.load_landmark_model(self.config.TH.common.checkpoints.landmark_model.dir, self.gpus)
            self.he_estimator = bfv2v.load_he_estimator(self.config.TH.common.checkpoints.he_estimator.config, self.config.TH.common.checkpoints.he_estimator.model, self.gpus)
        elif task == 'inference':
            self.faceformer = faceformer.load_faceformer(self.config.TH.inference.attr.faceformer, self.config.TH.common.checkpoints.faceformer.model, self.gpus)
            self.bfv2v = bfv2v.load_bfv2v(self.config.TH.common.checkpoints.bfv2v.config, self.config.TH.common.checkpoints.bfv2v.model, self.gpus)
            self.he_estimator = bfv2v.load_he_estimator(self.config.TH.common.checkpoints.he_estimator.config, self.config.TH.common.checkpoints.he_estimator.model, self.gpus)
            self.landmark_model = bfv2v.load_landmark_model(self.config.TH.common.checkpoints.landmark_model.dir, self.gpus)

    def preprocess(self, **kargs):
        try:
            drv_path = kargs['base_video_path']

            ## 0. Construct Preprocess Dir
            preprocess_path = self.config.TH.preprocess.output.save_path
            drv_landmark_dir = os.path.join(preprocess_path, 'drv')
            processed_drv_path = os.path.join(drv_landmark_dir, 'drv.mp4')
            
            if not os.path.exists(drv_landmark_dir):
                os.makedirs(drv_landmark_dir)
            
            ## 1. Crop and Resize source image, driving video
            drv_fps30_path = os.path.join(drv_landmark_dir, '30_drv.mp4')
            os.system(f"ffmpeg -y -i {drv_path} -vf fps=30 {drv_fps30_path}")
            cmds = self.landmark_model.preprocess_video(drv_fps30_path, processed_drv_path)

            ## 2. Extract landmarks
            bfv2v.extract_landmark_from_video(processed_drv_path, drv_landmark_dir, self.he_estimator, self.landmark_model)
            
            return {
                'status': self.config.const.SUCCESS
            }
        except Exception as e:
            traceback.print_exc()
            return {
                'status': self.config.const.FAILED
            }

    def train(self, **kargs):
        pass

    def finetune(self, **kargs):
        pass

    def inference(self, **kargs):

        try:
            src_path =  kargs['input_image_path']
            audio_path =  kargs['input_audio_path']
            is_high_resolution = kargs['is_high_resolution']
            output_name = kargs['output_name']
            ## 0. Setup Directories
            tmp_dir = self.config.TH.preprocess.output.save_path
            preprocess_path = self.config.TH.preprocess.output.save_path
            src_dir = os.path.join(preprocess_path, 'src')
            drv_dir = os.path.join(preprocess_path, 'drv')
            audio_dir = os.path.join(preprocess_path, 'audio')
            processed_drv_path = os.path.join(preprocess_path, 'drv', f'drv.mp4')
            processed_src_path = os.path.join(src_dir, 'src.png')
            processed_audio_path = os.path.join(audio_dir, 'audio.wav')
            output_file_path = self.config.TH.inference.output.save_path
            output_dir = os.path.dirname(output_file_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            ## 1. Process Image / Audio
            if not os.path.exists(src_dir):
                os.makedirs(src_dir)
            if not os.path.exists(audio_dir):
                os.makedirs(audio_dir)
            
            ## 1 - 0. Crop and Resize source image, driving video
            ######################
            # import pdb; pdb.set_trace()
            self.landmark_model.preprocess_image(src_path, processed_src_path)
            ######################
            os.system(f"ffmpeg -y -i {audio_path} {processed_audio_path}")

            ## 1 - 1. Extract landmarks
            bfv2v.extract_landmark_from_img(processed_src_path, src_dir, self.he_estimator, self.landmark_model)

            ## 2. Run Faceformer to predict face meshes corresponding to audio
            run_args = copy.copy(self.config.TH.inference.attr.faceformer)
            run_args.result_path = drv_dir
            run_args.wav_path = processed_audio_path
            run_args.id_landmarks_path = os.path.join(src_dir, '3d_landmarks.pt')
            faceformer.test_model(run_args, self.faceformer)

            ## 3. Run BFv2v to warp and generate video
            args_run = copy.copy(self.config.TH.inference.attr.bfv2v)
            args_run.model_config = self.config.TH.common.checkpoints.bfv2v.config
            args_run.source_image = processed_src_path
            args_run.driving_video = processed_drv_path
            args_run.driven_dir = tmp_dir
            args_run.result_dir = tmp_dir
            args_run.result_video = 'mute.mp4'
            args_run.fps = self.config.TH.common.attr.fps
            bfv2v.test_model(args_run, self.bfv2v, self.gpus)

            ## 4. Post Process
            # add audio
            SIZE = 256 if is_high_resolution else 128
            if output_name is not None:
                output_file_path = output_name
            os.system(f"ffmpeg -y -i {os.path.join(args_run.result_dir, args_run.result_video)} -i {processed_audio_path} -map 0:v:0 -map 1:a:0 -filter:v 'fps={self.config.TH.common.attr.fps}' {output_file_path}")
            
            return {
                'status': self.config.const.SUCCESS,
                'output': output_file_path
            }
        except Exception as e:
            traceback.print_exc()
            return {
                'status': self.config.const.FAILED
            }