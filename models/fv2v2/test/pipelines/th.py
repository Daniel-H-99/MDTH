from codecs import ignore_errors
import sys
import os
import copy
import torch
import pathlib
import shutil
import imageio
import numpy as np

root_dir = str(pathlib.Path(__file__).parent / '..' / '..')
sys.path.insert(0, root_dir)
import export
#######################################
#######################################

class THPipeline():
    def __init__(self, config, gpus):
        self.config = config
        self.gpus = gpus
        self.initialize()
        
    def process_name(self, name):
        return name.replace('/', '%')
    
    def initialize(self):
        ################################
        torch.cuda.empty_cache()
        ################################
        self.exp_transformer = export.load_exp_transformer(self.config.config.common.checkpoints.exp_transformer.config, self.config.config.common.checkpoints.exp_transformer.model, self.gpus)
        self.generator = export.load_generator(self.config.config.common.checkpoints.generator.config, self.config.config.common.checkpoints.generator.model, self.gpus)
        self.he_estimator = export.load_he_estimator(self.config.config.common.checkpoints.he_estimator.config, self.config.config.common.checkpoints.he_estimator.model, self.gpus)
        self.landmark_model = export.load_landmark_model(self.config.config.common.checkpoints.landmark_model.dir, self.gpus)
        self.kp_extractor = None
        
    def preprocess_image(self, img_name, rewrite=False, preprocess=True, extract_landmarks=True):
        src_dir = self.config.config.preprocess.input.dir
        dest_dir = self.config.config.preprocess.output.dir
        src_path = os.path.join(src_dir, img_name)
        dest_path = os.path.join(dest_dir, self.process_name(img_name))
        dest_file_path = os.path.join(dest_path, 'image.png')
        if preprocess:
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            elif rewrite:
                shutil.rmtree(dest_path, ignore_errors=True)
                os.makedirs(dest_path)
            self.landmark_model.preprocess_image(src_path, dest_file_path)
        if extract_landmarks:
            export.extract_landmark_from_img(dest_path, self.he_estimator, self.landmark_model, rewrite=rewrite)
            
    def preprocess_video(self, video_name, rewrite=False, preprocess=True, extract_landmarks=True):
        src_dir = self.config.config.preprocess.input.dir
        dest_dir = self.config.config.preprocess.output.dir
        src_path = os.path.join(src_dir, video_name)
        dest_path = os.path.join(dest_dir, self.process_name(video_name))
        dest_file_path = os.path.join(dest_path, 'video.mp4')
        if preprocess:
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            elif rewrite:
                shutil.rmtree(dest_path, ignore_errors=True)
                os.makedirs(dest_path)
            self.landmark_model.preprocess_video(src_path, dest_file_path)

            frames_dir = os.path.join(dest_path, 'frames')
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
            os.makedirs(frames_dir)
            # save each frames in frames directory
            reader = imageio.get_reader(os.path.join(dest_file_path))
            fps = reader.get_meta_data()['fps']
            driving_video = []
            try:
                for im in reader:
                    driving_video.append(im)
            except RuntimeError:
                pass
            reader.close()
            
            for i, frame in enumerate(driving_video):
                imageio.imwrite(os.path.join(frames_dir, '{:05d}.png'.format(i)), frame)
                
            
        print(f'extract_landmarks: {extract_landmarks}')
        if extract_landmarks:
            rewritten = export.extract_landmark_from_video(dest_path, self.he_estimator, self.landmark_model, rewrite=rewrite)
        else:
            rewritten = False
    

            

    def inference(self, src_name, drv_name, output_dir, use_transformer=True, extract_driving_code=False, stage=1, relative_headpose=True, save_frames=True):
        src_name = self.process_name(src_name)
        drv_name = self.process_name(drv_name)
        output_name = '_'.join([src_name, drv_name])
        input_dir = self.config.config.inference.input.dir
        output_dir = os.path.join(self.config.config.inference.output.dir, output_dir)
        src_path = os.path.join(input_dir, src_name)
        drv_path = os.path.join(input_dir, drv_name)
        output_path = os.path.join(output_dir, output_name)
        output_file_path = os.path.join(output_path, output_name + '.mp4')
        frames_dir = os.path.join(output_path, 'frames')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        os.makedirs(frames_dir)
        ## leave inputs (src_path, drv_path) as inputs.txt
        inputs = [src_path, drv_path]
        np.savetxt(os.path.join(output_path, 'inputs.txt'), inputs, fmt='%s', comments=None)
        # with open(os.path.join(output_path, 'inputs.txt'), 'w') as f:
        #     f.writelines([src_path+'\r\n', drv_path+'\r\n'])
            
        args_run = copy.copy(self.config.config.inference.attr)
        args_run.config = self.config.config.common.checkpoints.exp_transformer.config
        args_run.source_dir = src_path
        args_run.driving_dir = drv_path
        args_run.result_dir = output_path
        args_run.result_video = 'mute.mp4'
        args_run.fps = self.config.config.common.attr.fps
        export.test_model(args_run, self.generator, self.exp_transformer, self.kp_extractor, self.he_estimator, self.gpus, use_transformer=use_transformer, extract_driving_code=extract_driving_code, stage=stage, relative_headpose=relative_headpose, save_frames=save_frames)

        ## 4. Post Process
        # add audio
        SIZE = 256

        if os.path.exists(os.path.join(drv_path, 'video.mp4')):
            os.system(f"ffmpeg -y -i {os.path.join(args_run.result_dir, args_run.result_video)} -i {os.path.join(drv_path, 'video.mp4')} -map 0:v:0 -map 1:a:0 -filter:v 'fps={self.config.config.common.attr.fps}' {output_file_path}")

        else:
            os.system(f"ffmpeg -y -i {os.path.join(args_run.result_dir, args_run.result_video)} -filter:v 'fps={self.config.config.common.attr.fps}' {output_file_path}")
        
        return output_name
        
    def inference_expression(self, src_name, drv_name, output_dir):
        src_name = self.process_name(src_name)
        ### construct object of driving info
        drv_exp = np.loadtxt(drv_name, dtype=str)
        driving_info = {}
        for line in drv_exp:
            k, v = line.split(',')
            driving_info[k] = int(v)
        driving_info = AttrDict.from_nested_dicts(driving_info)
        ###
        drv_name = os.path.basename(drv_name)
        output_name = '_'.join([src_name, drv_name])
        input_dir = self.config.config.inference.input.dir
        output_dir = os.path.join(self.config.config.inference.output.dir, output_dir)
        src_path = os.path.join(input_dir, src_name)
        output_path = os.path.join(output_dir, output_name)
        output_file_path = os.path.join(output_path, output_name + '.mp4')

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        os.makedirs(frames_dir)
        ## leave inputs (src_path, drv_path) as inputs.txt
        inputs = [src_path, drv_name]
        np.savetxt(os.path.join(output_path, 'inputs.txt'), inputs, fmt='%s', comments=None)
        # with open(os.path.join(output_path, 'inputs.txt'), 'w') as f:
        #     f.writelines([src_path+'\r\n', drv_path+'\r\n'])
            
        args_run = copy.copy(self.config.config.inference.attr)
        args_run.config = self.config.config.common.checkpoints.exp_transformer.config
        args_run.source_dir = src_path
        args_run.driving_expression = driving_info
        args_run.result_dir = output_path
        args_run.result_video = 'video.mp4'
        args_run.fps = self.config.config.common.attr.fps
        export.test_model_with_exp(args_run, self.generator, self.exp_transformer, self.kp_extractor, self.he_estimator, self.gpus)

        return output_name