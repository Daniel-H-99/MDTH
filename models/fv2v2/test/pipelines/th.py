from codecs import ignore_errors
import sys
import os
import copy
import torch
import pathlib
import shutil
import imageio

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
    def preprocess_image(self, img_name, rewrite=False):
        src_dir = self.config.config.preprocess.input.dir
        dest_dir = self.config.config.preprocess.output.dir
        src_path = os.path.join(src_dir, img_name)
        dest_path = os.path.join(dest_dir, self.process_name(img_name))
        dest_file_path = os.path.join(dest_path, 'image.png')
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        elif rewrite:
            shutil.rmtree(dest_path, ignore_errors=True)
            os.makedirs(dest_path)
        self.landmark_model.preprocess_image(src_path, dest_file_path)
        export.extract_landmark_from_img(dest_path, self.he_estimator, self.landmark_model, rewrite=rewrite)
            
    def preprocess_video(self, video_name, rewrite=False):
        src_dir = self.config.config.preprocess.input.dir
        dest_dir = self.config.config.preprocess.output.dir
        src_path = os.path.join(src_dir, video_name)
        dest_path = os.path.join(dest_dir, self.process_name(video_name))
        dest_file_path = os.path.join(dest_path, 'video.mp4')
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        elif rewrite:
            shutil.rmtree(dest_path, ignore_errors=True)
            os.makedirs(dest_path)
        self.landmark_model.preprocess_video(src_path, dest_file_path)
        rewritten = export.extract_landmark_from_video(dest_path, self.he_estimator, self.landmark_model, rewrite=rewrite)
        
        if rewritten:
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
            

    def inference(self, src_name, drv_name, output_dir, use_transformer=True, extract_driving_code=False):
        src_name = self.process_name(src_name)
        drv_name = self.process_name(drv_name)
        output_name = '_'.join([src_name, drv_name])
        input_dir = self.config.config.inference.input.dir
        output_dir = os.path.join(self.config.config.inference.output.dir, output_dir)
        src_path = os.path.join(input_dir, src_name)
        drv_path = os.path.join(input_dir, drv_name)
        output_path = os.path.join(output_dir, output_name)
        output_file_path = os.path.join(output_path, output_name + '.mp4')
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        ## leave inputs (src_path, drv_path) as inputs.txt
        with open(os.path.join(output_path, 'inputs.txt'), 'w') as f:
            f.writelines([src_path, drv_path])
            
        args_run = copy.copy(self.config.config.inference.attr)
        args_run.config = self.config.config.common.checkpoints.exp_transformer.config
        args_run.source_dir = src_path
        args_run.driving_dir = drv_path
        args_run.result_dir = output_path
        args_run.result_video = 'mute.mp4'
        args_run.fps = self.config.config.common.attr.fps
        export.test_model(args_run, self.generator, self.exp_transformer, self.kp_extractor, self.he_estimator, self.gpus, use_transformer=use_transformer, extract_driving_code=extract_driving_code)

        ## 4. Post Process
        # add audio
        SIZE = 256

        os.system(f"ffmpeg -y -i {os.path.join(args_run.result_dir, args_run.result_video)} -i {os.path.join(drv_path, 'video.mp4')} -map 0:v:0 -map 1:a:0 -filter:v 'fps={self.config.config.common.attr.fps}' {output_file_path}")
        
            
        
        # ## 0. Setup Directories
        # tmp_dir = self.config.TH.preprocess.output.save_path
        # preprocess_path = self.config.TH.preprocess.output.save_path
        # src_dir = os.path.join(preprocess_path, 'src')
        # drv_dir = os.path.join(preprocess_path, 'drv')
        # audio_dir = os.path.join(preprocess_path, 'audio')
        # processed_drv_path = os.path.join(preprocess_path, 'drv', f'drv.mp4')
        # processed_src_path = os.path.join(src_dir, 'src.png')
        # processed_audio_path = os.path.join(audio_dir, 'audio.wav')
        # output_file_path = self.config.TH.inference.output.save_path
        # output_dir = os.path.dirname(output_file_path)
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        # ## 1. Process Image / Audio
        # if not os.path.exists(src_dir):
        #     os.makedirs(src_dir)
        # if not os.path.exists(audio_dir):
        #     os.makedirs(audio_dir)
        
        # ## 1 - 0. Crop and Resize source image, driving video
        # ######################
        # # import pdb; pdb.set_trace()
        # self.landmark_model.preprocess_image(src_path, processed_src_path)
        # ######################
        # os.system(f"ffmpeg -y -i {audio_path} {processed_audio_path}")

        # ## 1 - 1. Extract landmarks
        # bfv2v.extract_landmark_from_img(processed_src_path, src_dir, self.he_estimator, self.landmark_model)

        # ## 3. Run BFv2v to warp and generate video
        # args_run = copy.copy(self.config.TH.inference.attr.bfv2v)
        # args_run.model_config = self.config.TH.common.checkpoints.bfv2v.config
        # args_run.source_image = processed_src_path
        # args_run.driving_video = processed_drv_path
        # args_run.driven_dir = tmp_dir
        # args_run.result_dir = tmp_dir
        # args_run.result_video = 'mute.mp4'
        # args_run.fps = self.config.TH.common.attr.fps
        # bfv2v.test_model(args_run, self.bfv2v, self.gpus)

        # ## 4. Post Process
        # # add audio
        # SIZE = 256
        # if output_name is not None:
        #     output_file_path = output_name
        # os.system(f"ffmpeg -y -i {os.path.join(args_run.result_dir, args_run.result_video)} -i {processed_audio_path} -map 0:v:0 -map 1:a:0 -filter:v 'fps={self.config.TH.common.attr.fps}' {output_file_path}")
        
            

    # def preprocess(self, **kargs):
    #     try:
    #         drv_path = kargs['base_video_path']

    #         ## 0. Construct Preprocess Dir
    #         preprocess_path = self.config.config.preprocess.output.dir
    #         drv_landmark_dir = os.path.join(preprocess_path, 'drv')
    #         processed_drv_path = os.path.join(drv_landmark_dir, 'drv.mp4')
            
    #         if not os.path.exists(drv_landmark_dir):
    #             os.makedirs(drv_landmark_dir)
            
    #         ## 1. Crop and Resize source image, driving video
    #         drv_fps30_path = os.path.join(drv_landmark_dir, '30_drv.mp4')
    #         os.system(f"ffmpeg -y -i {drv_path} -vf fps=30 {drv_fps30_path}")
    #         cmds = self.landmark_model.preprocess_video(drv_fps30_path, processed_drv_path)

    #         ## 2. Extract landmarks
    #         bfv2v.extract_landmark_from_video(processed_drv_path, drv_landmark_dir, self.he_estimator, self.landmark_model)
            
    #     except Exception as e:
    #         traceback.print_exc()
    #         return {
    #             'status': self.config.const.FAILED
    #         }
            
# class THPipeline(AbstractPipeline):

#     def __init__(self, config, gpus=None):
#         super(THPipeline, self).__init__(config, gpus)
#         assert type(gpus) == list or type(gpus) == int, 'gpus should be list or integer'
#         self.gpus = gpus if type(gpus) == list else [gpus]
#         self.print('TH module has been initialized.')
#         self.set_twin(twin_id='temporal')
        
#     def initialize(self):
#         ################################
#         torch.cuda.empty_cache()
#         ################################
#         self.bfv2v = bfv2v.load_fv2v2(self.config.TH.common.checkpoints.bfv2v.config, self.config.TH.common.checkpoints.bfv2v.model, self.gpus)
#         self.he_estimator = bfv2v.load_he_estimator(self.config.TH.common.checkpoints.he_estimator.config, self.config.TH.common.checkpoints.he_estimator.model, self.gpus)
#         self.landmark_model = bfv2v.load_landmark_model(self.config.TH.common.checkpoints.landmark_model.dir, self.gpus)

#     def preprocess(self, **kargs):
#         try:
#             drv_path = kargs['base_video_path']

#             ## 0. Construct Preprocess Dir
#             preprocess_path = self.config.TH.preprocess.output.save_path
#             drv_landmark_dir = os.path.join(preprocess_path, 'drv')
#             processed_drv_path = os.path.join(drv_landmark_dir, 'drv.mp4')
            
#             if not os.path.exists(drv_landmark_dir):
#                 os.makedirs(drv_landmark_dir)
            
#             ## 1. Crop and Resize source image, driving video
#             drv_fps30_path = os.path.join(drv_landmark_dir, '30_drv.mp4')
#             os.system(f"ffmpeg -y -i {drv_path} -vf fps=30 {drv_fps30_path}")
#             cmds = self.landmark_model.preprocess_video(drv_fps30_path, processed_drv_path)

#             ## 2. Extract landmarks
#             bfv2v.extract_landmark_from_video(processed_drv_path, drv_landmark_dir, self.he_estimator, self.landmark_model)
            
#             return {
#                 'status': self.config.const.SUCCESS
#             }
#         except Exception as e:
#             traceback.print_exc()
#             return {
#                 'status': self.config.const.FAILED
#             }

#     def train(self, **kargs):
#         pass

#     def finetune(self, **kargs):
#         pass

#     def inference(self, **kargs):

#         try:
#             src_path =  kargs['input_image_path']
#             audio_path =  kargs['input_audio_path']
#             is_high_resolution = kargs['is_high_resolution']
#             output_name = kargs['output_name']
#             ## 0. Setup Directories
#             tmp_dir = self.config.TH.preprocess.output.save_path
#             preprocess_path = self.config.TH.preprocess.output.save_path
#             src_dir = os.path.join(preprocess_path, 'src')
#             drv_dir = os.path.join(preprocess_path, 'drv')
#             audio_dir = os.path.join(preprocess_path, 'audio')
#             processed_drv_path = os.path.join(preprocess_path, 'drv', f'drv.mp4')
#             processed_src_path = os.path.join(src_dir, 'src.png')
#             processed_audio_path = os.path.join(audio_dir, 'audio.wav')
#             output_file_path = self.config.TH.inference.output.save_path
#             output_dir = os.path.dirname(output_file_path)
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)

#             ## 1. Process Image / Audio
#             if not os.path.exists(src_dir):
#                 os.makedirs(src_dir)
#             if not os.path.exists(audio_dir):
#                 os.makedirs(audio_dir)
            
#             ## 1 - 0. Crop and Resize source image, driving video
#             ######################
#             # import pdb; pdb.set_trace()
#             self.landmark_model.preprocess_image(src_path, processed_src_path)
#             ######################
#             os.system(f"ffmpeg -y -i {audio_path} {processed_audio_path}")

#             ## 1 - 1. Extract landmarks
#             bfv2v.extract_landmark_from_img(processed_src_path, src_dir, self.he_estimator, self.landmark_model)

#             ## 3. Run BFv2v to warp and generate video
#             args_run = copy.copy(self.config.TH.inference.attr.bfv2v)
#             args_run.model_config = self.config.TH.common.checkpoints.bfv2v.config
#             args_run.source_image = processed_src_path
#             args_run.driving_video = processed_drv_path
#             args_run.driven_dir = tmp_dir
#             args_run.result_dir = tmp_dir
#             args_run.result_video = 'mute.mp4'
#             args_run.fps = self.config.TH.common.attr.fps
#             bfv2v.test_model(args_run, self.bfv2v, self.gpus)

#             ## 4. Post Process
#             # add audio
#             SIZE = 256 if is_high_resolution else 128
#             if output_name is not None:
#                 output_file_path = output_name
#             os.system(f"ffmpeg -y -i {os.path.join(args_run.result_dir, args_run.result_video)} -i {processed_audio_path} -map 0:v:0 -map 1:a:0 -filter:v 'fps={self.config.TH.common.attr.fps}' {output_file_path}")
            
#             return {
#                 'status': self.config.const.SUCCESS,
#                 'output': output_file_path
#             }
#         except Exception as e:
#             traceback.print_exc()
#             return {
#                 'status': self.config.const.FAILED
#             }