import gradio as gr
import numpy as np
import os
import tempfile
import shutil
import sys
import yaml
from PIL import Image

class WrappedTH():
    def __init__(self, config):
        self.config = config
        pipeline_config = config['pipeline_config']
        gpus = list(map(lambda x: int(x), config['gpus'].split(',')))
        self.pipeline = THPipeline(pipeline_config, gpus)

        self.pipeline.initialize(task='inference')
        self.workspace = config['workspace']
        self.drv_list = config['driving_video_list']
        
    def setup_dir(self):
        id_dir = tempfile.TemporaryDirectory(dir=self.workspace).name
        id = os.path.basename(id_dir)
        processed_dir = os.path.join(id_dir, 'processed')
        result_dir = os.path.join(id_dir, 'result')
        os.makedirs(processed_dir)
        os.makedirs(result_dir)

        return {
            'id': id,
            'id_dir': id_dir,
            'processed_dir': processed_dir,
            'result_dir': result_dir
        }

    def run(self, img_path, audio_path, drv_idx):
        ## 0. setup
        dirs = self.setup_dir()
        self.pipeline.set_twin(twin_id=dirs['id'])
        selected_drv_path = self.drv_list[drv_idx]
        drv_path = os.path.join(dirs['processed_dir'], 'drv')
        shutil.copytree(selected_drv_path, drv_path)
        output_path = os.path.join(dirs['result_dir'], 'output.mp4')

        ## 1. run_pipeline
        self.pipeline.inference(input_image_path=img_path, input_audio_path=audio_path, is_high_resolution=1, output_name=output_path)

        return output_path

def build_examples(example_paths):
    examples = []
    for paths in example_paths:
        img, audio, vid = paths
        # img = gr.components.Image(value=img, type='filepath')
        # audio = gr.components.Audio(value=audio)
        # vid = gr.components.Number(value=vid)
        img = os.path.join(os.path.dirname(__file__), img)
        audio = os.path.join(os.path.dirname(__file__), audio)
        examples.append([img, audio, vid])
    return examples

if __name__=='__main__':
    sys.path.insert(0, 'models')
    from pipelines.th import THPipeline
    config_path = 'config.yml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    pipeline = WrappedTH(config)
    iface = gr.Interface(fn=lambda image, audio, motion: pipeline.run(image, audio, motion),
                        inputs=[gr.components.Image(type='filepath'), gr.components.Audio(type='filepath'), gr.components.Slider(minimum=0, maximum=len(pipeline.drv_list)-1, step=1)],
                        outputs=gr.components.Video(type='filepath'), 
                        examples=build_examples(config['examples']))
    iface.launch(share=True)