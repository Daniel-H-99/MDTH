from doctest import run_docstring_examples
from symbol import pass_stmt
import sys
import os
import argparse
import shutil

from requests import session
from pipelines.th import THPipeline
import yaml
import datetime
from metric import MetricEvaluater
import numpy as np

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def from_nested_dicts(self, data):
        if not isinstance(data, dict):
            return data
        else:
            return self({key: self.from_nested_dicts(data[key]) for key in data})


def replace_env(env, d):
    for k, v in d.items():
        if type(v) == str:
            for env_k, env_v in env.items():
                env_k = '@' + env_k
                if env_k in v:
                    d[k] = v.replace(env_k, env_v)
        elif type(v) == dict:
            replace_env(env, v)

def read_config(config_path):
	with open(config_path) as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
        
	env = config['env']
	replace_env(env, config)
	config = AttrDict.from_nested_dicts(config)

	return config

def setup_exp(args, config):
    materials = {}
    pipeline = THPipeline(config, config.dynamic.gpus)
    materials['label'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if config.dynamic.label is not None:
        materials['label'] = '_'.join([config.dynamic.label, materials['label']])
    cwd = os.path.join(config.env.res_path, materials['label'])
    os.makedirs(cwd)
    shutil.copy(args.config, os.path.join(cwd, 'config.yaml'))
    os.makedirs(os.path.join(cwd, 'inputs'))
    os.makedirs(os.path.join(cwd, 'metrics'))
    test_samples = construct_test_samples(config, cwd=cwd)
    materials['root_dir'] = cwd
    materials['config'] = config
    materials['pipeline'] = pipeline
    materials['test_samples'] = test_samples
    materials['metric'] = MetricEvaluater(config)
    materials = AttrDict.from_nested_dicts(materials)
	
    return materials

def eval_iter_sessions(func, materials, session_names, post_fix=''):
    scores = []
    session_dirs = list(map(lambda x: os.path.join(materials.cwd, x), session_names))

    for session_name, session_dir in zip(session_names, session_dirs):
        inputs = np.loadtxt(os.path.join(session_dir, 'inputs.txt'))
        source, driving = inputs
        result_path = os.path.join(session, post_fix) if len(post_fix) > 0 else session
        GT_path = os.path.join(driving, post_fix) if len(post_fix) > 0 else driving

        score_session = func(result_path, GT_path)
        scores[f'{session_name}'] = score_session
	
    return scores

def eval_exp(materials, session_names):
    metric = materials.metric
    metrics = {}
    ### same identity
    #L1 ↓ FID ↓ SSIM ↑ LPIPS ↓ MS-SSIM ↑ AKD ↓ PSNR ↓
    criterions = [metric.L1, metric.FID, metric.SSIM, metric.LPIPS, metric.MS-SSIM, metric.AKD, metric.PSNR]
    for criterion in criterions:
        score = eval_iter_sessions(criterion, materials, session_names, )
        metric[''] = score
    
    

def run_session(config, src, drv, pipeline, label):
    ### preprocess
	pipeline.preprocess_image(src, rewrite=config.dynamic.rewrite)
	pipeline.preprocess_video(drv, rewrite=config.dynamic.rewrite)
    
	### inference
	session_dir = pipeline.inference(src, drv, label, use_transformer=config.dynamic.use_transformer, extract_driving_code=config.dynamic.extract_driving_code, stage=config.dynamic.stage, relative_headpose=config.dynamic.relative_headpose, save_frames=config.dynamic.save_frames)
	return session_dir


def run_exp(materials):
	session_names = []
	for src, drv in materials.test_samples:
		print(f'running session: {(src, drv)}')
		session_name = run_session(materials.config, src, drv, materials.pipeline, materials.label)
		session_names.append(session_name)
	
	### evaluation
	eval_exp(materials, session_names)
 
	
	

def construct_test_samples(config, cwd=None):
	samples = []
	inputs = config.dynamic.inputs
	if config.dynamic.input_as_file:
		for input_key, input_file in list(vars(inputs).items()):
			items = []
			with open(input_file) as f:
				lines = f.readlines()
				for line in lines:
					items.append(line.strip())
			setattr(inputs, input_key, items)
			if cwd is not None:
				shutil.copy(input_file, os.path.join(cwd, 'inputs', input_key))
	else:
		inputs = config.dynamics.inputs
	if config.dynamic.mode == 'pair':
		for line in inputs.pair:
			print(f'line: {line}')
			src, drv = line.split(',')
			samples.append((src, drv))
	elif config.dynamic.mode == 'combination':
		for source_line in inputs.source:
			src = source_line
			for driving_line in inputs.driving:
				drv = driving_line
				samples.append((src, drv))
	return samples

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, required=True) 
	args, unparsed  = parser.parse_known_args()

	config_path = args.config

	config = read_config(config_path)
	print(f'running with config: {config}')
 
	materials = setup_exp(args, config)
 
	run_exp(materials)