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
import csv
import torch
import types
import glob
import time
from tqdm import tqdm
import logging



class MetricItem():
    def __init__(self, name, func, post_fix='', src_label=False):
        self.name = name
        self.func = func
        self.post_fix=post_fix
        self.src_label = src_label


METRIC_META = {
    'L1': MetricItem('L1', 'L1', 'frames'),
    'FID': MetricItem('FID', 'FID', 'frames'),
    'SSIM': MetricItem('SSIM', 'SSIM', 'frames'),
    'LPIPS': MetricItem('LIPIPS', 'LPIPS', 'frames'),
    'MS-SSIM': MetricItem('MS-SSIM', 'MS_SSIM', 'frames'),
    'AKD': MetricItem('AKD', 'AKD', 'frames'),
    'PSNR': MetricItem('PSNR', 'PSNR', 'frames'),
    'AED': MetricItem('AED', 'AED', 'frames', True),
    'AUCON': MetricItem('AUCON', 'AUCON', 'frames')
}

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
	if len(config.dynamic.load_exp) == 0:
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
		materials['cwd'] = cwd
		materials['config'] = config
		materials['pipeline'] = pipeline
		materials['test_samples'] = test_samples
		materials['metric'] = MetricEvaluater(config)
		materials['metric_names'] = {
		'same_identity': ['L1', 'FID', 'SSIM', 'LPIPS', 'MS-SSIM', 'AKD', 'PSNR', 'AED'],
		'cross_identity': ['AED', 'AUCON']
		}
		materials['logger'] = make_logger(cwd)
		materials = AttrDict.from_nested_dicts(materials)

	else:
		loaded_materials = torch.load(config.dynamic.load_exp)

		materials['config'] = config
		materials['logger'] = make_logger(loaded_materials.cwd)
		materials = AttrDict.from_nested_dicts(materials)
		
		setattr(loaded_materials, 'metric', MetricEvaluater(loaded_materials.config))
		setattr(loaded_materials, 'metric_names', AttrDict.from_nested_dicts({
		'same_identity': ['L1', 'FID', 'SSIM', 'LPIPS', 'MS-SSIM', 'AKD', 'PSNR'],
		'cross_identity': ['AED', 'AUCON']
		}))
  
		setattr(materials, 'loaded_materials', loaded_materials)
	
	return materials

def make_logger(cwd):
	# 로그 생성
	logger = logging.getLogger()

	# 로그의 출력 기준 설정
	logger.setLevel(logging.INFO)

	# log 출력 형식
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

	# log 출력
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)

	# log를 파일에 출력
	file_handler = logging.FileHandler(os.path.join(cwd, 'log.log'))
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
 
	return logger

def write_scores(scores, materials):
	output_score_file = os.path.join(materials.cwd, 'metrics', 'scores.csv')
	output_summary_file = os.path.join(materials.cwd, 'metrics', 'summary.txt')
	metric_names = list(scores.keys())
	fields = ['vid'] + metric_names
	vids = list(list(scores.values())[0].keys())
 
	### raw data
	with open(output_score_file, "w") as f:
		w = csv.DictWriter(f, fields)
		print(f'fileds: {fields}')
		w.writeheader()
		for vid in vids:	
			w.writerow({field: scores.get(field)[vid] if field in scores else vid for field in fields})
   
	### summary
	s = []
	for metric_name in metric_names:
		metric_scores = np.array(list(scores[metric_name].values()))
		mean, std = metric_scores.mean(), metric_scores.std()
		s.append(f'{metric_name}: {mean}-{std}')
	
	np.savetxt(output_summary_file, s, fmt='%s')
	# summary_string = '\n'.join(s)
	
	# with open(output_summary_file, 'w') as f:
	# 	f.writelines(summary_string)


def eval_iter_sessions(func, materials, session_names, post_fix='', src_label=False):
	scores = {}
	print(f'cwd: {materials.cwd}')
	print(f'session names: {session_names}')
	session_dirs = list(map(lambda x: os.path.join(materials.cwd, x), session_names))

	for session_name, session_dir in zip(session_names, session_dirs):
		inputs = np.loadtxt(os.path.join(session_dir, 'inputs.txt'), dtype=str, comments=None)
		print(f'inputs: {inputs}')
		source, driving = inputs
		# with open(os.path.join(session_dir, 'inputs.txt'), 'r') as f:
		# 	inputs = f.readlines()

		# 	print(f'inputs: {inputs}')
   
		label = source if src_label else driving
  
		result_path = os.path.join(session_dir, post_fix) if len(post_fix) > 0 else session_name
		GT_path = os.path.join(label, post_fix) if len(post_fix) > 0 else driving
  
		score_session = func(result_path, GT_path).detach().cpu().numpy()
		scores[session_name] = score_session
	
	return scores

def eval_exp(materials):
	metric = materials.metric
	session_names = materials.session_names
	scores = {}
 
	### same identity evaluation
	metrics = getattr(materials.metric_names, 'cross_identity' if materials.config.dynamic.relative_headpose else 'same_identity')
	# metrics = materials.metric_names.cross_identity
	for metric_name in tqdm(metrics):
		meta_info = METRIC_META[metric_name]
		metric_score = eval_iter_sessions(getattr(metric, meta_info.func), materials, session_names, post_fix=meta_info.post_fix, src_label=meta_info.src_label)
		scores[metric_name] = metric_score
  
	write_scores(scores, materials)
 
	return scores

def run_session(config, src, drv, pipeline, label):
    ### preprocess
	if src.endswith('.mp4'):
		pipeline.preprocess_video(src, rewrite=config.dynamic.rewrite, preprocess=config.dynamic.preprocess, extract_landmarks=config.dynamic.extract_landmarks)
	else:
		pipeline.preprocess_image(src, rewrite=config.dynamic.rewrite, preprocess=config.dynamic.preprocess, extract_landmarks=config.dynamic.extract_landmarks)
		
	pipeline.preprocess_video(drv, rewrite=config.dynamic.rewrite, preprocess=config.dynamic.preprocess, extract_landmarks=config.dynamic.extract_landmarks)
		
	### inference
	session_dir = pipeline.inference(src, drv, label, use_transformer=config.dynamic.use_transformer, extract_driving_code=config.dynamic.extract_driving_code, stage=config.dynamic.stage, relative_headpose=config.dynamic.relative_headpose, save_frames=config.dynamic.save_frames)
	return session_dir

def run_exp(materials):
	if len(materials.config.dynamic.load_exp) == 0:
		materials.logger.info('running exp...')
		session_names = []
		for src, drv in tqdm(materials.test_samples):
			print(f'running session: {(src, drv)}')
			session_name = run_session(materials.config, src, drv, materials.pipeline, materials.label)
			session_names.append(session_name)
		materials.logger.info('finished exp')
		setattr(materials, 'session_names', session_names)
  
		if not materials.config.dynamic.skip_eval:
			### evaluation
			materials.logger.info('running evaluation...')
			eval_exp(materials)
			materials.logger.info('finished evaluation')
		
		# setattr(materials, 'result', types.SimpleNamespace())
  
		# setattr(materials.result, 'session_names', session_names)

		del materials.metric
		del materials.pipeline
		del materials.logger
	
		torch.save(materials, os.path.join(materials.cwd, 'materials.pt'))
  
	else:
		materials.logger.info('loading exp...')
		session_names = materials.loaded_materials
		print(f'check: {materials.loaded_materials.metric}')
  
		### evaluation
		materials.logger.info('running evaluation...')
		eval_exp(materials.loaded_materials)
		materials.logger.info('finished evaluation')
	
		# setattr(materials, 'result', types.SimpleNamespace())
		# materials.result.session_names = session_names
		# setattr(materials.result, 'session_names', session_names)

		# del materials.metric
		# del materials.pipeline
		# del materials.logger
	
		# torch.save(materials, os.path.join(materials.cwd, 'materials.pt'))


def construct_test_samples(config, cwd=None):
	samples = []
	inputs = config.dynamic.inputs
	data_dir = config.env.proc_path if not config.dynamic.preprocess else config.env.raw_path

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
			print(f'src, drv: {src} , {drv}')
			if '*' in src:
				print(f'globing src...')
				print(f'data dir: {data_dir}')
				print(f'globing: {os.path.join(data_dir, src)}')
    
				src = glob.glob(os.path.join(data_dir, src))
    
				src = list(map(lambda x: os.path.relpath(x, start=data_dir), src))
				print(f'globed src: {src}')
			if '*' in drv:
				drv = glob.glob(os.path.join(data_dir, drv))
				drv = list(map(lambda x: os.path.relpath(x, start=data_dir), drv))
			# assert (type(src) == list) * (type(drv) == list), 'src, drv pairs are not matched!'/
			if type(src) == list:
				for src_item, drv_item in zip(src, drv):
					samples.append((src_item, drv_item))
			else:
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
	
	materials.logger.info('exp setup finished')
 
	run_exp(materials)