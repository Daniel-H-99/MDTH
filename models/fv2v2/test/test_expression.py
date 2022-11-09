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
	test_samples = construct_test_samples(config, cwd=cwd)
	materials['cwd'] = cwd
	materials['config'] = config
	materials['pipeline'] = pipeline
	materials['test_samples'] = test_samples

	materials['logger'] = make_logger(cwd)
	materials = AttrDict.from_nested_dicts(materials)

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

def run_aggregation(materials):
	### concat videos
	cwd = materials.cwd
	output_video = os.path.join(materials.cwd, 'aggregation.mp4')
	concat_cmd = 'concat:' + '|'.join([os.path.join(cwd, session, 'video.mp4') for session in materials.session_names])
	cmd = f'ffmpeg -i {concat_cmd} -codec copy {output_video}'
	os.system(cmd)
	
def run_session(config, src, drv, pipeline, label):
	# src: path to raw image
	# drv: path to driving info 
    ### preprocess
	pipeline.preprocess_image(src, rewrite=config.dynamic.rewrite, preprocess=config.dynamic.preprocess, extract_landmarks=config.dynamic.extract_landmarks)

	while True:
		continue
	### inference
	session_dir = pipeline.inference_expression(src, drv, label)

	return session_dir

def run_exp(materials):
	materials.logger.info('running exp...')
	session_names = []
	for src, drv in tqdm(materials.test_samples):
		print(f'running session: {(src, drv)}')
		session_name = run_session(materials.config, src, drv, materials.pipeline, materials.label)
		session_names.append(session_name)
	materials.logger.info('finished exp')
	setattr(materials, 'session_names', session_names)

	### aggregate
	run_aggregation(materials, session_names)

	del materials.pipeline
	del materials.logger

	torch.save(materials, os.path.join(materials.cwd, 'materials.pt'))

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

	drv = inputs.driving[0]
	for source_line in inputs.source:
		src = source_line
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