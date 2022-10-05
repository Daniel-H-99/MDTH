from doctest import run_docstring_examples
from symbol import pass_stmt
import sys
import os
import argparse
import shutil
from pipelines.th import THPipeline
import yaml
import datetime

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

def setup_exp(config):
	materials = {}
	pipeline = THPipeline(config, config.dynamic.gpus)
	materials['label'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
	if config.dynamic.label is not None:
		materials['label'] = '_'.join([config.dynamic.label, materials['label']])
	cwd = os.path.join(config.env.res_path, materials['label'])
	os.makedirs(cwd)
	os.makedirs(os.path.join(cwd, 'inputs'))
	test_samples = construct_test_samples(config, cwd=cwd)
	materials['config'] = config
	materials['pipeline'] = pipeline
	materials['test_samples'] = test_samples
	materials = AttrDict.from_nested_dicts(materials)
	
	return materials

def run_session(config, src, drv, pipeline, label):
    ### preprocess
	pipeline.preprocess_image(src)
	pipeline.preprocess_video(drv)
    
	### inference
	pipeline.inference(src, drv, label)
 
def run_exp(materials):
	for src, drv in materials.test_samples:
		print(f'running session: {(src, drv)}')
		run_session(materials.config, src, drv, materials.pipeline, materials.label)
  

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
 
	materials = setup_exp(config)
 
	run_exp(materials)