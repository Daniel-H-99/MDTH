import yaml
import torch
import boto3
import importlib
import inspect

from abc import ABC, abstractmethod
from common.misc import *


class AbstractPipeline(ABC):
    def __init__(self, config, gpus=None):
        with open(config, 'r') as file:
            self._config = AttrDict.from_nested_dicts(yaml.safe_load(file))
        self.twin_id = None
        self.is_initialized = {
            'preprocess': False,'train': False,
            'finetune': False,'inference': False,
        }
        gpus = gpus.split(',') if type(gpus) == str else gpus
        if type(gpus)==list:
            self.devices = [torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu') for gpu_id in gpus]
        else:
            self.device = torch.device(f'cuda:{gpus}' if gpus is not None else 'cpu')

    @abstractmethod
    def initialize(self, task):
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self, **kargs):
        raise NotImplementedError()

    @abstractmethod
    def train(self, **kargs):
        raise NotImplementedError()

    @abstractmethod
    def finetune(self, **kargs):
        raise NotImplementedError()

    @abstractmethod
    def inference(self, **kargs):
        raise NotImplementedError()

    def set_twin(self, twin_id):
        self.twin_id = twin_id
        self.config = AttrDict.from_nested_dicts(self._config.copy())
        pu = PathUtil(self.config, self.twin_id)
        self.config = pu.replace_path(self.config)

    def print(self, mesg):
        print(f'[{now()}] {mesg}')

