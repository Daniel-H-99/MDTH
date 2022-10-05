import os, sys
import glob
import time
import traceback
from datetime import datetime

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

class PathUtil:
    def __init__(self, config, twin_id=None):
        self.config = config
        self.twin_id = twin_id
        self.dlim = '@'

    def replace_path(self, config, is_list=False):
        if is_list:
            list_replaced = []
            for v in config:
                if type(v) == AttrDict or type(v) == dict:
                    list_replaced.append(self.replace_path(v))
                elif type(v) == list:
                    list_replaced.append(self.replace_path(v, is_list=True))
                elif type(v) == str:
                    v = v.replace(f'{self.dlim}RAW_PATH{self.dlim}', str(self.config.env.raw_path))
                    v = v.replace(f'{self.dlim}COMMON_PATH{self.dlim}', str(self.config.env.common_path))
                    v = v.replace(f'{self.dlim}PROC_PATH{self.dlim}', str(self.config.env.proc_path))
                    v = v.replace(f'{self.dlim}SERV_PATH{self.dlim}', str(self.config.env.serv_path))
                    v = v.replace(f'{self.dlim}TEMP_PATH{self.dlim}', str(self.config.env.temp_path))
                    v = v.replace(f'{self.dlim}TWIN_ID{self.dlim}', str(self.twin_id))
                    list_replaced.append(v)
                else:
                    list_replaced.append(v)
            return list_replaced
        else:
            for k,v in config.items():
                if type(v) == AttrDict or type(v) == dict:
                    config[k] = self.replace_path(v)
                elif type(v) == list:
                    config[k] = self.replace_path(v, is_list=True)
                elif type(v) == str:
                    v = v.replace(f'{self.dlim}RAW_PATH{self.dlim}', str(self.config.env.raw_path))
                    v = v.replace(f'{self.dlim}COMMON_PATH{self.dlim}', str(self.config.env.common_path))
                    v = v.replace(f'{self.dlim}PROC_PATH{self.dlim}', str(self.config.env.proc_path))
                    v = v.replace(f'{self.dlim}SERV_PATH{self.dlim}', str(self.config.env.serv_path))
                    v = v.replace(f'{self.dlim}TEMP_PATH{self.dlim}', str(self.config.env.temp_path))
                    v = v.replace(f'{self.dlim}TWIN_ID{self.dlim}', str(self.twin_id))
                    config[k] = v
            return config


def now():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
