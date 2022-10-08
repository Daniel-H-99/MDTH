import os, sys
import glob
import time
import traceback
from datetime import datetime
from scipy import stats

def kstest_uniform(X):
    # X : 1-D numpy
    return stats.kstest(X, stats.uniform.cdf)

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

class Silent:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

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
                    config[k] = v
            return config

    def replace_twin(self, config, is_list=False):
        if is_list:
            list_replaced = []
            for v in config:
                if type(v) == AttrDict or type(v) == dict:
                    list_replaced.append(self.replace_twin(v))
                elif type(v) == list:
                    list_replaced.append(self.replace_twin(v, is_list=True))
                elif type(v) == str:
                    v = v.replace(f'{self.dlim}TWIN_ID{self.dlim}', str(self.twin_id))
                    list_replaced.append(v)
                else:
                    list_replaced.append(v)
            return list_replaced
        else:
            for k,v in config.items():
                if type(v) == AttrDict or type(v) == dict:
                    config[k] = self.replace_twin(v)
                elif type(v) == list:
                    config[k] = self.replace_twin(v, is_list=True)
                elif type(v) == str:
                    v = v.replace(f'{self.dlim}TWIN_ID{self.dlim}', str(self.twin_id))
                    config[k] = v
            return config

def now():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

async def get_twin_by_id(db, twin_id):
    # try:
    #     db.reconnect()
    #     mycursor = db.cursor()
    #     mycursor.execute(f"SELECT * FROM twins WHERE twin_id={twin_id}")
    #     myresult = mycursor.fetchall()
    #     ret_val = {
    #         'id':myresult[0][0],
    #         'name':myresult[0][1]}
    # except Exception as e:
    #     # traceback.print_exc()
    #     print('returning dummy information ... ')
    #     ret_val = {
    #         'id' : -1,
    #         'name' : 'dummy'}
    # finally:
    #     mycursor.close()
    try:
        info = await request_twin_info(twin_id)
        ret_val = {
            'id': info['id']
        }
    except Exception as e:
        print('returning dummy information ... ')
        ret_val = {'id' : '0181dba6-fcb6-7199-8e81-12560a39ab25'}
    return ret_val

async def request_twin_info(twin_id):
    now = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    request_url = f'https://api.studio.warping.aitricsdev.com/internal/v1/twins/{twin_id}'
    async with httpx.AsyncClient() as client:
        tasks = [request(client,request_url)]
        resp = await asyncio.gather(*tasks)
    return json.loads(resp[0])

def download_from_storage(s3, bucket_name, object_name, file_path):
    try:
        st = time.time()
        s3.download_file(bucket_name, object_name, file_path)
        print(f'download from cloud storage: done ({time.time()-st} s)')
    except Exception as e:
        traceback.print_exc()

def download_dir_from_storage(s3, bucket_name, remote_dir, local_file_path):
    try:
        st = time.time()
        os.makedirs(local_file_path, exist_ok=True)
        for obj in s3.list_objects(Bucket=bucket_name, Prefix=remote_dir)['Contents']:
            file_name=os.path.split(obj['Key'])[1]
            s3.download_file(bucket_name, obj['Key'], os.path.join(local_file_path, file_name)) # save to same path
        print(f'download from cloud storage: done ({time.time()-st} s)')
    except:
        traceback.print_exc()
        
def upload_to_storage(s3, bucket_name, bucket_path, file, is_obj=True):
    try:
        st = time.time()
        if is_obj:
            s3.upload_fileobj(file, bucket_name, bucket_path)
        else:
            s3.upload_file(file, bucket_name, os.path.join(bucket_path, os.path.split(file)[-1]))
        print(f'uploaded to cloud storage: done ({time.time()-st} s)')
    except Exception as e:
        traceback.print_exc()

def upload_dir_to_storage(s3, bucket_name, bucket_path, local_file_path):
    try:
        st = time.time()
        for paths in os.walk(local_file_path):
            for path in paths[2]:
                file_path = os.path.join(paths[0], path)
                dest_path = os.path.join(bucket_path, *file_path.replace(local_file_path, '').split('/'))
                print(f'{file_path} to {dest_path}')
                s3.upload_file(file_path, bucket_name, dest_path)
        print(f'download from cloud storage: done ({time.time()-st} s)')
    except:
        traceback.print_exc()
    
    
