import os
import shutil
import numpy as np

flist = [f'vox_eval_group_{i}.txt' for i in range(5)]
print(f'working on flist: {flist}')

save_dir = 'first_frames'
source_dir = '/mnt/hdd/minyeong_workspace/Experiment/proc'
inputs_dir = '/home/server19/minyeong_workspace/MDTH/models/fv2v2/test/inputs'


def new_name(src_dir, cnt):
    print(f'[new_name]: called with src_dir - {src_dir}, cnt - {cnt}')
    res = '{:02d}_'.format(cnt) + 'id' + os.path.basename(src_dir).split('id', 2)[1]
    res = res.replace('.mp4', '.png')
    return res

def get_src_path(src_dir):
    return os.path.join(src_dir, 'frames', '0000000.png')

def copy_first_frame_from_dir(src_dir, cnt):
    src_path = get_src_path(src_dir)
    tgt_name =  new_name(src_dir, cnt)
    tgt_dir = save_dir
    tgt_path = os.path.join(tgt_dir, tgt_name)
    print(f'working with:')
    print(f'src_path: {src_path}')
    print(f'tgt_name: {tgt_name}')
    shutil.copy(src_path, tgt_path)

idx = 0

shutil.rmtree(save_dir)
os.makedirs(save_dir)

for f in flist:
    f = os.path.join(inputs_dir, f)
    input_videos = np.loadtxt(f, dtype=str, comments=None)
    for input_video in input_videos:
        src_dir = os.path.join(source_dir, input_video)
        copy_first_frame_from_dir(src_dir, idx)
        idx += 1

print(f'work finished with idx: {idx}')    

