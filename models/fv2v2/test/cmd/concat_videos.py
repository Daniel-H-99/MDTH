import os
import shutil
import numpy as np

root_dir = 'demos'
baseline_root_dir = os.path.join(root_dir,'baseline')
baselines = ['MMRA','TPSMM', 'LIA', 'OURS', 'OURS_NOLOG']
output_root_dir = os.path.join(baselines_root_dir, 'output')
output_video_dir = os.path.join(output_root_dir, 'video')
label_file_path = os.path.join(output_root_dir, 'labels.txt')

def construct_video_list(vid_dir):
    res = []
    group_dirs = os.listdir(vid_dir)
    for g in group_dirs:
        print(f'working on group: {g}')
        g_dir = os.path.join(vid_dir, g)
        vids = os.listdir(g_dir)
        res.extend(vids)

    print(f'total videos in list: {len(res)}')
    return res

def construct_video_dict(vid_list):
    res = {}
    for baseline in baselines:
        print(f'working on baseline: {baseline}')
        res_baseline = {}
        baseline_dir = os.path.join(baseline_root_dir, baseline)
        groups = os.listdir(baseline_dir)
        print(f'found {len(groups)} groups')
        for g in groups:
            g_dir = os.path.join(baselin_dir, g)
            vids = os.listdir(g_dir)
            for vid in vids:
                if vid in vid_list:
                    vid_path = os.path.join(g_dir, vid)
                    res_baseline[vid] = vid_path
        res[baseline] = res_baseline

    cnt = -1
    for k, v in res.item():
        if cnt == -1:
            cnt = len(list(v.keys()))
        else:
            assert cnt == len(list(v.keys())), f'the number of videos in baselines are not matched at baseline - {k}'
    print(f'Each baselines have {cnt} videos')
    return res

def get_src_drv_path(src_dir, drv_dir):
    src_frame_path = os.path.join(src_dir, 'frames', '0000000.png')
    drv_frames_dir = os.path.join(drv_dir, 'frames')
    NotImplementedError

def make_labels(vid_list):
    N = len(vid_list)
    K = len(baselines)
    random_order = np.concatenate([np.random.permutation(baselines) for _ in range(N)], axis=0)
    return random_order

def mp4_to_gif(path):
    NotImplementedError


def make_concat_video_cmd(output_name, src_path, drv_path, result_path_list):
    cmd = f'ffmpeg'
    cwd = output_video_dir
	output_video = os.path.join(cwd, output_name)
    concat_src_list = [src_path, drv_path] + result_path_list
	concat_cmd = ' -i '.join(concat_src_list)
	cmd = f'ffmpeg -i {concat_cmd} -filter_complex hstack=inputs={len(concat_src_list)} -y "{output_video}"'
	print(f'cmd: {cmd}')
    return cmd

def main():



    
