import os
import shutil
import numpy as np
from tqdm import tqdm

root_dir = 'demos'
baseline_root_dir = os.path.join(root_dir,'baseline_k-actors')
# baselines = ['MMRA','TPSMM', 'LIA', 'OURS', 'OURS_NOLOG']
baselines = ['FT', '3.2.1']
output_root_dir = os.path.join(root_dir, 'output_k-actors_ours')
output_video_dir = os.path.join(output_root_dir, 'video')
label_file_path = os.path.join(output_root_dir, 'labels.txt')
ROOT = '/home/server19/minyeong_workspace/MDTH/models/fv2v2/test'
os.makedirs(output_root_dir, exist_ok=True)
os.makedirs(output_video_dir, exist_ok=True)

def construct_video_list(baseline_dir):
    res = []
    src_dict = {}
    drv_dict = {}
    group_dirs = os.listdir(baseline_dir)
    for g in group_dirs:
        print(f'working on group: {g}')
        g_dir = os.path.join(baseline_dir, g)
        vids = os.listdir(g_dir)
        vids = list(filter(lambda x: '.mp4' in x, vids))
        for vid in vids:
            vid_dir = os.path.join(g_dir, vid)
            inputs= np.loadtxt(os.path.join(vid_dir, 'inputs.txt'), dtype=str, comments=None)
            src_path, drv_path = inputs
            src_dict[vid] = src_path
            drv_dict[vid] = drv_path
        res.extend(vids)

    print(f'total videos in list: {len(res)}')
    return res, src_dict, drv_dict

def construct_video_dict(vid_list):
    res = {}
    for baseline in baselines:
        print(f'working on baseline: {baseline}')
        res_baseline = {}
        baseline_dir = os.path.join(baseline_root_dir, baseline)
        groups = os.listdir(baseline_dir)
        print(f'found {len(groups)} groups')
        for g in groups:
            g_dir = os.path.join(baseline_dir, g)
            vids = os.listdir(g_dir)
            vids = list(filter(lambda x: '.mp4' in x, vids))
            for vid in vids:
                if vid in vid_list:
                    vid_path = os.path.join(g_dir, vid, 'mute.mp4')
                    res_baseline[vid] = vid_path
        res[baseline] = res_baseline

    cnt = -1
    for k, v in res.items():
        if cnt == -1:
            cnt = len(list(v.keys()))
        else:
            assert cnt == len(list(v.keys())), f'the number of videos in baselines are not matched at baseline - {k}'
    print(f'Each baselines have {cnt} videos')
    return res

def get_src_drv_path(src_dir, drv_dir):
    if 'frames' in src_dir:
        src_frame_path = os.path.join(src_dir, 'frames', '0000000.png')
    else:
        src_frame_path = os.path.join(src_dir, 'image.png')
    drv_frames_dir = os.path.join(drv_dir, 'frames')
    video_path = os.path.join(drv_dir, 'video.mp4')
    if not os.path.exists(video_path):
        cmd = [f'cd {drv_frames_dir}']
        cmd.append(f'ffmpeg -y -framerate 25 -i %07d.png -c:v libx264 -r 25 {video_path}')
        cmd.append(f'cd {ROOT}')
        os.system(';'.join(cmd))
        print(f'cmd: {cmd}')
    return src_frame_path, video_path


def make_labels(vid_list):
    N = len(vid_list)
    K = len(baselines)
    random_order = np.stack([np.random.permutation(baselines) for _ in range(N)], axis=0)
    return random_order

def mp4_to_gif(path):
    basename = os.path.basename(path)
    output_basename = basename[:-4] + '.gif'
    output_path = os.path.join(os.path.dirname(path), output_basename)
    cmd = f'ffmpeg -y -i {path} -r 25 {output_path}'
    os.system(cmd)
    return cmd

def make_concat_video_cmd(output_name, src_path, drv_path, result_path_list):
    output_video = output_name
    concat_src_list = [src_path, drv_path] + result_path_list
    concat_src_list = [f'"{v}"' for v in concat_src_list]
    concat_cmd = ' -i '.join(concat_src_list)
    cmd = f'ffmpeg -y -i {concat_cmd} -filter_complex hstack=inputs={len(concat_src_list)} -y "{output_video}"'
    print(f'cmd: {cmd}')
    return cmd

def main():
    print(f'started')
    ours_dir = os.path.join(baseline_root_dir, 'FT')
    video_list, src_dict, drv_dict = construct_video_list(ours_dir)
    video_dict = construct_video_dict(video_list)
    print(f'video_list_sample: {video_list[:5]}')
    labels = make_labels(video_list)
    print(f'label samples: {labels[:5]}')
    np.savetxt(label_file_path, labels, fmt="%s", comments=None)
    for vid, label in tqdm(zip(video_list, labels)):
        print(f'woring on vid - label: {vid} - {label}')
        src_dir = src_dict[vid]
        drv_dir = drv_dict[vid]
        src_path, drv_path = get_src_drv_path(src_dir, drv_dir)
        results = [video_dict[baseline][vid] for baseline in label]
        output_name = os.path.join(output_video_dir, vid + '.mp4')
        concat_cmd = make_concat_video_cmd(output_name, src_path, drv_path, results)
        os.system(concat_cmd)
        gif_cmd = mp4_to_gif(output_name)
    print(f'Done')

if __name__=='__main__':
    main()

    
