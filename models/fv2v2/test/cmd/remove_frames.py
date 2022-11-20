import shutil
import os

# dirs = ['/home/server19/minyeong_workspace/LIA/test/res/vox_eval_group_0_hetero_2022-10-31T08:33:24', '/home/server19/minyeong_workspace/LIA/test/res/vox_eval_group_0_homo_2022-10-31T11:46:14', '/home/server19/minyeong_workspace/LIA/test/res/vox_eval_group_1_hetero_2022-10-31T08:43:04', '/home/server19/minyeong_workspace/LIA/test/res/vox_eval_group_1_homo_2022-10-31T11:55:32', '/home/server19/minyeong_workspace/LIA/test/res/vox_eval_group_2_hetero_2022-10-31T08:50:33', '/home/server19/minyeong_workspace/LIA/test/res/vox_eval_group_2_homo_2022-10-31T12:02:44', '/home/server19/minyeong_workspace/LIA/test/res/vox_eval_group_3_hetero_2022-10-31T09:02:24', '/home/server19/minyeong_workspace/LIA/test/res/vox_eval_group_3_homo_2022-10-31T12:14:15', '/home/server19/minyeong_workspace/LIA/test/res/vox_eval_group_3_homo_2022-10-31T12:14:15', '/home/server19/minyeong_workspace/LIA/test/res/vox_eval_group_3_homo_2022-10-31T12:14:15', '/home/server19/minyeong_workspace/LIA/test/res/vox_eval_group_3_homo_2022-10-31T12:14:15', '/home/server19/minyeong_workspace/LIA/test/res/vox_eval_group_3_homo_2022-10-31T12:14:15', '/home/server19/minyeong_workspace/LIA/test/res/vox_eval_group_3_homo_2022-10-31T12:14:15', '/home/server19/minyeong_workspace/LIA/test/res/vox_eval_group_4_hetero_2022-10-31T09:11:31', '/home/server19/minyeong_workspace/LIA/test/res/vox_eval_group_4_homo_2022-10-31T12:22:36']

# tgt_dir = '/home/server19/minyeong_workspace/LIA/test/res/'
tgt_dir = '/home/server19/minyeong_workspace/MDTH/models/fv2v2/test/res/'
# tgt_dir = '/home/server19/minyeong_workspace/TPSMM/test/res'
# tgt_dir = '/home/server19/minyeong_workspace/MRAA/test/res/'
# tgt_dir = '/home/server19/minyeong_workspace/FOM/test/res/'


dirs = []
for dir in os.listdir(tgt_dir):
    if dir.startswith('v4'):
        dirs.append(os.path.join(tgt_dir, dir))
     
for dir in dirs:
    shutil.copytree(dir, os.path.join(tgt_dir, 'noframe_' + os.path.basename(dir)), ignore=shutil.ignore_patterns('*frames*'), dirs_exist_ok=True)
    print(f'copied {dir}')