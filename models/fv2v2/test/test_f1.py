import torch
from test import read_config
from metric import MetricEvaluater

config = '/home/server19/minyeong_workspace/MDTH/models/fv2v2/test/config/config_homo_0.yaml'
config = read_config(config)

m = MetricEvaluater(config)

x_ours = '/home/server19/minyeong_workspace/MDTH/models/fv2v2/test/res/f_vox_eval_group_0_homo_2022-11-04T10:41:22/vox_eval%id10280#XiKRlssBw2M#000330#001148.mp4_vox_eval%id10280#XiKRlssBw2M#000330#001148.mp4/frames'
x_lia = '/home/server19/minyeong_workspace/LIA/test/res/vox_eval_group_0_homo_2022-10-31T11:46:14/vox_eval%id10280#XiKRlssBw2M#000330#001148.mp4_vox_eval%id10280#XiKRlssBw2M#000330#001148.mp4/frames'
y = '/mnt/hdd/minyeong_workspace/Experiment/proc/vox_eval%id10280#XiKRlssBw2M#000330#001148.mp4/frames'

aucon_ours = m.AUCON(x_ours, y)
vs = []
for v in list(aucon_ours.values()):
    if not torch.isnan(v):
        vs.append(v)
mean_ours = torch.stack(vs).mean()

aucon_lia = m.AUCON(x_lia, y)
vs = []
for v in list(aucon_lia.values()):
    if not torch.isnan(v):
        vs.append(v)
mean_lia = torch.stack(vs).mean()


print(f'aucon_ours: {aucon_ours}')
print(f'aucon_lia: {aucon_lia}')

print(f'mean_ours: {mean_ours}')
print(f'mean_lia: {mean_lia}')