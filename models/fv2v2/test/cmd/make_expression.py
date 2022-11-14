import numpy as np
import os

NUM_HEADS = 128
START = -1
END = 1
SLICES = 100
result_dir = '/home/server19/minyeong_workspace/MDTH/models/fv2v2/test/expressions'

def write_contents(h, start, end, slices):
    lines = []
    lines.append(f'num_heads,{NUM_HEADS}')
    lines.append(f'head,{h}')
    lines.append(f'start,{start}')
    lines.append(f'end,{end}')
    lines.append(f'slices,{slices}')
    return lines

for h in range(NUM_HEADS):
    fname = f'{h}_{START}_{END}_{SLICES}'
    lines = write_contents(h, START, END, SLICES)
    np.savetxt(os.path.join(result_dir, fname), lines, fmt='%s')
