import os

import torch
import dummy_collectives

import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

dist.init_process_group("dummy", rank=0, world_size=1)

x = torch.ones(6)
dist.all_reduce(x)
print(f"cpu allreduce: {x}")

if torch.cuda.is_available():
    y = x.cuda()
    dist.all_reduce(y)
    print(f"cuda allreduce: {y}")

try:
    dist.broadcast(x, 0)
except RuntimeError:
    print("got RuntimeError when calling broadcast")
