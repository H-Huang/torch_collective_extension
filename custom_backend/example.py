import os

import torch
import dummy_collectives

import torch.distributed as dist

from torch._subclasses.fake_tensor import FakeTensorMode
from transformers import AutoModelForCausalLM, LlamaConfig
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

dist.init_process_group("cpu:dummy", rank=0, world_size=1)

fake_mode = FakeTensorMode()

model_config = LlamaConfig()

bs = 4
seq_len = 512

with fake_mode:
  model = AutoModelForCausalLM.from_config(model_config)
  ddp_model = DDP(model)
  x = torch.randint(model_config.vocab_size, size=(bs, seq_len), dtype=torch.long)
  loss = ddp_model(x).logits.sum()
  loss.backward()
