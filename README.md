# torch collective extension

A minimum demo for PyTorch distributed extension functionality for collectives.

This repository contains miminal implementation of two different workflows for extending `torch.distributed` collectives in C++.

1. Custom Backend Implementation (`custom_backend`) (new workflow, recommended)

2. Custom Process Group Implementation (`custom_process_group`) (old workflow, deprecated)

See the READMEs in each folder for more details.

### FAQ

Why are there two different workflows?

- With the introduction of dispatchable collectives in PyTorch 2.0, pytorch distributed collectives allow routing to different backends based on the device type of the tensor arguments. `custom_process_group` was the old extension method and 
`custom_backend` is the new extension method.

What are the differences?

- `custom_backend` is the more flexible alternative as it allows users to route to respective backend based on device type. For example `init_process_group(backend="cpu:gloo,cuda:dummy", ...)` will dispatch collectives with cpu tensor arguments to gloo and 
cuda tensor arguments to dummy. On the other hand, `custom_process_group` is more limited as it only allows users to route to a single backend.

Which one should I use?

- We recommend using the `custom_backend` implementation.