# Causal depthwise conv1d in CUDA with a PyTorch interface

Features:
- Support fp32, fp16, bf16.
- Kernel size 2, 3, 4.

## How to use

```python
from causal_conv1d import causal_conv1d_fn
```

```python
def causal_conv1d_fn(x, weight, bias=None, activation=None):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
```

Equivalent to:
```python
import torch.nn.functional as F

F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)[..., :seqlen]
```

## Additional Prerequisites for AMD cards

For TheRock source builds, install the matching PyTorch runtime and development SDK from AMD's stable index:

```bash
uv pip install --group build --no-sources --index https://stable.repo.amd.com/rocm/whl-next/ "rocm[devel]"
rocm-sdk init
```

Set `ROCM_PATH` to the output of `rocm-sdk path --root`.
On Windows also set `CXX=amdclang-cl` and `HIP_DEVICE_LIB_PATH=%ROCM_PATH%\lib\llvm\amdgcn\bitcode`.
When building without a GPU, set `HIP_ARCHITECTURES` to the target GPU architecture, such as `gfx942`.

### Patching ROCm

If you are on ROCm 6.0, run the following steps to avoid errors during compilation. This is not required for ROCm 6.1 onwards.

1. Locate your ROCm installation directory. This is typically found at `/opt/rocm/`, but may vary depending on your installation.

2. Apply the Patch. Run with `sudo` in case you encounter permission issues.
   ```bash
    patch /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h < rocm_patch/rocm6_0.patch 
   ```
