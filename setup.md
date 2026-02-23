# setup
```bash
sudo apt update
sudo apt -y install clangd ninja-build

uv init

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1

```

# compile
```bash
# You shouldn’t compile with nvcc directly. This kernel is built as a PyTorch extension via torch.utils.cpp_extension.load(), which adds the Torch, CUDA, and Python include paths.
TORCH_CUDA_ARCH_LIST=8.6 python elementwise.py
```

## compile speedup
First run is slow (1–5 min). To speed up:

1. **ninja** – parallel build (install with `apt install ninja-build`)
2. **TORCH_CUDA_ARCH_LIST** – compile only for your GPU (e.g. RTX 30xx = 8.6):
   ```bash
   TORCH_CUDA_ARCH_LIST=8.6 python elementwise.py
   ```
   Check your arch: `python -c "import torch; print(torch.cuda.get_device_capability())"` → (8, 6) = 8.6