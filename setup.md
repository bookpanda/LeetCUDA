# setup
```bash
sudo apt update
sudo apt -y install clangd

uv init

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1

```

# compile
```bash
# You shouldn’t compile with nvcc directly. This kernel is built as a PyTorch extension via torch.utils.cpp_extension.load(), which adds the Torch, CUDA, and Python include paths.
python elementwise.py

```