# easy cuda checker

since: 2026-01-12

A minimal, single-file CUDA environment checker.
It verifies NVIDIA driver visibility, CUDA availability, and PyTorch runtime status
with clear, colored output.

---

## How to run

```bash
uv run main.py
```
Warning: This tool may trigger a download of several GB of files to install PyTorch(with uv pacage manager).
If you don't need checking for PyTorch, comment out the PyTorch dependency in pyproject.toml.

## output example
an output from a poor man's machine may look like this.
```
~~~$uv run main.py
CUDA Environment Check
----------------------------------------
    OS      : Linux
    Python  : 3.9.19
    CPU     : Intel Core i7-3930K @ 3.20GHz
    Cores   : 12 (logical)
    Memory  : 64.00 GB
✔ nvidia-smi found (NVIDIA driver OK)
    GPU info from nvidia-smi
    GPU count (smi): 3
        [GPU 0]
            Name        : NVIDIA GeForce RTX 3080
            Memory (GB) : 10.00
        [GPU 1]
            Name        : NVIDIA GeForce GTX 1070 Ti
            Memory (GB) : 8.00
        [GPU 2]
            Name        : NVIDIA GeForce GTX 1060
            Memory (GB) : 6.00
! nvcc not found (Toolkit may be missing)
✔ PyTorch 2.1.2 found
✔ torch.cuda.is_available() == True
    CUDA version (torch): 11.8
    GPU count (torch): 3
        [GPU 0]
            Name        : NVIDIA GeForce RTX 3080
            Capability  : 8.6
            Memory (GB) : 10.00
        [GPU 1]
            Name        : NVIDIA GeForce GTX 1070 Ti
            Capability  : 6.1
            Memory (GB) : 8.00
        [GPU 2]
            Name        : NVIDIA GeForce GTX 1060 6GB
            Capability  : 6.1
            Memory (GB) : 6.00
✔ CUDA runtime test (tensor matmul) SUCCESS
----------------------------------------
```