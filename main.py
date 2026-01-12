#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
import platform


# no warning while this execution (some package may print noisy warning while cheking without this...)
import warnings
warnings.filterwarnings("ignore")


# ================== Color ==================
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

def ok(msg): print(f"{GREEN}✔ {msg}{RESET}")
def ng(msg): print(f"{RED}✘ {msg}{RESET}")
def info(msg): print(f"{BLUE}    {msg}{RESET}")
def warn(msg): print(f"{YELLOW}! {msg}{RESET}")

# ================== Helpers ==================
def exists(cmd):
    return shutil.which(cmd) is not None

# ================== Start ==================
print(f"CUDA Environment Check")
print("-" * 40)

# 1. System
info(f"OS      : {platform.system()}")
info(f"Python  : {platform.python_version()}")

# ---- CPU info ----
cpu_name = "unknown"
try:
    if platform.system() == "Linux":
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    elif platform.system() == "Darwin":
        cpu_name = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True
        ).strip()
    elif platform.system() == "Windows":
        cpu_name = platform.processor()
except Exception:
    pass

logical_cores = os.cpu_count()

info(f"CPU     : {cpu_name}")
info(f"Cores   : {logical_cores} (logical)")

# ---- Memory info ----
mem_gb = "unknown"
try:
    if platform.system() == "Linux":
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    mem_gb = f"{kb / 1024 / 1024:.2f} GB"
                    break
    elif platform.system() == "Darwin":
        bytes_ = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True
        ).strip())
        mem_gb = f"{bytes_ / 1024**3:.2f} GB"
    elif platform.system() == "Windows":
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        mem_gb = f"{stat.ullTotalPhys / 1024**3:.2f} GB"
except Exception:
    pass

info(f"Memory  : {mem_gb}")

# 2. NVIDIA smi
if exists("nvidia-smi"):
    ok("nvidia-smi found (NVIDIA driver OK)")

    try:
        smi_gpus = []
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()

        for i, line in enumerate(out.splitlines()):
            name, mem = [x.strip() for x in line.split(",")]
            smi_gpus.append((name, float(mem) / 1024))

        info(f"nvidia-smi >> GPU count: {len(smi_gpus)}")
        for i, (name, mem_gb) in enumerate(smi_gpus):
            info(f"  [GPU {i}]")
            info(f"    Name        : {name}")
            info(f"    Memory (GB) : {mem_gb:.2f}")

    except Exception as e:
        warn("Failed to query GPU info from nvidia-smi")
        warn(str(e))
else:
    ng("nvidia-smi NOT found")
    sys.exit(0)


# 3. CUDA Toolkit
if exists("nvcc"):
    ok("nvcc found (CUDA Toolkit installed)")
else:
    warn("nvcc not found (Toolkit may be missing)")

# 4. PyTorch
try:
    import torch
    ok(f"PyTorch {torch.__version__} found")
except Exception:
    ng("PyTorch not available")
    sys.exit(0)

# 5. CUDA availability
if torch.cuda.is_available():
    ok("torch.cuda.is_available() == True")
else:
    ng("torch.cuda.is_available() == False")
    sys.exit(0)

info(f"Pytorch CUDA version: {torch.version.cuda}")

# 6. GPU Devices
device_count = torch.cuda.device_count()
info(f"PyTorch >> GPU count: {device_count}")

for i in range(device_count):
    p = torch.cuda.get_device_properties(i)
    info(f"  [GPU {i}]")
    info(f"    Name        : {p.name}")
    info(f"    Capability  : {p.major}.{p.minor}")
    info(f"    Memory (GB) : {p.total_memory / 1024**3:.2f}")

# 7. Runtime Test
try:
    x = torch.randn(512, 512, device="cuda")
    y = torch.randn(512, 512, device="cuda")
    z = x @ y
    torch.cuda.synchronize()
    ok("torch device=\"cuda\" matmul test SUCCESS")
except Exception as e:
    ng("torch device=\"cuda\" matmul test FAILED")
    warn(str(e))

print("-" * 40)
