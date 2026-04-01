#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
import platform


# no warning while this execution (some package may print noisy warning while checking without this...)
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


def safe_check_output(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()


def collect_system_info():
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

    return cpu_name, logical_cores, mem_gb


def query_rocm_smi():
    commands = [
        [
            "rocm-smi",
            "--showproductname",
            "--showmeminfo",
            "vram",
            "--csv",
        ],
        [
            "/opt/rocm/bin/rocm-smi",
            "--showproductname",
            "--showmeminfo",
            "vram",
            "--csv",
        ],
    ]

    for cmd in commands:
        executable = cmd[0]
        if os.path.isabs(executable):
            if not os.path.exists(executable):
                continue
        elif not exists(executable):
            continue

        try:
            return safe_check_output(cmd), executable
        except Exception:
            continue

    return None, None


def show_rocm_smi_summary(raw_output):
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if not lines:
        warn("rocm-smi returned no GPU details")
        return

    info("rocm-smi summary:")
    for line in lines[:12]:
        info(f"  {line}")
    if len(lines) > 12:
        info("  ...")


def query_rocminfo():
    commands = [
        ["rocminfo"],
        ["/opt/rocm/bin/rocminfo"],
    ]

    for cmd in commands:
        executable = cmd[0]
        if os.path.isabs(executable):
            if not os.path.exists(executable):
                continue
        elif not exists(executable):
            continue

        try:
            return safe_check_output(cmd), executable
        except Exception:
            continue

    return None, None


# ================== Start ==================
print("ROCm Environment Check")
print("-" * 40)

# 1. System
info(f"OS      : {platform.system()}")
info(f"Python  : {platform.python_version()}")

cpu_name, logical_cores, mem_gb = collect_system_info()
info(f"CPU     : {cpu_name}")
info(f"Cores   : {logical_cores} (logical)")
info(f"Memory  : {mem_gb}")

# 2. ROCm tools
rocm_smi_output, rocm_smi_cmd = query_rocm_smi()
if rocm_smi_output:
    ok(f"{rocm_smi_cmd} found (ROCm driver/tooling OK)")
    show_rocm_smi_summary(rocm_smi_output)
else:
    warn("rocm-smi not found or failed")

rocminfo_output, rocminfo_cmd = query_rocminfo()
if rocminfo_output:
    ok(f"{rocminfo_cmd} found (ROCm runtime available)")
    agents = sum(1 for line in rocminfo_output.splitlines() if "Agent" in line)
    info(f"rocminfo >> matched agent lines: {agents}")
else:
    ng("rocminfo NOT found")
    sys.exit(0)

# 3. PyTorch
try:
    import torch
    ok(f"PyTorch {torch.__version__} found")
except Exception:
    ng("PyTorch not available")
    sys.exit(0)

# 4. ROCm build check
hip_version = getattr(torch.version, "hip", None)
if hip_version:
    ok("PyTorch ROCm build detected")
    info(f"PyTorch HIP version: {hip_version}")
else:
    ng("PyTorch is not built with ROCm/HIP support")
    sys.exit(0)

# 5. Device availability
if torch.cuda.is_available():
    ok("torch.cuda.is_available() == True")
else:
    ng("torch.cuda.is_available() == False")
    sys.exit(0)

device_count = torch.cuda.device_count()
info(f"PyTorch >> GPU count: {device_count}")

for i in range(device_count):
    p = torch.cuda.get_device_properties(i)
    info(f"  [GPU {i}]")
    info(f"    Name        : {p.name}")
    info(f"    Memory (GB) : {p.total_memory / 1024**3:.2f}")
    gcn_arch = getattr(p, "gcnArchName", None)
    if gcn_arch:
        info(f"    GCN Arch    : {gcn_arch}")

# 6. Runtime Test
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
