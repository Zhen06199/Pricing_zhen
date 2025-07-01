import platform
import psutil

import sys
import scipy
import numpy
import matplotlib

print("Python版本:", sys.version)
print("NumPy版本:", numpy.__version__)
print("SciPy版本:", scipy.__version__)
print("Matplotlib版本:", matplotlib.__version__)


# 查看 CPU 信息
print("处理器型号:", platform.processor())
print("物理核心数:", psutil.cpu_count(logical=False))
print("逻辑核心数:", psutil.cpu_count(logical=True))

# 查看内存信息
mem = psutil.virtual_memory()
print(f"总内存: {mem.total / (1024 ** 3):.2f} GB")
