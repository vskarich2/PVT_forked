import os
import sys

BASE_DIR = None

if sys.platform == 'win32':
    print("This is a Windows machine.")
    BASE_DIR = r"C:\Users\mberm\PVT_forked"
elif sys.platform == 'darwin':
    print("This is a macOS machine.")
    BASE_DIR = "/Users/vskarich/cs231n_final_project/PVT_forked_repo/PVT_forked"
else:
    print(f"This is a different operating system: {sys.platform}")

DATA_DIR = os.path.join(BASE_DIR, 'data', 'dev_modelnet40_normal_resampled')
NUM_POINTS_TEST = 1024