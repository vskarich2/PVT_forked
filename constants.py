import os
import sys

# This is only for local development, there are other BASE_DIR definitions
# in the code and those are based on the location of the file that defines DATA_DIR,
# so we don't want to change those and screw things up. See data.py.
BASE_DIR = None

if sys.platform == 'win32':
    BASE_DIR = r"C:\Users\mberm\PVT_forked"
elif sys.platform == 'darwin':
    BASE_DIR = "/Users/vskarich/cs231n_final_project/PVT_forked_repo/PVT_forked"
else:
    BASE_DIR = "/content/PVT_forked"

# This is only for local development, there are other DATA_DIR definitions
# in the code and those are based on the location of the file that defines DATA_DIR,
# so we don't want to change those and screw things up. See data.py.
DATA_DIR = os.path.join(BASE_DIR, 'data', 'dev_modelnet40_normal_resampled')

NUM_POINTS_TEST = 1024


# These are flags that can change value
ANNOUNCE_WHICH_ATTENTION=True