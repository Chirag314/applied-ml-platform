import os, random
import numpy as np

def set_global_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
