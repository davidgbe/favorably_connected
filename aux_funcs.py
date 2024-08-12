import numpy as np
from pathlib import Path

def zero_pad(s, n):
    s_str = str(s)
    pad = n - len(s_str)
    zero_padding = '0' * pad
    return zero_padding + s_str

def make_path_if_not_exists(path_str):
    Path(path_str).mkdir(parents=True, exist_ok=True)