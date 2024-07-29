import numpy as np

def zero_pad(s, n):
    s_str = str(s)
    pad = n - len(s_str)
    zero_padding = '0' * pad
    return zero_padding + s_str