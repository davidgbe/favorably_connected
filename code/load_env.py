import json
import sys
import os
from pathlib import Path

def get_env_vars(env):
    curr_file_path = Path(__file__)
    return json.load(open(os.path.join(curr_file_path.parent, 'ENV_FILE.json')))[env]