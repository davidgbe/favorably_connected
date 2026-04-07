import json
import sys
import os
from pathlib import Path

if __name__ == '__main__':
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))

    from aux_funcs import make_path_if_not_exists

    for i in range(6):
        curr_style = 'FIXED' if i < 3 else 'MIXED'

        run_data = {
            'run_idx': i,
            'curr_style': curr_style,
            'connectivity': 'VANILLA',
            'activity_reg': 1.,
            'noise_var': 1e-4,
        }

        run_dir_path = os.path.join(
            curr_file_path.parent.parent.parent,
            f'results',
        ).replace('\\', '/')

        make_path_if_not_exists(run_dir_path)

        file_name = f'run_{i}.json'

        print(os.path.join(run_dir_path, file_name))

        with open(os.path.join(run_dir_path, file_name), 'w') as f:
            json.dump(run_data, f)