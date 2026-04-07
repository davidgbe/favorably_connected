if __name__ == '__main__':
    import sys
    from pathlib import Path
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))

import numpy as np
from environments.treadmill_session import TreadmillSession
from environments.components.patch import Patch

DWELL_TIME_FOR_REWARD = 4
SPATIAL_BUFFER_FOR_VISUAL_CUES = 1.5

if __name__ == '__main__':
    # generate 3 types of patches
    patches = [
        Patch(2, 3, 3, lambda x: 1, 0),
        Patch(3, 2, 2, lambda x: 2, 1),
        Patch(3, 3, 3, lambda x: 3, 2),
    ]

    transition_mat = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ])

    sesh = TreadmillSession(
        patches,
        transition_mat,
        10,
        DWELL_TIME_FOR_REWARD,
        SPATIAL_BUFFER_FOR_VISUAL_CUES,
        obs_size=len(patches) + 2,
        verbosity=True,
        first_patch_start=5,
    )

    if not (sesh.total_reward == 0 and sesh.current_position == 0 and sesh.reward_site_dwell_time == 0 and (sesh.get_observations() == np.zeros((5))).all()):
        print()
        raise ValueError('0')

    for k in range(5):
        sesh.move_forward(1)

    if not (sesh.total_reward == 0 and sesh.current_position == 5 and sesh.reward_site_dwell_time == 0 and (sesh.get_observations() == np.array([1, 0, 0, 0, 0])).all()):
        print()
        raise ValueError('1')

    for k in range(3):
        sesh.move_forward(1)

    if not (sesh.total_reward == 0 and sesh.current_position == 8 and sesh.reward_site_dwell_time == 1 and (sesh.get_observations() == np.array([0, 0, 1, 0, 0])).all()):
        raise ValueError('2')

    for k in range(3):
        sesh.move_forward(0)

    if not (sesh.total_reward == 1 and sesh.current_position == 8 and sesh.reward_site_dwell_time == 4 and (sesh.get_observations() == np.array([0, 0, 1, 0, 0])).all()):
        raise ValueError('3')

    if not (sesh.get_reward_site_idx_of_current_pos() == 0):
        raise ValueError(f'Wrong reward_site_idx: {sesh.get_reward_site_idx_of_current_pos()}')

    for k in range(3):
        sesh.move_forward(1)

    if not (sesh.total_reward == 1 and sesh.current_position == 11 and sesh.reward_site_dwell_time == 0 and (sesh.get_observations() == np.array([0, 0, 0, 0, 0])).all()):
        raise ValueError('4')

    for k in range(3):
        sesh.move_forward(1)

    if not (sesh.total_reward == 1 and sesh.current_position == 14 and sesh.reward_site_dwell_time == 1 and (sesh.get_observations() == np.array([0, 0, 1, 0, 0])).all()):
        raise ValueError('5')

    for k in range(3):
        sesh.move_forward(0)

    if not (sesh.total_reward == 2 and sesh.current_position == 14 and sesh.reward_site_dwell_time == 4):
        raise ValueError('6')

    for k in range(3):
        sesh.move_forward(0)

    if not (sesh.total_reward == 2 and sesh.current_position == 14 and sesh.reward_site_dwell_time == 7):
        raise ValueError('6b')

    for k in range(2):
        sesh.move_forward(1)

    if not (sesh.total_reward == 2 and sesh.current_position == 16 and sesh.reward_site_dwell_time == 9 and (sesh.get_observations() == np.array([0, 1, 1, 0, 0])).all()):
        raise ValueError('5')

    sesh.move_forward(1)

    if not (sesh.total_reward == 2 and sesh.current_position == 17 and sesh.reward_site_dwell_time == 0):
        raise ValueError('5b')

    for k in range(10):
        sesh.move_forward(1)

    if not (sesh.total_reward == 2 and sesh.current_position == 27 and sesh.reward_site_dwell_time == 0):
        raise ValueError('6')

    sesh.move_forward(1)

    if not (sesh.total_reward == 2 and sesh.current_position == 28 and sesh.reward_site_dwell_time == 0):
        raise ValueError('7')
    
    sesh.move_forward(1)

    if not (sesh.total_reward == 2 and sesh.current_position == 29 and sesh.reward_site_dwell_time == 1):
        raise ValueError('8')

    for k in range(3):
        sesh.move_forward(0)

    if not (sesh.total_reward == 4 and sesh.current_position == 29 and sesh.reward_site_dwell_time == 4):
        raise ValueError('9')

    for k in range(6):
        sesh.move_forward(1)

    if not (sesh.total_reward == 4 and sesh.current_position == 35 and sesh.reward_site_dwell_time == 0):
        raise ValueError('10')