import numpy as np
from pathlib import Path
from copy import deepcopy as copy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import warnings
import pickle
import blosc
import os
import json


def zero_pad(s, n):
    s_str = str(s)
    pad = n - len(s_str)
    zero_padding = '0' * pad
    return zero_padding + s_str


def make_path_if_not_exists(path_str):
    Path(path_str).mkdir(parents=True, exist_ok=True)


def parse_string_with_regex(s, pattern):
    s_copy = copy(s)
    pattern_copy = copy(pattern)

    fragments_to_match = []

    while pattern_copy != '':
        ast_idx = pattern_copy.find('*')
        if ast_idx < 0:
            if pattern_copy != '':
                fragments_to_match.append(pattern_copy)
            break
        fragment = pattern_copy[:ast_idx]
        pattern_copy = pattern_copy[ast_idx + 1:]
        if fragment == '' and ast_idx != len(pattern_copy) - 1:
            fragments_to_match.append('*')
            continue
        fragments_to_match.append(fragment)
        if ast_idx >= 0:
            fragments_to_match.append('*')

    wildcard_matches = []
    star_encountered = False
    for idx, fragment in enumerate(fragments_to_match):
        if fragment == '*':
            star_encountered = True
            if idx == len(fragments_to_match) - 1:
                wildcard_matches.append(s_copy)
                s_copy = ''
        else:
            frag_idx = s_copy.find(fragment)
            if frag_idx < 0:
                raise ValueError(f'String does not match regex:\n{s}\n{pattern}')
            else:
                if frag_idx > 0:
                    if star_encountered:
                        wildcard_matches.append(s_copy[:frag_idx])
                    else:
                        raise ValueError(f'String does not match regex:\n{s}\n{pattern}')
                s_copy = s_copy[frag_idx + len(fragment):]
    if s_copy != '':
        raise ValueError(f'String does not match regex:\n{s}\n{pattern}')   
    return wildcard_matches


def ordered_colors_from_cmap(cmap_name, n, cmap_range=(0, 1)):
    cmap = matplotlib.colormaps[cmap_name]
    colors = cmap(np.linspace(cmap_range[0], cmap_range[1], n))
    return colors


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.s
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def compressed_write(data, dest):
    pickled_data = pickle.dumps(data)
    compressed_pickle = blosc.compress(pickled_data)
    with open(dest, 'wb') as f:
        f.write(compressed_pickle)


def compressed_read(source):
    with open(source, 'rb') as f:
        compressed_pickle = f.read()
    inflated_pickle = blosc.decompress(compressed_pickle)
    return pickle.loads(inflated_pickle)


def logical_and(*args):
    v = None
    for i in range(len(args)):
        if v is None:
            v = args[i]
        else:
            v = np.logical_and(v, args[i])
    return v


def load_first_json(directory):
    """Loads the first JSON file found in a directory."""
    try:
        # List all files in the directory
        files = sorted(os.listdir(directory))
        
        # Find the first JSON file
        for file in files:
            if file.lower().endswith('.json'):
                json_path = os.path.join(directory, file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data, file
        
        print("No JSON files found in the directory.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None