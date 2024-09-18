import numpy as np
from pathlib import Path
from copy import deepcopy as copy


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