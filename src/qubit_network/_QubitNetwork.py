"""
Internal functions used in `QubitNetwork.QubitNetwork`.
"""
import os
import re

def _find_suitable_name(filename):
    # if the file doesn't already exists, just return unchanged
    if not os.path.isfile(filename):
        return filename

    basename, ext = os.path.splitext(filename)
    matcher = re.compile(r'(.*)(\([0-9]+\))$')
    match = matcher.match(basename)
    if match is None:
        new_index = 1
    else:
        basename = match.group(1)
        new_index = int(match.group(2)[1:-1])

    new_basename = basename + '(' + str(new_index) + ')'
    while os.path.isfile(new_basename + ext):
        new_index += 1
        new_basename = basename + '(' + str(new_index) + ')'

    filename = new_basename + ext
    return filename
