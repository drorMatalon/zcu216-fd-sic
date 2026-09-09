# Reads the settings every executable shares out of config/parameters.json.

# =========================================
# Imports and defines
# =========================================
import json
import os

CONFIG_PATH = "config/parameters.json"

# =========================================
# Classes and functions
# =========================================
def load_config(path = CONFIG_PATH):
    """The shared settings as a nested dict, one section per group.

    The notebooks used to spell these values out one by one, and a run only meant
    something if all four spellings agreed. One file makes disagreement impossible
    rather than merely detected.
    """
    with open(resolve_config_path(path)) as handle:
        return json.load(handle)

def resolve_config_path(path):
    """Find the config from the project root or from beside the package."""
    if os.path.isfile(path):
        return path
    beside_package = os.path.join(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))), path)
    if os.path.isfile(beside_package):
        return beside_package
    raise FileNotFoundError("Cannot find %s" % path)
