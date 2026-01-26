"""Manages configuration

It's effectively a singleton using module imports.
"""
from os import path
import getpass
import logging
import pkg_resources

import yaml
from types import SimpleNamespace

log = logging.getLogger(__name__)

# Try home directory first, then fall back to default config in package
ORCA_CONF_PATH = path.expanduser(f'~/orca-conf.yml')
DEFAULT_CONF_PATH = path.join(path.dirname(__file__), 'default-orca-conf.yml')

"""
This can potentially be turned into something that's more object-oriented. But for now this seems to do fine.
"""


def load_yaml(p: str):
    with open(p, 'r') as f:
        return yaml.safe_load(f)


if path.isfile(ORCA_CONF_PATH):
    config = load_yaml(ORCA_CONF_PATH)
elif path.isfile(DEFAULT_CONF_PATH):
    log.warning(f'{ORCA_CONF_PATH} not found. Using default config from {DEFAULT_CONF_PATH}')
    config = load_yaml(DEFAULT_CONF_PATH)
else:
    msg = f'{ORCA_CONF_PATH} not found. Please put the config file there.'
    log.error(msg)
    raise FileNotFoundError(msg)

queue_config = SimpleNamespace(**config['queue'])
telescope = SimpleNamespace(**config['telescope'])
execs = SimpleNamespace(**config['execs'])
cluster = config['cluster']
