from os import path
import getpass
import logging

import yaml

log = logging.getLogger(__name__)
ORCA_CONF_PATH = f'/home/{getpass.getuser()}/orca-conf.yml'

"""
This can potentially be turned into something that's more object-oriented. But for now this seems to do fine.
"""


def load_yaml(p: str):
    with open(p, 'r') as f:
        return yaml.safe_load(f)


if path.isfile(ORCA_CONF_PATH):
    config = load_yaml(ORCA_CONF_PATH)
else:
    log.info(f'{ORCA_CONF_PATH} not found. Using default configuration.')
    config = load_yaml(path.join(path.dirname(__file__), 'default-orca-conf.yml'))

queue_config = config['queue']