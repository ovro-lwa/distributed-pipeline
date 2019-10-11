import yaml
from os import path

config = yaml.safe_load(open(f'{path.dirname(__file__)}/../resources/test_config.yml'))
