import yaml
from os import path

CONFIG = yaml.safe_load(open(f'{path.dirname(__file__)}/../resources/test_config.yml'))
TEST_FITS = f'{path.dirname(__file__)}/../resources/2018-03-22T02:07:54-dirty.fits'
TEST_MS = f'{path.dirname(__file__)}/../resources/14_2018-03-23T03:26:18_calibrated_flagged.ms'
