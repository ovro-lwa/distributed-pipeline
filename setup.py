from setuptools import setup, find_packages

setup(name="ovro-lwa-orca", packages=find_packages(),
      # scripts specify those that can be executed from command-line.
      scripts=['orca/flagging/flag_bls.py', 'orca/flagging/flag_bad_chans.py', 'orca/proj/start_workers.py'])
