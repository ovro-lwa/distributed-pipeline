from setuptools import setup, find_packages

setup(name="ovro-lwa-orca", packages=find_packages(exclude=['tests', 'tests.*']),
      # scripts specify those that can be executed from command-line.
      scripts=['orca/flagging/flag_bls.py', 'orca/flagging/flag_bad_chans.py', 'orca/proj/start_workers.py'],
      version='0.1',
      package_data={'': 'orca.resources.*'},
      include_package_data=True
      )
