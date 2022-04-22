'''
https://stackoverflow.com/questions/9905743/how-do-i-run-the-python-sdist-command-from-within-a-python-automated-script-wi
'''
from distutils.core import run_setup

import os
from .setup import get_version
# Get module name
MODULE_NAME = 'author_classification'

# Get version
VERSION = get_version(os.path.join(MODULE_NAME, "__init__.py"))

# Get real path of this file
this_file_path = os.path.realpath(__file__)

# Get the dir
this_file_dir = os.path.dirname(this_file_path)

# Change cwd to this file dir
os.chdir(this_file_dir)

# Run setup now that the cwd points to the right place (this dir)
run_setup('setup.py', script_args=['sdist', 'bdist_wheel'])