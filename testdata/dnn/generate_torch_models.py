#!/usr/bin/env python

import os
from subprocess import check_call

if __name__ == '__main__':
    check_call(['th', os.path.abspath('torch/torch_gen_test_data.lua')], cwd='torch')
