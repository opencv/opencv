#/usr/bin/env python

import sys
from gen_matlab import MatlabWrapperGenerator

# get the IO from the command line arguments
input_files = sys.argv[1:-1]
output_dir  = sys.argv[-1]

# create the generator
mwg = MatlabWrapperGenerator()
mwg.gen(input_files, output_dir)
