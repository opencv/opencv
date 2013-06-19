#/usr/bin/env python

# add the hdr_parser to the path
import sys
sys.path.append(sys.argv[1])

# get the IO from the command line arguments
input_files = sys.argv[2:-1]
output_dir  = sys.argv[-1]

# create the generator
from gen_matlab import MatlabWrapperGenerator
mwg = MatlabWrapperGenerator()
mwg.gen(input_files, output_dir)
