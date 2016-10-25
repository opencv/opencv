#!/bin/bash -e
echo "Generate files for CL runtime..."
python parser_cl.py opencl_core < sources/cl.h
python parser_clamdblas.py < sources/clAmdBlas.h
python parser_clamdfft.py < sources/clAmdFft.h

python parser_cl.py opencl_gl < sources/cl_gl.h
echo "Generate files for CL runtime... Done"
