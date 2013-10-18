#!/bin/bash -e
echo "Generate files for CL runtime..."
cat sources/opencl11/cl.h | python parser_cl.py cl_runtime_opencl11
cat sources/opencl12/cl.h | python parser_cl.py cl_runtime_opencl12
cat sources/clAmdBlas.h | python parser_clamdblas.py
cat sources/clAmdFft.h | python parser_clamdfft.py
echo "Generate files for CL runtime... Done"
