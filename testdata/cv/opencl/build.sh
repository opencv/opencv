#!/bin/bash -e
# Usage: build.sh <path_to_intel_opencl_sdk>/bin/ioc64
# 
# ioc64 is a tool from Intel(R) SDK for OpenCL(TM): https://software.intel.com/en-us/intel-opencl

IOC64_TOOL=${1:-ioc64}

${IOC64_TOOL} -input=test_kernel.cl \
  -spir64=test_kernel.spir64 -spir32=test_kernel.spir32 \
  -spirv64=test_kernel.spirv64 -spirv32=test_kernel.spirv32
