// This file is auto-generated. Do not edit!

#include "precomp.hpp"
#include "cvconfig.h"
#include "opencl_kernels_optflow.hpp"

#ifdef HAVE_OPENCL

namespace cv
{
namespace ocl
{
namespace optflow
{

const struct ProgramEntry sparse_matching_gpc={"sparse_matching_gpc",
"__kernel void getPatchDescriptor(\n"
"__global const uchar* imgCh0, int ic0step, int ic0off,\n"
"__global const uchar* imgCh1, int ic1step, int ic1off,\n"
"__global const uchar* imgCh2, int ic2step, int ic2off,\n"
"__global uchar* out, int outstep, int outoff,\n"
"const int gh, const int gw, const int PR  )\n"
"{\n"
"const int i = get_global_id(0);\n"
"const int j = get_global_id(1);\n"
"if (i >= gh || j >= gw)\n"
"return;\n"
"__global double* desc = (__global double*)(out + (outstep * (i * gw + j) + outoff));\n"
"const int patchRadius = PR * 2;\n"
"float patch[PATCH_RADIUS_DOUBLED][PATCH_RADIUS_DOUBLED];\n"
"for (int i0 = 0; i0 < patchRadius; ++i0) {\n"
"__global const float* ch0Row = (__global const float*)(imgCh0 + (ic0step * (i + i0) + ic0off + j * sizeof(float)));\n"
"for (int j0 = 0; j0 < patchRadius; ++j0)\n"
"patch[i0][j0] = ch0Row[j0];\n"
"}\n"
"#pragma unroll\n"
"for (int n0 = 0; n0 < 4; ++n0) {\n"
"#pragma unroll\n"
"for (int n1 = 0; n1 < 4; ++n1) {\n"
"double sum = 0;\n"
"for (int i0 = 0; i0 < patchRadius; ++i0)\n"
"for (int j0 = 0; j0 < patchRadius; ++j0)\n"
"sum += patch[i0][j0] * cos(CV_PI * (i0 + 0.5) * n0 / patchRadius) * cos(CV_PI * (j0 + 0.5) * n1 / patchRadius);\n"
"desc[n0 * 4 + n1] = sum / PR;\n"
"}\n"
"}\n"
"for (int k = 0; k < 4; ++k) {\n"
"desc[k] *= SQRT2_INV;\n"
"desc[k * 4] *= SQRT2_INV;\n"
"}\n"
"double sum = 0;\n"
"for (int i0 = 0; i0 < patchRadius; ++i0) {\n"
"__global const float* ch1Row = (__global const float*)(imgCh1 + (ic1step * (i + i0) + ic1off + j * sizeof(float)));\n"
"for (int j0 = 0; j0 < patchRadius; ++j0)\n"
"sum += ch1Row[j0];\n"
"}\n"
"desc[16] = sum / patchRadius;\n"
"sum = 0;\n"
"for (int i0 = 0; i0 < patchRadius; ++i0) {\n"
"__global const float* ch2Row = (__global const float*)(imgCh2 + (ic2step * (i + i0) + ic2off + j * sizeof(float)));\n"
"for (int j0 = 0; j0 < patchRadius; ++j0)\n"
"sum += ch2Row[j0];\n"
"}\n"
"desc[17] = sum / patchRadius;\n"
"}\n"
, "4de6dbd7b34900887da8399ec2e431b0"};
ProgramSource sparse_matching_gpc_oclsrc(sparse_matching_gpc.programStr);
const struct ProgramEntry updatemotionhistory={"updatemotionhistory",
"__kernel void updateMotionHistory(__global const uchar * silh, int silh_step, int silh_offset,\n"
"__global uchar * mhiptr, int mhi_step, int mhi_offset, int mhi_rows, int mhi_cols,\n"
"float timestamp, float delbound)\n"
"{\n"
"int x = get_global_id(0);\n"
"int y = get_global_id(1);\n"
"if (x < mhi_cols && y < mhi_rows)\n"
"{\n"
"int silh_index = mad24(y, silh_step, silh_offset + x);\n"
"int mhi_index = mad24(y, mhi_step, mhi_offset + x * (int)sizeof(float));\n"
"silh += silh_index;\n"
"__global float * mhi = (__global float *)(mhiptr + mhi_index);\n"
"float val = mhi[0];\n"
"val = silh[0] ? timestamp : val < delbound ? 0 : val;\n"
"mhi[0] = val;\n"
"}\n"
"}\n"
, "b19beb01d0c6052524049341b55a2be5"};
ProgramSource updatemotionhistory_oclsrc(updatemotionhistory.programStr);
}
}}
#endif
