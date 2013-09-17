//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//

#ifdef DOUBLE_SUPPORT
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif

__kernel void convert_to(
        __global const srcT* restrict srcMat,
        __global dstT* dstMat,
        int cols1, int rows,
        int sstep1, int soffset1,
        int dstep1, int doffset1,
        float alpha, float beta)
{
        int x = get_global_id(0);
        int y = get_global_id(1);

        int srcidx = mad24(y, sstep1, x + soffset1);
        int dstidx = mad24(y, dstep1, x + doffset1);

        if ( (x < cols1) && (y < rows) )
        {
            float temp_src = convert_float(srcMat[srcidx]);
            dstMat[dstidx] = convertToDstType(temp_src*alpha+beta);
        }
}
