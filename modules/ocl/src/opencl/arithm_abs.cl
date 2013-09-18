/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Sebastian Kramer, mail@kraymer.de
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
//M*/

// TODO: use vectors
// TODO: use offset for ROI support (see absdiff)
// TODO: support other image depths

/**************************************abs******************************************/
__kernel void arithm_abs_C1_D5 (__global const float *src, int src_step,
                                __global       float *dst, int dst_step,
                                int rows, int cols )
{
    const int2 g = (int2) ( get_global_id(0), get_global_id(1) );

    const int src_step_float = src_step / sizeof(float);
    const int dst_step_float = dst_step / sizeof(float);
    
    if ( g.x < cols && g.y < rows)
    {
        //if ( (x % 4) == 0 )  // in the future, use uchar4..
        //uchar4 src_values = vload4( ..
        
        dst[ mad24( g.y, dst_step_float, g.x) ] = fabs( src[ mad24( g.y, src_step_float, g.x ) ] );
    }
}
