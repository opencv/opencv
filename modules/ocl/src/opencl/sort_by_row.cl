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
//    Matthias Bady, aegirxx ==> gmail.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
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

#ifndef pow2Size
 #define pow2Size 0
#endif

#ifndef cols
 #define cols 0
#endif

#ifndef rows
 #define rows 0
#endif

#ifndef rowStep
 define rowStep 0
#endif

#ifndef rowKey
 #define rowKey 0
#endif

#ifndef ch
 #define ch 1
#endif

#ifndef ValueT
 #define ValueT float
#endif

#ifndef IS_GT
 #define IS_GT false
#endif

#ifndef VEC_CMP
 #define VEC_CMP any
#endif

#if IS_GT
 #if ch > 1
  #define my_comp(x,y) (VEC_CMP(isless(x, y)))
 #else
  #define my_comp(x,y) ((x) < (y))
 #endif
#else
 #if ch > 1
  #define my_comp(x,y) (VEC_CMP(isgreater(x, y)))
 #else
  #define my_comp(x,y) ((x) > (y))
 #endif
#endif

__kernel void bitonicSortByRow( __global ValueT * vals, int stage, int pass)
{
    const int threadId = get_global_id(0);
    if(threadId >= pow2Size / 2)
    {
        return;
    }
    const int pairDistance = 1 << (stage - pass);
    const int blockWidth   = 2 * pairDistance;
    const int leftId = (threadId % pairDistance) + (threadId / pairDistance) * blockWidth;
    if(leftId >= cols)
    {
        return;
    }
    const int rightId = (leftId + blockWidth) - ((pass==0) ? (1 + 2*(leftId % blockWidth)) : pairDistance);
    if(rightId >= cols)
    {
        return;
    }
    const bool compareResult = my_comp( vals[leftId + rowKey*rowStep], vals[rightId + rowKey*rowStep] );
    if(compareResult)
    {
        ValueT tmp;
        for(int i = 0; i<rows; ++i)
        {
            tmp = vals[leftId + i*rowStep];
            vals[leftId + i*rowStep] = vals[rightId + i*rowStep];
            vals[rightId  + i*rowStep] = tmp;
        }
    }
}