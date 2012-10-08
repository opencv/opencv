//M///////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
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
// This software is provided by the copyright holders and contributors "as is" and
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
//M


#ifndef __OPENCV_ICF_HPP__
#define __OPENCV_ICF_HPP__

#include <opencv2/gpu/device/common.hpp>
#include <stdio.h>

#if defined __CUDACC__
# define __device __device__ __forceinline__
#else
# define __device
#endif


namespace cv { namespace gpu { namespace device {
namespace icf {

struct __align__(16) Octave
{
    ushort index;
    ushort stages;
    ushort shrinkage;
    ushort2 size;
    float scale;

    Octave(const ushort i, const ushort s, const ushort sh, const ushort2 sz, const float sc)
    : index(i), stages(s), shrinkage(sh), size(sz), scale(sc) {}
};

struct __align__(8) Level //is actually 24 bytes
{
    int octave;

    float relScale;
    float shrScale;   // used for marking detection
    float scaling[2]; // calculated according to Dollal paper

    // for 640x480 we can not get overflow
    uchar2 workRect;
    uchar2 objSize;

    Level(int idx, const Octave& oct, const float scale, const int w, const int h)
    :  octave(idx), relScale(scale / oct.scale), shrScale (relScale / (float)oct.shrinkage)
    {
        workRect.x = round(w / (float)oct.shrinkage);
        workRect.y = round(h / (float)oct.shrinkage);

        objSize.x  = round(oct.size.x * relScale);
        objSize.y  = round(oct.size.y * relScale);
    }

    __device Level(){}
};

struct __align__(8) Node
{
    uchar4 rect;
    // ushort channel;
    uint threshold;

    enum { THRESHOLD_MASK = 0x0FFFFFFF };

    Node(const uchar4 r, const uint ch, const uint t) : rect(r), threshold(t + (ch << 28))
    {
        // printf("%d\n", t);
        // printf("[%d %d %d %d] %d, %d\n",rect.x, rect.y, rect.z, rect.w, (int)(threshold >> 28),
        //     (int)(0x0FFFFFFF & threshold));
    }
};

struct __align__(16) Detection
{
    ushort x;
    ushort y;
    ushort w;
    ushort h;
    float confidence;
    int kind;

    Detection(){}
    __device Detection(int _x, int _y, uchar _w, uchar _h, float c)
    : x(_x), y(_y), w(_w), h(_h), confidence(c), kind(0) {};
};

}
}}}

#endif