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
//M*/

#include <opencv2/gpu/device/common.hpp>

#ifndef __OPENCV_ICF_HPP__
#define __OPENCV_ICF_HPP__

#if defined __CUDACC__
# define __hd__ __host__ __device__ __forceinline__
#else
# define __hd__
#endif


namespace icf {

using cv::gpu::PtrStepSzb;
using cv::gpu::PtrStepSzf;

struct Cascade
{
    Cascade() {}
    Cascade(const cv::gpu::PtrStepSzb& octs, const cv::gpu::PtrStepSzf& sts, const cv::gpu::PtrStepSzb& nds,
        const cv::gpu::PtrStepSzf& lvs, const cv::gpu::PtrStepSzb& fts, const cv::gpu::PtrStepSzb& lls)
    : octaves(octs), stages(sts), nodes(nds), leaves(lvs), features(fts), levels(lls) {}

    void detect(const cv::gpu::PtrStepSzb& hogluv) const;

    PtrStepSzb octaves;
    PtrStepSzf stages;
    PtrStepSzb nodes;
    PtrStepSzf leaves;
    PtrStepSzb features;

    PtrStepSzb levels;

};

struct ChannelStorage
{
    ChannelStorage(){}
    ChannelStorage(const cv::gpu::PtrStepSzb& buff, const cv::gpu::PtrStepSzb& shr,
        const cv::gpu::PtrStepSzb& itg, const int s)
    : dmem (buff), shrunk(shr), hogluv(itg), shrinkage(s) {}

    void frame(const cv::gpu::PtrStepSz<uchar4>& image);

    PtrStepSzb dmem;
    PtrStepSzb shrunk;
    PtrStepSzb hogluv;

    int shrinkage;
};

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

struct __align__(8) Node
{
    int feature;
    float threshold;

    Node(const int f, const float t) : feature(f), threshold(t) {}
};

struct __align__(8) Feature
{
    int channel;
    uchar4 rect;

    Feature(const int c, const uchar4 r) : channel(c), rect(r) {}
};

struct __align__(8) Level //is actually 24 bytes
{
    int octave;

    // float origScale; //not actually used
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
};
}

#endif