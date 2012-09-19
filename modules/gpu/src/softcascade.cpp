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

#include <precomp.hpp>

#if !defined (HAVE_CUDA)

cv::gpu::SoftCascade::SoftCascade() : filds(0) { throw_nogpu(); }

cv::gpu::SoftCascade::SoftCascade( const string&, const float, const float) : filds(0) { throw_nogpu(); }

cv::gpu::SoftCascade::~SoftCascade() { throw_nogpu(); }

bool cv::gpu::SoftCascade::load( const string&, const float, const float) { throw_nogpu(); }

void cv::gpu::SoftCascade::detectMultiScale(const GpuMat&, const GpuMat&, GpuMat&, const int, Stream) { throw_nogpu(); }

#else

struct cv::gpu::SoftCascade::Filds
{
    bool fill(const FileNode &root, const float mins, const float maxs){return true;}
    void calcLevels(int frameW, int frameH, int scales) {}
};

cv::gpu::SoftCascade::SoftCascade() : filds(0) {}

cv::gpu::SoftCascade::SoftCascade( const string& filename, const float minScale, const float maxScale) : filds(0)
{
    load(filename, minScale, maxScale);
}

cv::gpu::SoftCascade::~SoftCascade()
{
    delete filds;
}

bool cv::gpu::SoftCascade::load( const string& filename, const float minScale, const float maxScale)
{
    if (filds)
        delete filds;
    filds = 0;

    cv::FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()) return false;

    filds = new Filds;
    Filds& flds = *filds;
    if (!flds.fill(fs.getFirstTopLevelNode(), minScale, maxScale)) return false;
    flds.calcLevels(FRAME_WIDTH, FRAME_HEIGHT, TOTAL_SCALES);

    return true;
}

void cv::gpu::SoftCascade::detectMultiScale(const GpuMat& /*image*/, const GpuMat& /*rois*/,
                                GpuMat& /*objects*/, const int /*rejectfactor*/, Stream /*stream*/)
{
    // empty
}

#endif
