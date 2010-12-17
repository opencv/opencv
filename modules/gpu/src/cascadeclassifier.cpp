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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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

#include "precomp.hpp"




using namespace cv;
using namespace cv::gpu;
using namespace std;

#if !defined (HAVE_CUDA)

cv::gpu::CascadeClassifier::CascadeClassifier()  { throw_nogpu(); }
cv::gpu::CascadeClassifier::CascadeClassifier(const string&)  { throw_nogpu(); }
cv::gpu::CascadeClassifier::~CascadeClassifier()  { throw_nogpu(); }

bool cv::gpu::CascadeClassifier::empty() const { throw_nogpu(); return true; }
bool cv::gpu::CascadeClassifier::load(const string& filename)  { throw_nogpu(); return true; }
bool cv::gpu::CascadeClassifier::read(const FileNode& node)  { throw_nogpu(); return true; }

void cv::gpu::CascadeClassifier::detectMultiScale( const Mat&, vector<Rect>&, double, int, int, Size, Size) { throw_nogpu(); }

       



#else


cv::gpu::CascadeClassifier::CascadeClassifier()
{

}

cv::gpu::CascadeClassifier::CascadeClassifier(const string& filename)
{

}

cv::gpu::CascadeClassifier::~CascadeClassifier()
{
    
}

bool cv::gpu::CascadeClassifier::empty() const
{
    int *a = (int*)&nppiStTranspose_32u_C1R;
    return *a == 0xFFFFF;
    return true;
}

bool cv::gpu::CascadeClassifier::load(const string& filename)
{
    return true;
}

bool cv::gpu::CascadeClassifier::read(const FileNode& node)
{
    return true;
}

void cv::gpu::CascadeClassifier::detectMultiScale( const Mat& image, vector<Rect>& objects, double scaleFactor, 
    int minNeighbors, int flags, Size minSize, Size maxSize)

{

}

#endif