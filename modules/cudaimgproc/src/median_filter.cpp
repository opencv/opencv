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

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::cuda::medianFiltering(InputArray, OutputArray, int,int) { throw_no_cuda(); }

#else /* !defined (HAVE_CUDA) */

////////////////////////////////////////////////////////////////////////
// medianFiltering

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        void medianFiltering_gpu(const PtrStepSzb src, PtrStepSzb dst, PtrStepSzi devHist, PtrStepSzi devCoarseHist,int kernel, int partitions);
    }
}}}



void cv::cuda::medianFiltering(InputArray _src, OutputArray _dst, int _kernel, int _partitions){
    using namespace cv::cuda::device::imgproc;

    GpuMat src = _src.getGpuMat();
    CV_Assert( src.type() == CV_8UC1 );

    int partitions = _partitions;
	if (partitions>src.rows)
        partitions=src.rows;

    int kernel=_kernel;
    if (kernel>src.rows)
        kernel=src.rows;
    if (kernel>src.cols)
        kernel=src.cols;
    if(kernel%2==0)
        kernel--;
    CV_Assert(kernel>=3);

     _dst.create(src.rows, src.cols, src.type());
    GpuMat dst = _dst.getGpuMat();
    src.copyTo(dst);

    // Note - these are hardcoded in the actual GPU kernel. Do not change these values.
    int histSize=256, histCoarseSize=8;

    GpuMat devHist(1, src.cols*histSize*partitions,CV_32SC1);
    GpuMat devCoarseHist(1,src.cols*histCoarseSize*partitions,CV_32SC1);
    devHist.setTo(0);
    devCoarseHist.setTo(0);

    medianFiltering_gpu(src,dst,devHist, devCoarseHist,kernel,partitions);

    devHist.release();
    devCoarseHist.release();
}


#endif /* !defined (HAVE_CUDA) */
