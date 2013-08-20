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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
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

#include "perf_precomp.hpp"
///////////// StereoMatchBM ////////////////////////
PERFTEST(StereoMatchBM)
{
	Mat left_image = imread(abspath("aloeL.jpg"), cv::IMREAD_GRAYSCALE);
	Mat right_image = imread(abspath("aloeR.jpg"), cv::IMREAD_GRAYSCALE);
	Mat disp,dst;
	ocl::oclMat d_left, d_right,d_disp;
	int n_disp= 128;
	int winSize =19;

	SUBTEST << left_image.cols << 'x' << left_image.rows << "; aloeL.jpg ;"<< right_image.cols << 'x' << right_image.rows << "; aloeR.jpg ";

	Ptr<StereoBM> bm = createStereoBM(n_disp, winSize);
	bm->compute(left_image, right_image, dst);

	CPU_ON;
	bm->compute(left_image, right_image, dst);
	CPU_OFF;

	d_left.upload(left_image);
	d_right.upload(right_image);

	ocl::StereoBM_OCL d_bm(0, n_disp, winSize);

	WARMUP_ON;
	d_bm(d_left, d_right, d_disp);
	WARMUP_OFF;

    cv::Mat ocl_mat;
    d_disp.download(ocl_mat);
    ocl_mat.convertTo(ocl_mat, dst.type());

	GPU_ON;
	d_bm(d_left, d_right, d_disp);
	GPU_OFF;

	GPU_FULL_ON;
	d_left.upload(left_image);
	d_right.upload(right_image);
	d_bm(d_left, d_right, d_disp);
	d_disp.download(disp);
	GPU_FULL_OFF;
    
    TestSystem::instance().setAccurate(-1, 0.);
}








	