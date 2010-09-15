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

void cv::gpu::add(const GpuMat& src1, const GpuMat& src2, GpuMat& dst) { throw_nogpu(); }
void cv::gpu::subtract(const GpuMat& src1, const GpuMat& src2, GpuMat& dst) { throw_nogpu(); }
void cv::gpu::multiply(const GpuMat& src1, const GpuMat& src2, GpuMat& dst) { throw_nogpu(); }
void cv::gpu::divide(const GpuMat& src1, const GpuMat& src2, GpuMat& dst) { throw_nogpu(); }

void cv::gpu::transpose(const GpuMat& src1, GpuMat& dst) { throw_nogpu(); }

void cv::gpu::absdiff(const GpuMat& src1, const GpuMat& src2, GpuMat& dst) { throw_nogpu(); }

double cv::gpu::threshold(const GpuMat& src, GpuMat& dst, double thresh, double maxVal, int thresholdType) { throw_nogpu(); return 0.0; }

void cv::gpu::compare(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, int cmpop) { throw_nogpu(); }

void cv::gpu::meanStdDev(const GpuMat& mtx, Scalar& mean, Scalar& stddev) { throw_nogpu(); }

double cv::gpu::norm(const GpuMat& src1, int normType) { throw_nogpu(); return 0.0; }
double cv::gpu::norm(const GpuMat& src1, const GpuMat& src2, int normType) { throw_nogpu(); return 0.0; }

void cv::gpu::flip(const GpuMat& a, GpuMat& b, int flipCode) { throw_nogpu(); }

void cv::gpu::resize(const GpuMat& src, GpuMat& dst, Size dsize, double fx, double fy, int interpolation) { throw_nogpu(); }

Scalar cv::gpu::sum(const GpuMat& m) { throw_nogpu(); return Scalar(); }

void cv::gpu::minMax(const GpuMat& src, double* minVal, double* maxVal) { throw_nogpu(); }

void cv::gpu::copyConstBorder(const GpuMat& src, GpuMat& dst, int top, int bottom, int left, int right, const Scalar& value) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace
{
	typedef NppStatus (*npp_binary_func_8u_scale_t)(const Npp8u* pSrc1, int nSrc1Step, const Npp8u* pSrc2, int nSrc2Step, Npp8u* pDst, int nDstStep, 
											  NppiSize oSizeROI, int nScaleFactor);
	typedef NppStatus (*npp_binary_func_32f_t)(const Npp32f* pSrc1, int nSrc1Step, const Npp32f* pSrc2, int nSrc2Step, Npp32f* pDst, 
									     int nDstStep, NppiSize oSizeROI);

	void nppFuncCaller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, 
					   npp_binary_func_8u_scale_t npp_func_8uc1, npp_binary_func_8u_scale_t npp_func_8uc4, npp_binary_func_32f_t npp_func_32fc1)
	{
        CV_DbgAssert(src1.size() == src2.size() && src1.type() == src2.type());

        CV_Assert(src1.type() == CV_8UC1 || src1.type() == CV_8UC4 || src1.type() == CV_32FC1);

        dst.create( src1.size(), src1.type() );

		NppiSize sz;
		sz.width  = src1.cols;
		sz.height = src1.rows;

		if (src1.depth() == CV_8U)
		{
			if (src1.channels() == 1)
			{
				npp_func_8uc1((const Npp8u*)src1.ptr<char>(), src1.step, 
					(const Npp8u*)src2.ptr<char>(), src2.step, 
					(Npp8u*)dst.ptr<char>(), dst.step, sz, 0);
			}
			else
			{
				npp_func_8uc4((const Npp8u*)src1.ptr<char>(), src1.step, 
					(const Npp8u*)src2.ptr<char>(), src2.step, 
					(Npp8u*)dst.ptr<char>(), dst.step, sz, 0);
			}        
		}
		else //if (src1.depth() == CV_32F)
		{
			npp_func_32fc1((const Npp32f*)src1.ptr<float>(), src1.step,
				(const Npp32f*)src2.ptr<float>(), src2.step,
				(Npp32f*)dst.ptr<float>(), dst.step, sz);
		}
	}
}

void cv::gpu::add(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)
{
	nppFuncCaller(src1, src2, dst, nppiAdd_8u_C1RSfs, nppiAdd_8u_C4RSfs, nppiAdd_32f_C1R);
}

void cv::gpu::subtract(const GpuMat& src1, const GpuMat& src2, GpuMat& dst) 
{
	nppFuncCaller(src2, src1, dst, nppiSub_8u_C1RSfs, nppiSub_8u_C4RSfs, nppiSub_32f_C1R);
}

void cv::gpu::multiply(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)
{
	nppFuncCaller(src1, src2, dst, nppiMul_8u_C1RSfs, nppiMul_8u_C4RSfs, nppiMul_32f_C1R);
}

void cv::gpu::divide(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)
{
	nppFuncCaller(src2, src1, dst, nppiDiv_8u_C1RSfs, nppiDiv_8u_C4RSfs, nppiDiv_32f_C1R);
}

void cv::gpu::transpose(const GpuMat& src, GpuMat& dst)
{
    CV_Assert(src.type() == CV_8UC1);

    dst.create( src.cols, src.rows, src.type() );

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    nppiTranspose_8u_C1R((const Npp8u*)src.ptr<char>(), src.step, (Npp8u*)dst.ptr<char>(), dst.step, sz);
}

void cv::gpu::absdiff(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)
{
	CV_DbgAssert(src1.size() == src2.size() && src1.type() == src2.type());

	CV_Assert((src1.depth() == CV_8U || src1.depth() == CV_32F) && src1.channels() == 1);

    dst.create( src1.size(), src1.type() );

    NppiSize sz;
    sz.width  = src1.cols;
    sz.height = src1.rows;

    if (src1.depth() == CV_8U)
    {
        nppiAbsDiff_8u_C1R((const Npp8u*)src1.ptr<char>(), src1.step, 
                (const Npp8u*)src2.ptr<char>(), src2.step, 
                (Npp8u*)dst.ptr<char>(), dst.step, sz);
    }
    else //if (src1.depth() == CV_32F)
    {
        nppiAbsDiff_32f_C1R((const Npp32f*)src1.ptr<float>(), src1.step,
            (const Npp32f*)src2.ptr<float>(), src2.step,
            (Npp32f*)dst.ptr<float>(), dst.step, sz);
    }
}

double cv::gpu::threshold(const GpuMat& src, GpuMat& dst, double thresh, double /*maxVal*/, int thresholdType) 
{ 
    CV_Assert(src.type() == CV_32FC1 && thresholdType == THRESH_TRUNC);

    dst.create( src.size(), src.type() );

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    nppiThreshold_32f_C1R((const Npp32f*)src.ptr<float>(), src.step, 
        (Npp32f*)dst.ptr<float>(), dst.step, sz, (Npp32f)thresh, NPP_CMP_GREATER);

    return thresh;
}

void cv::gpu::compare(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, int cmpop) 
{
    CV_DbgAssert(src1.size() == src2.size() && src1.type() == src2.type());

    CV_Assert((src1.type() == CV_8UC4 || src1.type() == CV_32FC1) && cmpop != CMP_NE);

    dst.create( src1.size(), CV_8UC1 );

    static const NppCmpOp nppCmpOp[] = { NPP_CMP_EQ, NPP_CMP_GREATER, NPP_CMP_GREATER_EQ, NPP_CMP_LESS, NPP_CMP_LESS_EQ };

    NppiSize sz;
    sz.width  = src1.cols;
    sz.height = src1.rows;

    if (src1.depth() == CV_8U)
    {
        nppiCompare_8u_C4R((const Npp8u*)src1.ptr<char>(), src1.step, 
            (const Npp8u*)src2.ptr<char>(), src2.step, 
            (Npp8u*)dst.ptr<char>(), dst.step, sz, nppCmpOp[cmpop]);
    }
    else //if (src1.depth() == CV_32F)
    {
        nppiCompare_32f_C1R((const Npp32f*)src1.ptr<float>(), src1.step,
            (const Npp32f*)src2.ptr<float>(), src2.step,
            (Npp8u*)dst.ptr<char>(), dst.step, sz, nppCmpOp[cmpop]);
    }
}

void cv::gpu::meanStdDev(const GpuMat& src, Scalar& mean, Scalar& stddev) 
{
    CV_Assert(src.type() == CV_8UC1);

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    nppiMean_StdDev_8u_C1R((const Npp8u*)src.ptr<char>(), src.step, sz, mean.val, stddev.val);
}

double cv::gpu::norm(const GpuMat& src1, int normType) 
{
    return norm(src1, GpuMat(src1.size(), src1.type(), Scalar::all(0.0)), normType);
}

double cv::gpu::norm(const GpuMat& src1, const GpuMat& src2, int normType)
{
    CV_DbgAssert(src1.size() == src2.size() && src1.type() == src2.type());

    CV_Assert((src1.type() == CV_8UC1) && (normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2));

    typedef NppStatus (*npp_norm_diff_func_t)(const Npp8u* pSrc1, int nSrcStep1, const Npp8u* pSrc2, int nSrcStep2, 
        NppiSize oSizeROI, Npp64f* pRetVal);

    static const npp_norm_diff_func_t npp_norm_diff_func[] = {nppiNormDiff_Inf_8u_C1R, nppiNormDiff_L1_8u_C1R, nppiNormDiff_L2_8u_C1R};

    NppiSize sz;
    sz.width  = src1.cols;
    sz.height = src1.rows;

    int funcIdx = normType >> 1;
    Scalar retVal;

    npp_norm_diff_func[funcIdx]((const Npp8u*)src1.ptr<char>(), src1.step, 
        (const Npp8u*)src2.ptr<char>(), src2.step, 
        sz, retVal.val);

    return retVal[0];
}

void cv::gpu::flip(const GpuMat& src, GpuMat& dst, int flipCode)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4);

    dst.create( src.size(), src.type() );

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    if (src.channels() == 1)
    {
        nppiMirror_8u_C1R((const Npp8u*)src.ptr<char>(), src.step, 
            (Npp8u*)dst.ptr<char>(), dst.step, sz, 
            (flipCode == 0 ? NPP_HORIZONTAL_AXIS : (flipCode > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS)));
    }
    else
    {
        nppiMirror_8u_C4R((const Npp8u*)src.ptr<char>(), src.step, 
            (Npp8u*)dst.ptr<char>(), dst.step, sz, 
            (flipCode == 0 ? NPP_HORIZONTAL_AXIS : (flipCode > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS)));
    }
}

void cv::gpu::resize(const GpuMat& src, GpuMat& dst, Size dsize, double fx, double fy, int interpolation)
{
    static const int npp_inter[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC, 0, NPPI_INTER_LANCZOS};

    CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC4) && 
        (interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC || interpolation == INTER_LANCZOS4));

    CV_Assert( src.size().area() > 0 );
    CV_Assert( !(dsize == Size()) || (fx > 0 && fy > 0) );
    if( dsize == Size() )
    {
        dsize = Size(saturate_cast<int>(src.cols * fx), saturate_cast<int>(src.rows * fy));
    }
    else
    {
        fx = (double)dsize.width / src.cols;
        fy = (double)dsize.height / src.rows;
    }
    dst.create(dsize, src.type());

    NppiSize srcsz;
    srcsz.width  = src.cols;
    srcsz.height = src.rows;
    NppiRect srcrect;
    srcrect.x = srcrect.y = 0;
    srcrect.width  = src.cols;
    srcrect.height = src.rows;
    NppiSize dstsz;
    dstsz.width  = dst.cols;
    dstsz.height = dst.rows;

    if (src.channels() == 1)
    {
        nppiResize_8u_C1R((const Npp8u*)src.ptr<char>(), srcsz, src.step, srcrect,
            (Npp8u*)dst.ptr<char>(), dst.step, dstsz, fx, fy, npp_inter[interpolation]);
    }
    else
    {
        nppiResize_8u_C4R((const Npp8u*)src.ptr<char>(), srcsz, src.step, srcrect,
            (Npp8u*)dst.ptr<char>(), dst.step, dstsz, fx, fy, npp_inter[interpolation]);
    }
}

Scalar cv::gpu::sum(const GpuMat& src)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4);
    
    Scalar res;

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    if (src.channels() == 1)
    {
        nppiSum_8u_C1R((const Npp8u*)src.ptr<char>(), src.step, sz, res.val);
    }
    else
    {
        nppiSum_8u_C4R((const Npp8u*)src.ptr<char>(), src.step, sz, res.val);
    }

    return res;
}

void cv::gpu::minMax(const GpuMat& src, double* minVal, double* maxVal) 
{
    CV_Assert(src.type() == CV_8UC1);

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    Npp8u min_res, max_res;

    nppiMinMax_8u_C1R((const Npp8u*)src.ptr<char>(), src.step, sz, &min_res, &max_res);

    if (minVal)
        *minVal = min_res;

    if (maxVal)
        *maxVal = max_res;
}

void cv::gpu::copyConstBorder(const GpuMat& src, GpuMat& dst, int top, int bottom, int left, int right, const Scalar& value) 
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4 || src.type() == CV_32SC1);

    dst.create(src.rows + top + bottom, src.cols + left + right, src.type());

	NppiSize srcsz;
	srcsz.width  = src.cols;
	srcsz.height = src.rows;
    NppiSize dstsz;
	dstsz.width  = dst.cols;
	dstsz.height = dst.rows;

	if (src.depth() == CV_8U)
	{
		if (src.channels() == 1)
		{
            Npp8u nVal = (Npp8u)value[0];
            nppiCopyConstBorder_8u_C1R((const Npp8u*)src.ptr<char>(), src.step, srcsz, 
                (Npp8u*)dst.ptr<char>(), dst.step, dstsz, top, left, nVal);
		}
		else
		{
            Npp8u nVal[] = {(Npp8u)value[0], (Npp8u)value[1], (Npp8u)value[2], (Npp8u)value[3]};
            nppiCopyConstBorder_8u_C4R((const Npp8u*)src.ptr<char>(), src.step, srcsz, 
                (Npp8u*)dst.ptr<char>(), dst.step, dstsz, top, left, nVal);
		}        
	}
	else //if (src.depth() == CV_32S)
	{
        Npp32s nVal = (Npp32s)value[0];
        nppiCopyConstBorder_32s_C1R((const Npp32s*)src.ptr<char>(), src.step, srcsz, 
            (Npp32s*)dst.ptr<char>(), dst.step, dstsz, top, left, nVal);
	}
}

#endif /* !defined (HAVE_CUDA) */