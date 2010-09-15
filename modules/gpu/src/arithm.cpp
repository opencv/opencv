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

void cv::gpu::add(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::subtract(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::multiply(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::divide(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }

void cv::gpu::transpose(const GpuMat&, GpuMat&) { throw_nogpu(); }

void cv::gpu::absdiff(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }

double cv::gpu::threshold(const GpuMat&, GpuMat&, double, double, int) { throw_nogpu(); return 0.0; }

void cv::gpu::compare(const GpuMat&, const GpuMat&, GpuMat&, int) { throw_nogpu(); }

void cv::gpu::meanStdDev(const GpuMat&, Scalar&, Scalar&) { throw_nogpu(); }

double cv::gpu::norm(const GpuMat&, int) { throw_nogpu(); return 0.0; }
double cv::gpu::norm(const GpuMat&, const GpuMat&, int) { throw_nogpu(); return 0.0; }

void cv::gpu::flip(const GpuMat&, GpuMat&, int) { throw_nogpu(); }

void cv::gpu::resize(const GpuMat&, GpuMat&, Size, double, double, int) { throw_nogpu(); }

Scalar cv::gpu::sum(const GpuMat&) { throw_nogpu(); return Scalar(); }

void cv::gpu::minMax(const GpuMat&, double*, double*) { throw_nogpu(); }

void cv::gpu::copyMakeBorder(const GpuMat&, GpuMat&, int, int, int, int, const Scalar&) { throw_nogpu(); }

void cv::gpu::warpAffine(const GpuMat&, GpuMat&, const Mat&, Size, int) { throw_nogpu(); }

void cv::gpu::warpPerspective(const GpuMat&, GpuMat&, const Mat&, Size, int) { throw_nogpu(); }

void cv::gpu::rotate(const GpuMat&, GpuMat&, Size, double, double, double, int) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace
{    
    typedef NppStatus (*npp_warp_8u_t)(const Npp8u* pSrc, NppiSize srcSize, int srcStep, NppiRect srcRoi, Npp8u* pDst, 
                                       int dstStep, NppiRect dstRoi, const double coeffs[][3], 
                                       int interpolation);
    typedef NppStatus (*npp_warp_16u_t)(const Npp16u* pSrc, NppiSize srcSize, int srcStep, NppiRect srcRoi, Npp16u* pDst, 
                                       int dstStep, NppiRect dstRoi, const double coeffs[][3], 
                                       int interpolation);
    typedef NppStatus (*npp_warp_32s_t)(const Npp32s* pSrc, NppiSize srcSize, int srcStep, NppiRect srcRoi, Npp32s* pDst, 
                                       int dstStep, NppiRect dstRoi, const double coeffs[][3], 
                                       int interpolation);
    typedef NppStatus (*npp_warp_32f_t)(const Npp32f* pSrc, NppiSize srcSize, int srcStep, NppiRect srcRoi, Npp32f* pDst, 
                                       int dstStep, NppiRect dstRoi, const double coeffs[][3], 
                                       int interpolation);

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
				nppSafeCall( npp_func_8uc1((const Npp8u*)src1.ptr<char>(), src1.step, 
					(const Npp8u*)src2.ptr<char>(), src2.step, 
					(Npp8u*)dst.ptr<char>(), dst.step, sz, 0) );
			}
			else
			{
				nppSafeCall( npp_func_8uc4((const Npp8u*)src1.ptr<char>(), src1.step, 
					(const Npp8u*)src2.ptr<char>(), src2.step, 
					(Npp8u*)dst.ptr<char>(), dst.step, sz, 0) );
			}        
		}
		else //if (src1.depth() == CV_32F)
		{
			nppSafeCall( npp_func_32fc1((const Npp32f*)src1.ptr<float>(), src1.step,
				(const Npp32f*)src2.ptr<float>(), src2.step,
				(Npp32f*)dst.ptr<float>(), dst.step, sz) );
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

    nppSafeCall( nppiTranspose_8u_C1R((const Npp8u*)src.ptr<char>(), src.step, (Npp8u*)dst.ptr<char>(), dst.step, sz) );
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
        nppSafeCall( nppiAbsDiff_8u_C1R((const Npp8u*)src1.ptr<char>(), src1.step, 
                (const Npp8u*)src2.ptr<char>(), src2.step, 
                (Npp8u*)dst.ptr<char>(), dst.step, sz) );
    }
    else //if (src1.depth() == CV_32F)
    {
        nppSafeCall( nppiAbsDiff_32f_C1R((const Npp32f*)src1.ptr<float>(), src1.step,
            (const Npp32f*)src2.ptr<float>(), src2.step,
            (Npp32f*)dst.ptr<float>(), dst.step, sz) );
    }
}

double cv::gpu::threshold(const GpuMat& src, GpuMat& dst, double thresh, double /*maxVal*/, int thresholdType) 
{ 
    CV_Assert(src.type() == CV_32FC1 && thresholdType == THRESH_TRUNC);

    dst.create( src.size(), src.type() );

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    nppSafeCall( nppiThreshold_32f_C1R((const Npp32f*)src.ptr<float>(), src.step, 
        (Npp32f*)dst.ptr<float>(), dst.step, sz, (Npp32f)thresh, NPP_CMP_GREATER) );

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
        nppSafeCall( nppiCompare_8u_C4R((const Npp8u*)src1.ptr<char>(), src1.step, 
            (const Npp8u*)src2.ptr<char>(), src2.step, 
            (Npp8u*)dst.ptr<char>(), dst.step, sz, nppCmpOp[cmpop]) );
    }
    else //if (src1.depth() == CV_32F)
    {
        nppSafeCall( nppiCompare_32f_C1R((const Npp32f*)src1.ptr<float>(), src1.step,
            (const Npp32f*)src2.ptr<float>(), src2.step,
            (Npp8u*)dst.ptr<char>(), dst.step, sz, nppCmpOp[cmpop]) );
    }
}

void cv::gpu::meanStdDev(const GpuMat& src, Scalar& mean, Scalar& stddev) 
{
    CV_Assert(src.type() == CV_8UC1);

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    nppSafeCall( nppiMean_StdDev_8u_C1R((const Npp8u*)src.ptr<char>(), src.step, sz, mean.val, stddev.val) );
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

    nppSafeCall( npp_norm_diff_func[funcIdx]((const Npp8u*)src1.ptr<char>(), src1.step, 
        (const Npp8u*)src2.ptr<char>(), src2.step, 
        sz, retVal.val) );

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
        nppSafeCall( nppiMirror_8u_C1R((const Npp8u*)src.ptr<char>(), src.step, 
            (Npp8u*)dst.ptr<char>(), dst.step, sz, 
            (flipCode == 0 ? NPP_HORIZONTAL_AXIS : (flipCode > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS))) );
    }
    else
    {
        nppSafeCall( nppiMirror_8u_C4R((const Npp8u*)src.ptr<char>(), src.step, 
            (Npp8u*)dst.ptr<char>(), dst.step, sz, 
            (flipCode == 0 ? NPP_HORIZONTAL_AXIS : (flipCode > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS))) );
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
        nppSafeCall( nppiResize_8u_C1R((const Npp8u*)src.ptr<char>(), srcsz, src.step, srcrect,
            (Npp8u*)dst.ptr<char>(), dst.step, dstsz, fx, fy, npp_inter[interpolation]) );
    }
    else
    {
        nppSafeCall( nppiResize_8u_C4R((const Npp8u*)src.ptr<char>(), srcsz, src.step, srcrect,
            (Npp8u*)dst.ptr<char>(), dst.step, dstsz, fx, fy, npp_inter[interpolation]) );
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
        nppSafeCall( nppiSum_8u_C1R((const Npp8u*)src.ptr<char>(), src.step, sz, res.val) );
    }
    else
    {
        nppSafeCall( nppiSum_8u_C4R((const Npp8u*)src.ptr<char>(), src.step, sz, res.val) );
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

    nppSafeCall( nppiMinMax_8u_C1R((const Npp8u*)src.ptr<char>(), src.step, sz, &min_res, &max_res) );

    if (minVal)
        *minVal = min_res;

    if (maxVal)
        *maxVal = max_res;
}

void cv::gpu::copyMakeBorder(const GpuMat& src, GpuMat& dst, int top, int bottom, int left, int right, const Scalar& value) 
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
            nppSafeCall( nppiCopyConstBorder_8u_C1R((const Npp8u*)src.ptr<char>(), src.step, srcsz, 
                (Npp8u*)dst.ptr<char>(), dst.step, dstsz, top, left, nVal) );
		}
		else
		{
            Npp8u nVal[] = {(Npp8u)value[0], (Npp8u)value[1], (Npp8u)value[2], (Npp8u)value[3]};
            nppSafeCall( nppiCopyConstBorder_8u_C4R((const Npp8u*)src.ptr<char>(), src.step, srcsz, 
                (Npp8u*)dst.ptr<char>(), dst.step, dstsz, top, left, nVal) );
		}        
	}
	else //if (src.depth() == CV_32S)
	{
        Npp32s nVal = (Npp32s)value[0];
        nppSafeCall( nppiCopyConstBorder_32s_C1R((const Npp32s*)src.ptr<char>(), src.step, srcsz, 
            (Npp32s*)dst.ptr<char>(), dst.step, dstsz, top, left, nVal) );
	}
}

namespace
{
    void nppWarpCaller(const GpuMat& src, GpuMat& dst, double coeffs[][3], const Size& dsize, int flags, 
                       npp_warp_8u_t npp_warp_8u[][2], npp_warp_16u_t npp_warp_16u[][2], 
                       npp_warp_32s_t npp_warp_32s[][2], npp_warp_32f_t npp_warp_32f[][2]) 
    {
        static const int npp_inter[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};
    
        int interpolation = flags & INTER_MAX;

        CV_Assert((src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32S || src.depth() == CV_32F) && src.channels() != 2);
        CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC);

        dst.create(dsize, src.type());

        NppiSize srcsz;
        srcsz.height = src.rows;
        srcsz.width = src.cols;
        NppiRect srcroi;
        srcroi.x = srcroi.y = 0;
        srcroi.height = src.rows;
        srcroi.width = src.cols;
        NppiRect dstroi;
        dstroi.x = dstroi.y = 0;
        dstroi.height = dst.rows;
        dstroi.width = dst.cols;

        int warpInd = (flags & WARP_INVERSE_MAP) >> 4;

        switch (src.depth())
        {
        case CV_8U:
            nppSafeCall( npp_warp_8u[src.channels()][warpInd]((const Npp8u*)src.ptr<char>(), srcsz, src.step, srcroi, 
                (Npp8u*)dst.ptr<char>(), dst.step, dstroi, coeffs, npp_inter[interpolation]) );
            break;
        case CV_16U:
            nppSafeCall( npp_warp_16u[src.channels()][warpInd]((const Npp16u*)src.ptr<char>(), srcsz, src.step, srcroi, 
                (Npp16u*)dst.ptr<char>(), dst.step, dstroi, coeffs, npp_inter[interpolation]) );
            break;
        case CV_32SC1:
            nppSafeCall( npp_warp_32s[src.channels()][warpInd]((const Npp32s*)src.ptr<char>(), srcsz, src.step, srcroi, 
                (Npp32s*)dst.ptr<char>(), dst.step, dstroi, coeffs, npp_inter[interpolation]) );
            break;
        case CV_32FC1:
            nppSafeCall( npp_warp_32f[src.channels()][warpInd]((const Npp32f*)src.ptr<char>(), srcsz, src.step, srcroi, 
                (Npp32f*)dst.ptr<char>(), dst.step, dstroi, coeffs, npp_inter[interpolation]) );
            break;
        default:
            CV_Assert(!"Unsupported source type");
        }
    }
}

void cv::gpu::warpAffine(const GpuMat& src, GpuMat& dst, const Mat& M, Size dsize, int flags) 
{
    static npp_warp_8u_t npp_warpAffine_8u[][2] = 
        {
            {0, 0}, 
            {nppiWarpAffine_8u_C1R, nppiWarpAffineBack_8u_C1R}, 
            {0, 0}, 
            {nppiWarpAffine_8u_C3R, nppiWarpAffineBack_8u_C3R}, 
            {nppiWarpAffine_8u_C4R, nppiWarpAffineBack_8u_C4R}
        };
    static npp_warp_16u_t npp_warpAffine_16u[][2] = 
        {
            {0, 0}, 
            {nppiWarpAffine_16u_C1R, nppiWarpAffineBack_16u_C1R}, 
            {0, 0}, 
            {nppiWarpAffine_16u_C3R, nppiWarpAffineBack_16u_C3R}, 
            {nppiWarpAffine_16u_C4R, nppiWarpAffineBack_16u_C4R}
        };
    static npp_warp_32s_t npp_warpAffine_32s[][2] = 
        {
            {0, 0}, 
            {nppiWarpAffine_32s_C1R, nppiWarpAffineBack_32s_C1R}, 
            {0, 0}, 
            {nppiWarpAffine_32s_C3R, nppiWarpAffineBack_32s_C3R}, 
            {nppiWarpAffine_32s_C4R, nppiWarpAffineBack_32s_C4R}
        };
    static npp_warp_32f_t npp_warpAffine_32f[][2] = 
        {
            {0, 0}, 
            {nppiWarpAffine_32f_C1R, nppiWarpAffineBack_32f_C1R}, 
            {0, 0}, 
            {nppiWarpAffine_32f_C3R, nppiWarpAffineBack_32f_C3R}, 
            {nppiWarpAffine_32f_C4R, nppiWarpAffineBack_32f_C4R}
        };

    CV_Assert(M.rows == 2 && M.cols == 3);

    double coeffs[2][3];
    Mat coeffsMat(2, 3, CV_64F, (void*)coeffs);
    M.convertTo(coeffsMat, coeffsMat.type());

    nppWarpCaller(src, dst, coeffs, dsize, flags, npp_warpAffine_8u, npp_warpAffine_16u, npp_warpAffine_32s, npp_warpAffine_32f);
}

void cv::gpu::warpPerspective(const GpuMat& src, GpuMat& dst, const Mat& M, Size dsize, int flags)
{
    static npp_warp_8u_t npp_warpPerspective_8u[][2] = 
        {
            {0, 0}, 
            {nppiWarpPerspective_8u_C1R, nppiWarpPerspectiveBack_8u_C1R}, 
            {0, 0}, 
            {nppiWarpPerspective_8u_C3R, nppiWarpPerspectiveBack_8u_C3R}, 
            {nppiWarpPerspective_8u_C4R, nppiWarpPerspectiveBack_8u_C4R}
        };
    static npp_warp_16u_t npp_warpPerspective_16u[][2] = 
        {
            {0, 0}, 
            {nppiWarpPerspective_16u_C1R, nppiWarpPerspectiveBack_16u_C1R}, 
            {0, 0}, 
            {nppiWarpPerspective_16u_C3R, nppiWarpPerspectiveBack_16u_C3R}, 
            {nppiWarpPerspective_16u_C4R, nppiWarpPerspectiveBack_16u_C4R}
        };
    static npp_warp_32s_t npp_warpPerspective_32s[][2] = 
        {
            {0, 0}, 
            {nppiWarpPerspective_32s_C1R, nppiWarpPerspectiveBack_32s_C1R}, 
            {0, 0}, 
            {nppiWarpPerspective_32s_C3R, nppiWarpPerspectiveBack_32s_C3R}, 
            {nppiWarpPerspective_32s_C4R, nppiWarpPerspectiveBack_32s_C4R}
        };
    static npp_warp_32f_t npp_warpPerspective_32f[][2] = 
        {
            {0, 0}, 
            {nppiWarpPerspective_32f_C1R, nppiWarpPerspectiveBack_32f_C1R}, 
            {0, 0}, 
            {nppiWarpPerspective_32f_C3R, nppiWarpPerspectiveBack_32f_C3R}, 
            {nppiWarpPerspective_32f_C4R, nppiWarpPerspectiveBack_32f_C4R}
        };

    CV_Assert(M.rows == 3 && M.cols == 3);

    double coeffs[3][3];
    Mat coeffsMat(3, 3, CV_64F, (void*)coeffs);
    M.convertTo(coeffsMat, coeffsMat.type());

    nppWarpCaller(src, dst, coeffs, dsize, flags, npp_warpPerspective_8u, npp_warpPerspective_16u, npp_warpPerspective_32s, npp_warpPerspective_32f);
}

void cv::gpu::rotate(const GpuMat& src, GpuMat& dst, Size dsize, double angle, double xShift, double yShift, int interpolation)
{
    static const int npp_inter[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};
    
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4);
    CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC);

    dst.create(dsize, src.type());

    NppiSize srcsz;
    srcsz.height = src.rows;
    srcsz.width = src.cols;
    NppiRect srcroi;
    srcroi.x = srcroi.y = 0;
    srcroi.height = src.rows;
    srcroi.width = src.cols;
    NppiRect dstroi;
    dstroi.x = dstroi.y = 0;
    dstroi.height = dst.rows;
    dstroi.width = dst.cols;

    if (src.channels() == 1)
    {
        nppSafeCall( nppiRotate_8u_C1R((const Npp8u*)src.ptr<char>(), srcsz, src.step, srcroi, 
            (Npp8u*)dst.ptr<char>(), dst.step, dstroi, angle, xShift, yShift, npp_inter[interpolation]) );
    }
    else
    {
        nppSafeCall( nppiRotate_8u_C4R((const Npp8u*)src.ptr<char>(), srcsz, src.step, srcroi, 
            (Npp8u*)dst.ptr<char>(), dst.step, dstroi, angle, xShift, yShift, npp_inter[interpolation]) );
    }
}

#endif /* !defined (HAVE_CUDA) */