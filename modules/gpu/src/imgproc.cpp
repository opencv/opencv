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
using namespace cv::gpu;

#if !defined (HAVE_CUDA)

void cv::gpu::meanShiftFiltering(const GpuMat&, GpuMat&, int, int, TermCriteria, Stream&) { throw_nogpu(); }
void cv::gpu::meanShiftProc(const GpuMat&, GpuMat&, GpuMat&, int, int, TermCriteria, Stream&) { throw_nogpu(); }
void cv::gpu::drawColorDisp(const GpuMat&, GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::reprojectImageTo3D(const GpuMat&, GpuMat&, const Mat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::copyMakeBorder(const GpuMat&, GpuMat&, int, int, int, int, int, const Scalar&, Stream&) { throw_nogpu(); }
void cv::gpu::buildWarpPlaneMaps(Size, Rect, const Mat&, const Mat&, const Mat&, float, GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::buildWarpCylindricalMaps(Size, Rect, const Mat&, const Mat&, float, GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::buildWarpSphericalMaps(Size, Rect, const Mat&, const Mat&, float, GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::rotate(const GpuMat&, GpuMat&, Size, double, double, double, int, Stream&) { throw_nogpu(); }
void cv::gpu::integral(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::integralBuffered(const GpuMat&, GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::sqrIntegral(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::columnSum(const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::rectStdDev(const GpuMat&, const GpuMat&, GpuMat&, const Rect&, Stream&) { throw_nogpu(); }
void cv::gpu::evenLevels(GpuMat&, int, int, int) { throw_nogpu(); }
void cv::gpu::histEven(const GpuMat&, GpuMat&, int, int, int, Stream&) { throw_nogpu(); }
void cv::gpu::histEven(const GpuMat&, GpuMat&, GpuMat&, int, int, int, Stream&) { throw_nogpu(); }
void cv::gpu::histEven(const GpuMat&, GpuMat*, int*, int*, int*, Stream&) { throw_nogpu(); }
void cv::gpu::histEven(const GpuMat&, GpuMat*, GpuMat&, int*, int*, int*, Stream&) { throw_nogpu(); }
void cv::gpu::histRange(const GpuMat&, GpuMat&, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::histRange(const GpuMat&, GpuMat&, const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::histRange(const GpuMat&, GpuMat*, const GpuMat*, Stream&) { throw_nogpu(); }
void cv::gpu::histRange(const GpuMat&, GpuMat*, const GpuMat*, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::calcHist(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::calcHist(const GpuMat&, GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::equalizeHist(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::equalizeHist(const GpuMat&, GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::equalizeHist(const GpuMat&, GpuMat&, GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::cornerHarris(const GpuMat&, GpuMat&, int, int, double, int) { throw_nogpu(); }
void cv::gpu::cornerHarris(const GpuMat&, GpuMat&, GpuMat&, GpuMat&, int, int, double, int) { throw_nogpu(); }
void cv::gpu::cornerHarris(const GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, int, int, double, int, Stream&) { throw_nogpu(); }
void cv::gpu::cornerMinEigenVal(const GpuMat&, GpuMat&, int, int, int) { throw_nogpu(); }
void cv::gpu::cornerMinEigenVal(const GpuMat&, GpuMat&, GpuMat&, GpuMat&, int, int, int) { throw_nogpu(); }
void cv::gpu::cornerMinEigenVal(const GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, int, int, int, Stream&) { throw_nogpu(); }
void cv::gpu::mulSpectrums(const GpuMat&, const GpuMat&, GpuMat&, int, bool, Stream&) { throw_nogpu(); }
void cv::gpu::mulAndScaleSpectrums(const GpuMat&, const GpuMat&, GpuMat&, int, float, bool, Stream&) { throw_nogpu(); }
void cv::gpu::dft(const GpuMat&, GpuMat&, Size, int, Stream&) { throw_nogpu(); }
void cv::gpu::ConvolveBuf::create(Size, Size) { throw_nogpu(); }
void cv::gpu::convolve(const GpuMat&, const GpuMat&, GpuMat&, bool) { throw_nogpu(); }
void cv::gpu::convolve(const GpuMat&, const GpuMat&, GpuMat&, bool, ConvolveBuf&, Stream&) { throw_nogpu(); }
void cv::gpu::Canny(const GpuMat&, GpuMat&, double, double, int, bool) { throw_nogpu(); }
void cv::gpu::Canny(const GpuMat&, CannyBuf&, GpuMat&, double, double, int, bool) { throw_nogpu(); }
void cv::gpu::Canny(const GpuMat&, const GpuMat&, GpuMat&, double, double, bool) { throw_nogpu(); }
void cv::gpu::Canny(const GpuMat&, const GpuMat&, CannyBuf&, GpuMat&, double, double, bool) { throw_nogpu(); }
cv::gpu::CannyBuf::CannyBuf(const GpuMat&, const GpuMat&) { throw_nogpu(); }
void cv::gpu::CannyBuf::create(const Size&, int) { throw_nogpu(); }
void cv::gpu::CannyBuf::release() { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

////////////////////////////////////////////////////////////////////////
// meanShiftFiltering_GPU

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        void meanShiftFiltering_gpu(const DevMem2Db& src, DevMem2Db dst, int sp, int sr, int maxIter, float eps, cudaStream_t stream);
    }
}}}

void cv::gpu::meanShiftFiltering(const GpuMat& src, GpuMat& dst, int sp, int sr, TermCriteria criteria, Stream& stream)
{
    using namespace ::cv::gpu::device::imgproc;

    if( src.empty() )
        CV_Error( CV_StsBadArg, "The input image is empty" );

    if( src.depth() != CV_8U || src.channels() != 4 )
        CV_Error( CV_StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported" );

    dst.create( src.size(), CV_8UC4 );

    if( !(criteria.type & TermCriteria::MAX_ITER) )
        criteria.maxCount = 5;

    int maxIter = std::min(std::max(criteria.maxCount, 1), 100);

    float eps;
    if( !(criteria.type & TermCriteria::EPS) )
        eps = 1.f;
    eps = (float)std::max(criteria.epsilon, 0.0);

    meanShiftFiltering_gpu(src, dst, sp, sr, maxIter, eps, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// meanShiftProc_GPU

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        void meanShiftProc_gpu(const DevMem2Db& src, DevMem2Db dstr, DevMem2Db dstsp, int sp, int sr, int maxIter, float eps, cudaStream_t stream);
    }
}}}

void cv::gpu::meanShiftProc(const GpuMat& src, GpuMat& dstr, GpuMat& dstsp, int sp, int sr, TermCriteria criteria, Stream& stream)
{
    using namespace ::cv::gpu::device::imgproc;

    if( src.empty() )
        CV_Error( CV_StsBadArg, "The input image is empty" );

    if( src.depth() != CV_8U || src.channels() != 4 )
        CV_Error( CV_StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported" );

    dstr.create( src.size(), CV_8UC4 );
    dstsp.create( src.size(), CV_16SC2 );

    if( !(criteria.type & TermCriteria::MAX_ITER) )
        criteria.maxCount = 5;

    int maxIter = std::min(std::max(criteria.maxCount, 1), 100);

    float eps;
    if( !(criteria.type & TermCriteria::EPS) )
        eps = 1.f;
    eps = (float)std::max(criteria.epsilon, 0.0);

    meanShiftProc_gpu(src, dstr, dstsp, sp, sr, maxIter, eps, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// drawColorDisp

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        void drawColorDisp_gpu(const DevMem2Db& src, const DevMem2Db& dst, int ndisp, const cudaStream_t& stream);
        void drawColorDisp_gpu(const DevMem2D_<short>& src, const DevMem2Db& dst, int ndisp, const cudaStream_t& stream);
    }
}}}

namespace
{
    template <typename T>
    void drawColorDisp_caller(const GpuMat& src, GpuMat& dst, int ndisp, const cudaStream_t& stream)
    {
        using namespace ::cv::gpu::device::imgproc;

        dst.create(src.size(), CV_8UC4);

        drawColorDisp_gpu((DevMem2D_<T>)src, dst, ndisp, stream);
    }

    typedef void (*drawColorDisp_caller_t)(const GpuMat& src, GpuMat& dst, int ndisp, const cudaStream_t& stream);

    const drawColorDisp_caller_t drawColorDisp_callers[] = {drawColorDisp_caller<unsigned char>, 0, 0, drawColorDisp_caller<short>, 0, 0, 0, 0};
}

void cv::gpu::drawColorDisp(const GpuMat& src, GpuMat& dst, int ndisp, Stream& stream)
{
    CV_Assert(src.type() == CV_8U || src.type() == CV_16S);

    drawColorDisp_callers[src.type()](src, dst, ndisp, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// reprojectImageTo3D

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        template <typename T, typename D>
        void reprojectImageTo3D_gpu(const DevMem2Db disp, DevMem2Db xyz, const float* q, cudaStream_t stream);
    }
}}}

void cv::gpu::reprojectImageTo3D(const GpuMat& disp, GpuMat& xyz, const Mat& Q, int dst_cn, Stream& stream)
{
    using namespace cv::gpu::device::imgproc;

    typedef void (*func_t)(const DevMem2Db disp, DevMem2Db xyz, const float* q, cudaStream_t stream);
    static const func_t funcs[2][4] = 
    {
        {reprojectImageTo3D_gpu<uchar, float3>, 0, 0, reprojectImageTo3D_gpu<short, float3>},
        {reprojectImageTo3D_gpu<uchar, float4>, 0, 0, reprojectImageTo3D_gpu<short, float4>}
    };

    CV_Assert(disp.type() == CV_8U || disp.type() == CV_16S);
    CV_Assert(Q.type() == CV_32F && Q.rows == 4 && Q.cols == 4 && Q.isContinuous());
    CV_Assert(dst_cn == 3 || dst_cn == 4);

    xyz.create(disp.size(), CV_MAKE_TYPE(CV_32F, dst_cn));

    funcs[dst_cn == 4][disp.type()](disp, xyz, Q.ptr<float>(), StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// copyMakeBorder

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        template <typename T, int cn> void copyMakeBorder_gpu(const DevMem2Db& src, const DevMem2Db& dst, int top, int left, int borderMode, const T* borderValue, cudaStream_t stream);
    }
}}}

namespace
{
    template <typename T, int cn> void copyMakeBorder_caller(const DevMem2Db& src, const DevMem2Db& dst, int top, int left, int borderType, const Scalar& value, cudaStream_t stream)
    {
        using namespace ::cv::gpu::device::imgproc;

        Scalar_<T> val(saturate_cast<T>(value[0]), saturate_cast<T>(value[1]), saturate_cast<T>(value[2]), saturate_cast<T>(value[3]));

        copyMakeBorder_gpu<T, cn>(src, dst, top, left, borderType, val.val, stream);
    }
}

void cv::gpu::copyMakeBorder(const GpuMat& src, GpuMat& dst, int top, int bottom, int left, int right, int borderType, const Scalar& value, Stream& s)
{
    CV_Assert(src.depth() <= CV_32F && src.channels() <= 4);
    CV_Assert(borderType == BORDER_REFLECT101 || borderType == BORDER_REPLICATE || borderType == BORDER_CONSTANT || borderType == BORDER_REFLECT || borderType == BORDER_WRAP);

    dst.create(src.rows + top + bottom, src.cols + left + right, src.type());

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (borderType == BORDER_CONSTANT && (src.type() == CV_8UC1 || src.type() == CV_8UC4 || src.type() == CV_32SC1 || src.type() == CV_32FC1))
    {
        NppiSize srcsz;
        srcsz.width  = src.cols;
        srcsz.height = src.rows;

        NppiSize dstsz;
        dstsz.width  = dst.cols;
        dstsz.height = dst.rows;

        NppStreamHandler h(stream);

        switch (src.type())
        {
        case CV_8UC1:
            {
                Npp8u nVal = saturate_cast<Npp8u>(value[0]);
                nppSafeCall( nppiCopyConstBorder_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step), srcsz,
                    dst.ptr<Npp8u>(), static_cast<int>(dst.step), dstsz, top, left, nVal) );
                break;
            }
        case CV_8UC4:
            {
                Npp8u nVal[] = {saturate_cast<Npp8u>(value[0]), saturate_cast<Npp8u>(value[1]), saturate_cast<Npp8u>(value[2]), saturate_cast<Npp8u>(value[3])};
                nppSafeCall( nppiCopyConstBorder_8u_C4R(src.ptr<Npp8u>(), static_cast<int>(src.step), srcsz,
                    dst.ptr<Npp8u>(), static_cast<int>(dst.step), dstsz, top, left, nVal) );
                break;
            }
        case CV_32SC1:
            {
                Npp32s nVal = saturate_cast<Npp32s>(value[0]);
                nppSafeCall( nppiCopyConstBorder_32s_C1R(src.ptr<Npp32s>(), static_cast<int>(src.step), srcsz,
                    dst.ptr<Npp32s>(), static_cast<int>(dst.step), dstsz, top, left, nVal) );
                break;
            }
        case CV_32FC1:
            {
                Npp32f val = saturate_cast<Npp32f>(value[0]);
                Npp32s nVal = *(reinterpret_cast<Npp32s*>(&val));
                nppSafeCall( nppiCopyConstBorder_32s_C1R(src.ptr<Npp32s>(), static_cast<int>(src.step), srcsz,
                    dst.ptr<Npp32s>(), static_cast<int>(dst.step), dstsz, top, left, nVal) );
                break;
            }
        }

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    else
    {
        typedef void (*caller_t)(const DevMem2Db& src, const DevMem2Db& dst, int top, int left, int borderType, const Scalar& value, cudaStream_t stream);
        static const caller_t callers[6][4] =
        {
            {   copyMakeBorder_caller<uchar, 1>  , 0/*copyMakeBorder_caller<uchar, 2>*/ ,    copyMakeBorder_caller<uchar, 3>  ,    copyMakeBorder_caller<uchar, 4>},
            {0/*copyMakeBorder_caller<schar, 1>*/, 0/*copyMakeBorder_caller<schar, 2>*/ , 0/*copyMakeBorder_caller<schar, 3>*/, 0/*copyMakeBorder_caller<schar, 4>*/},
            {   copyMakeBorder_caller<ushort, 1> , 0/*copyMakeBorder_caller<ushort, 2>*/,    copyMakeBorder_caller<ushort, 3> ,    copyMakeBorder_caller<ushort, 4>},
            {   copyMakeBorder_caller<short, 1>  , 0/*copyMakeBorder_caller<short, 2>*/ ,    copyMakeBorder_caller<short, 3>  ,    copyMakeBorder_caller<short, 4>},
            {0/*copyMakeBorder_caller<int, 1>*/  , 0/*copyMakeBorder_caller<int, 2>*/   , 0/*copyMakeBorder_caller<int, 3>*/  , 0/*copyMakeBorder_caller<int, 4>*/},
            {   copyMakeBorder_caller<float, 1>  , 0/*copyMakeBorder_caller<float, 2>*/ ,    copyMakeBorder_caller<float, 3>  ,    copyMakeBorder_caller<float ,4>}
        };

        caller_t func = callers[src.depth()][src.channels() - 1];
        CV_Assert(func != 0);

        int gpuBorderType;
        CV_Assert(tryConvertToGpuBorderType(borderType, gpuBorderType));

        func(src, dst, top, left, gpuBorderType, value, stream);
    }
}

//////////////////////////////////////////////////////////////////////////////
// buildWarpPlaneMaps

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        void buildWarpPlaneMaps(int tl_u, int tl_v, DevMem2Df map_x, DevMem2Df map_y,
                                const float k_rinv[9], const float r_kinv[9], const float t[3], float scale,
                                cudaStream_t stream);
    }
}}}

void cv::gpu::buildWarpPlaneMaps(Size src_size, Rect dst_roi, const Mat &K, const Mat& R, const Mat &T,
                                 float scale, GpuMat& map_x, GpuMat& map_y, Stream& stream)
{
    using namespace ::cv::gpu::device::imgproc;

    CV_Assert(K.size() == Size(3,3) && K.type() == CV_32F);
    CV_Assert(R.size() == Size(3,3) && R.type() == CV_32F);
    CV_Assert((T.size() == Size(3,1) || T.size() == Size(1,3)) && T.type() == CV_32F && T.isContinuous());

    Mat K_Rinv = K * R.t();
    Mat R_Kinv = R * K.inv();
    CV_Assert(K_Rinv.isContinuous());
    CV_Assert(R_Kinv.isContinuous());

    map_x.create(dst_roi.size(), CV_32F);
    map_y.create(dst_roi.size(), CV_32F);
    device::imgproc::buildWarpPlaneMaps(dst_roi.tl().x, dst_roi.tl().y, map_x, map_y, K_Rinv.ptr<float>(), R_Kinv.ptr<float>(),
                       T.ptr<float>(), scale, StreamAccessor::getStream(stream));
}

//////////////////////////////////////////////////////////////////////////////
// buildWarpCylyndricalMaps

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        void buildWarpCylindricalMaps(int tl_u, int tl_v, DevMem2Df map_x, DevMem2Df map_y,
                                      const float k_rinv[9], const float r_kinv[9], float scale,
                                      cudaStream_t stream);
    }
}}}

void cv::gpu::buildWarpCylindricalMaps(Size src_size, Rect dst_roi, const Mat &K, const Mat& R, float scale,
                                       GpuMat& map_x, GpuMat& map_y, Stream& stream)
{
    using namespace ::cv::gpu::device::imgproc;

    CV_Assert(K.size() == Size(3,3) && K.type() == CV_32F);
    CV_Assert(R.size() == Size(3,3) && R.type() == CV_32F);

    Mat K_Rinv = K * R.t();
    Mat R_Kinv = R * K.inv();
    CV_Assert(K_Rinv.isContinuous());
    CV_Assert(R_Kinv.isContinuous());

    map_x.create(dst_roi.size(), CV_32F);
    map_y.create(dst_roi.size(), CV_32F);
    device::imgproc::buildWarpCylindricalMaps(dst_roi.tl().x, dst_roi.tl().y, map_x, map_y, K_Rinv.ptr<float>(), R_Kinv.ptr<float>(), scale, StreamAccessor::getStream(stream));
}


//////////////////////////////////////////////////////////////////////////////
// buildWarpSphericalMaps

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        void buildWarpSphericalMaps(int tl_u, int tl_v, DevMem2Df map_x, DevMem2Df map_y,
                                    const float k_rinv[9], const float r_kinv[9], float scale,
                                    cudaStream_t stream);
    }
}}}

void cv::gpu::buildWarpSphericalMaps(Size src_size, Rect dst_roi, const Mat &K, const Mat& R, float scale,
                                     GpuMat& map_x, GpuMat& map_y, Stream& stream)
{
    using namespace ::cv::gpu::device::imgproc;

    CV_Assert(K.size() == Size(3,3) && K.type() == CV_32F);
    CV_Assert(R.size() == Size(3,3) && R.type() == CV_32F);

    Mat K_Rinv = K * R.t();
    Mat R_Kinv = R * K.inv();
    CV_Assert(K_Rinv.isContinuous());
    CV_Assert(R_Kinv.isContinuous());

    map_x.create(dst_roi.size(), CV_32F);
    map_y.create(dst_roi.size(), CV_32F);
    device::imgproc::buildWarpSphericalMaps(dst_roi.tl().x, dst_roi.tl().y, map_x, map_y, K_Rinv.ptr<float>(), R_Kinv.ptr<float>(), scale, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// rotate

namespace
{
    template<int DEPTH> struct NppTypeTraits;
    template<> struct NppTypeTraits<CV_8U>  { typedef Npp8u npp_t; };
    template<> struct NppTypeTraits<CV_8S>  { typedef Npp8s npp_t; };
    template<> struct NppTypeTraits<CV_16U> { typedef Npp16u npp_t; };
    template<> struct NppTypeTraits<CV_16S> { typedef Npp16s npp_t; };
    template<> struct NppTypeTraits<CV_32S> { typedef Npp32s npp_t; };
    template<> struct NppTypeTraits<CV_32F> { typedef Npp32f npp_t; };
    template<> struct NppTypeTraits<CV_64F> { typedef Npp64f npp_t; };

    template <int DEPTH> struct NppRotateFunc
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

        typedef NppStatus (*func_t)(const npp_t* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                    npp_t* pDst, int nDstStep, NppiRect oDstROI,
                                    double nAngle, double nShiftX, double nShiftY, int eInterpolation);
    };

    template <int DEPTH, typename NppRotateFunc<DEPTH>::func_t func> struct NppRotate
    {
        typedef typename NppRotateFunc<DEPTH>::npp_t npp_t;

        static void call(const GpuMat& src, GpuMat& dst, Size dsize, double angle, double xShift, double yShift, int interpolation, cudaStream_t stream)
        {
            static const int npp_inter[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};

            NppStreamHandler h(stream);

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

            nppSafeCall( func(src.ptr<npp_t>(), srcsz, static_cast<int>(src.step), srcroi,
                dst.ptr<npp_t>(), static_cast<int>(dst.step), dstroi, angle, xShift, yShift, npp_inter[interpolation]) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
}

void cv::gpu::rotate(const GpuMat& src, GpuMat& dst, Size dsize, double angle, double xShift, double yShift, int interpolation, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& src, GpuMat& dst, Size dsize, double angle, double xShift, double yShift, int interpolation, cudaStream_t stream);

    static const func_t funcs[6][4] =
    {
        {NppRotate<CV_8U, nppiRotate_8u_C1R>::call, 0, NppRotate<CV_8U, nppiRotate_8u_C3R>::call, NppRotate<CV_8U, nppiRotate_8u_C4R>::call},
        {0,0,0,0},
        {NppRotate<CV_16U, nppiRotate_16u_C1R>::call, 0, NppRotate<CV_16U, nppiRotate_16u_C3R>::call, NppRotate<CV_16U, nppiRotate_16u_C4R>::call},
        {0,0,0,0},
        {0,0,0,0},
        {NppRotate<CV_32F, nppiRotate_32f_C1R>::call, 0, NppRotate<CV_32F, nppiRotate_32f_C3R>::call, NppRotate<CV_32F, nppiRotate_32f_C4R>::call}
    };

    CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
    CV_Assert(src.channels() == 1 || src.channels() == 3 || src.channels() == 4);
    CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC);

    dst.create(dsize, src.type());

    funcs[src.depth()][src.channels() - 1](src, dst, dsize, angle, xShift, yShift, interpolation, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// integral

void cv::gpu::integral(const GpuMat& src, GpuMat& sum, Stream& s)
{
    GpuMat buffer;
    integralBuffered(src, sum, buffer, s);
}

void cv::gpu::integralBuffered(const GpuMat& src, GpuMat& sum, GpuMat& buffer, Stream& s)
{
    CV_Assert(src.type() == CV_8UC1);

    sum.create(src.rows + 1, src.cols + 1, CV_32S);

    NcvSize32u roiSize;
    roiSize.width = src.cols;
    roiSize.height = src.rows;

    cudaDeviceProp prop;
    cudaSafeCall( cudaGetDeviceProperties(&prop, cv::gpu::getDevice()) );

    Ncv32u bufSize;
    ncvSafeCall( nppiStIntegralGetSize_8u32u(roiSize, &bufSize, prop) );
    ensureSizeIsEnough(1, bufSize, CV_8UC1, buffer);

    cudaStream_t stream = StreamAccessor::getStream(s);

    NppStStreamHandler h(stream);

    ncvSafeCall( nppiStIntegral_8u32u_C1R(const_cast<Ncv8u*>(src.ptr<Ncv8u>()), static_cast<int>(src.step),
        sum.ptr<Ncv32u>(), static_cast<int>(sum.step), roiSize, buffer.ptr<Ncv8u>(), bufSize, prop) );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

//////////////////////////////////////////////////////////////////////////////
// sqrIntegral

void cv::gpu::sqrIntegral(const GpuMat& src, GpuMat& sqsum, Stream& s)
{
    CV_Assert(src.type() == CV_8U);

    NcvSize32u roiSize;
    roiSize.width = src.cols;
    roiSize.height = src.rows;

    cudaDeviceProp prop;
    cudaSafeCall( cudaGetDeviceProperties(&prop, cv::gpu::getDevice()) );

    Ncv32u bufSize;
    ncvSafeCall(nppiStSqrIntegralGetSize_8u64u(roiSize, &bufSize, prop));
    GpuMat buf(1, bufSize, CV_8U);

    cudaStream_t stream = StreamAccessor::getStream(s);

    NppStStreamHandler h(stream);

    sqsum.create(src.rows + 1, src.cols + 1, CV_64F);
    ncvSafeCall(nppiStSqrIntegral_8u64u_C1R(const_cast<Ncv8u*>(src.ptr<Ncv8u>(0)), static_cast<int>(src.step),
            sqsum.ptr<Ncv64u>(0), static_cast<int>(sqsum.step), roiSize, buf.ptr<Ncv8u>(0), bufSize, prop));

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

//////////////////////////////////////////////////////////////////////////////
// columnSum

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        void columnSum_32F(const DevMem2Db src, const DevMem2Db dst);
    }
}}}

void cv::gpu::columnSum(const GpuMat& src, GpuMat& dst)
{
    using namespace ::cv::gpu::device::imgproc;

    CV_Assert(src.type() == CV_32F);

    dst.create(src.size(), CV_32F);

    device::imgproc::columnSum_32F(src, dst);
}

void cv::gpu::rectStdDev(const GpuMat& src, const GpuMat& sqr, GpuMat& dst, const Rect& rect, Stream& s)
{
    CV_Assert(src.type() == CV_32SC1 && sqr.type() == CV_64FC1);

    dst.create(src.size(), CV_32FC1);

    NppiSize sz;
    sz.width = src.cols;
    sz.height = src.rows;

    NppiRect nppRect;
    nppRect.height = rect.height;
    nppRect.width = rect.width;
    nppRect.x = rect.x;
    nppRect.y = rect.y;

    cudaStream_t stream = StreamAccessor::getStream(s);

    NppStreamHandler h(stream);

    nppSafeCall( nppiRectStdDev_32s32f_C1R(src.ptr<Npp32s>(), static_cast<int>(src.step), sqr.ptr<Npp64f>(), static_cast<int>(sqr.step),
                dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz, nppRect) );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}


////////////////////////////////////////////////////////////////////////
// Histogram

namespace
{
    typedef NppStatus (*get_buf_size_c1_t)(NppiSize oSizeROI, int nLevels, int* hpBufferSize);
    typedef NppStatus (*get_buf_size_c4_t)(NppiSize oSizeROI, int nLevels[], int* hpBufferSize);

    template<int SDEPTH> struct NppHistogramEvenFuncC1
    {
        typedef typename NppTypeTraits<SDEPTH>::npp_t src_t;

    typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist,
            int nLevels, Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u * pBuffer);
    };
    template<int SDEPTH> struct NppHistogramEvenFuncC4
    {
        typedef typename NppTypeTraits<SDEPTH>::npp_t src_t;

        typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, NppiSize oSizeROI,
            Npp32s * pHist[4], int nLevels[4], Npp32s nLowerLevel[4], Npp32s nUpperLevel[4], Npp8u * pBuffer);
    };

    template<int SDEPTH, typename NppHistogramEvenFuncC1<SDEPTH>::func_ptr func, get_buf_size_c1_t get_buf_size>
    struct NppHistogramEvenC1
    {
        typedef typename NppHistogramEvenFuncC1<SDEPTH>::src_t src_t;

        static void hist(const GpuMat& src, GpuMat& hist, GpuMat& buffer, int histSize, int lowerLevel, int upperLevel, cudaStream_t stream)
        {
            int levels = histSize + 1;
            hist.create(1, histSize, CV_32S);

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            int buf_size;
            get_buf_size(sz, levels, &buf_size);

            ensureSizeIsEnough(1, buf_size, CV_8U, buffer);

            NppStreamHandler h(stream);

            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, hist.ptr<Npp32s>(), levels,
                lowerLevel, upperLevel, buffer.ptr<Npp8u>()) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int SDEPTH, typename NppHistogramEvenFuncC4<SDEPTH>::func_ptr func, get_buf_size_c4_t get_buf_size>
    struct NppHistogramEvenC4
    {
        typedef typename NppHistogramEvenFuncC4<SDEPTH>::src_t src_t;

        static void hist(const GpuMat& src, GpuMat hist[4], GpuMat& buffer, int histSize[4], int lowerLevel[4], int upperLevel[4], cudaStream_t stream)
        {
            int levels[] = {histSize[0] + 1, histSize[1] + 1, histSize[2] + 1, histSize[3] + 1};
            hist[0].create(1, histSize[0], CV_32S);
            hist[1].create(1, histSize[1], CV_32S);
            hist[2].create(1, histSize[2], CV_32S);
            hist[3].create(1, histSize[3], CV_32S);

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Npp32s* pHist[] = {hist[0].ptr<Npp32s>(), hist[1].ptr<Npp32s>(), hist[2].ptr<Npp32s>(), hist[3].ptr<Npp32s>()};

            int buf_size;
            get_buf_size(sz, levels, &buf_size);

            ensureSizeIsEnough(1, buf_size, CV_8U, buffer);

            NppStreamHandler h(stream);

            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, pHist, levels, lowerLevel, upperLevel, buffer.ptr<Npp8u>()) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    template<int SDEPTH> struct NppHistogramRangeFuncC1
    {
        typedef typename NppTypeTraits<SDEPTH>::npp_t src_t;
        typedef Npp32s level_t;
        enum {LEVEL_TYPE_CODE=CV_32SC1};

        typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s* pHist,
            const Npp32s* pLevels, int nLevels, Npp8u* pBuffer);
    };
    template<> struct NppHistogramRangeFuncC1<CV_32F>
    {
        typedef Npp32f src_t;
        typedef Npp32f level_t;
        enum {LEVEL_TYPE_CODE=CV_32FC1};

        typedef NppStatus (*func_ptr)(const Npp32f* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s* pHist,
            const Npp32f* pLevels, int nLevels, Npp8u* pBuffer);
    };
    template<int SDEPTH> struct NppHistogramRangeFuncC4
    {
        typedef typename NppTypeTraits<SDEPTH>::npp_t src_t;
        typedef Npp32s level_t;
        enum {LEVEL_TYPE_CODE=CV_32SC1};

        typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s* pHist[4],
            const Npp32s* pLevels[4], int nLevels[4], Npp8u* pBuffer);
    };
    template<> struct NppHistogramRangeFuncC4<CV_32F>
    {
        typedef Npp32f src_t;
        typedef Npp32f level_t;
        enum {LEVEL_TYPE_CODE=CV_32FC1};

        typedef NppStatus (*func_ptr)(const Npp32f* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s* pHist[4],
            const Npp32f* pLevels[4], int nLevels[4], Npp8u* pBuffer);
    };

    template<int SDEPTH, typename NppHistogramRangeFuncC1<SDEPTH>::func_ptr func, get_buf_size_c1_t get_buf_size>
    struct NppHistogramRangeC1
    {
        typedef typename NppHistogramRangeFuncC1<SDEPTH>::src_t src_t;
        typedef typename NppHistogramRangeFuncC1<SDEPTH>::level_t level_t;
        enum {LEVEL_TYPE_CODE=NppHistogramRangeFuncC1<SDEPTH>::LEVEL_TYPE_CODE};

        static void hist(const GpuMat& src, GpuMat& hist, const GpuMat& levels, GpuMat& buffer, cudaStream_t stream)
        {
            CV_Assert(levels.type() == LEVEL_TYPE_CODE && levels.rows == 1);

            hist.create(1, levels.cols - 1, CV_32S);

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            int buf_size;
            get_buf_size(sz, levels.cols, &buf_size);

            ensureSizeIsEnough(1, buf_size, CV_8U, buffer);

            NppStreamHandler h(stream);

            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, hist.ptr<Npp32s>(), levels.ptr<level_t>(), levels.cols, buffer.ptr<Npp8u>()) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int SDEPTH, typename NppHistogramRangeFuncC4<SDEPTH>::func_ptr func, get_buf_size_c4_t get_buf_size>
    struct NppHistogramRangeC4
    {
        typedef typename NppHistogramRangeFuncC4<SDEPTH>::src_t src_t;
        typedef typename NppHistogramRangeFuncC1<SDEPTH>::level_t level_t;
        enum {LEVEL_TYPE_CODE=NppHistogramRangeFuncC1<SDEPTH>::LEVEL_TYPE_CODE};

        static void hist(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], GpuMat& buffer, cudaStream_t stream)
        {
            CV_Assert(levels[0].type() == LEVEL_TYPE_CODE && levels[0].rows == 1);
            CV_Assert(levels[1].type() == LEVEL_TYPE_CODE && levels[1].rows == 1);
            CV_Assert(levels[2].type() == LEVEL_TYPE_CODE && levels[2].rows == 1);
            CV_Assert(levels[3].type() == LEVEL_TYPE_CODE && levels[3].rows == 1);

            hist[0].create(1, levels[0].cols - 1, CV_32S);
            hist[1].create(1, levels[1].cols - 1, CV_32S);
            hist[2].create(1, levels[2].cols - 1, CV_32S);
            hist[3].create(1, levels[3].cols - 1, CV_32S);

            Npp32s* pHist[] = {hist[0].ptr<Npp32s>(), hist[1].ptr<Npp32s>(), hist[2].ptr<Npp32s>(), hist[3].ptr<Npp32s>()};
            int nLevels[] = {levels[0].cols, levels[1].cols, levels[2].cols, levels[3].cols};
            const level_t* pLevels[] = {levels[0].ptr<level_t>(), levels[1].ptr<level_t>(), levels[2].ptr<level_t>(), levels[3].ptr<level_t>()};

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            int buf_size;
            get_buf_size(sz, nLevels, &buf_size);

            ensureSizeIsEnough(1, buf_size, CV_8U, buffer);

            NppStreamHandler h(stream);

            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, pHist, pLevels, nLevels, buffer.ptr<Npp8u>()) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
}

void cv::gpu::evenLevels(GpuMat& levels, int nLevels, int lowerLevel, int upperLevel)
{
    Mat host_levels(1, nLevels, CV_32SC1);
    nppSafeCall( nppiEvenLevelsHost_32s(host_levels.ptr<Npp32s>(), nLevels, lowerLevel, upperLevel) );
    levels.upload(host_levels);
}

void cv::gpu::histEven(const GpuMat& src, GpuMat& hist, int histSize, int lowerLevel, int upperLevel, Stream& stream)
{
    GpuMat buf;
    histEven(src, hist, buf, histSize, lowerLevel, upperLevel, stream);
}

void cv::gpu::histEven(const GpuMat& src, GpuMat& hist, GpuMat& buf, int histSize, int lowerLevel, int upperLevel, Stream& stream)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_16UC1 || src.type() == CV_16SC1 );

    typedef void (*hist_t)(const GpuMat& src, GpuMat& hist, GpuMat& buf, int levels, int lowerLevel, int upperLevel, cudaStream_t stream);
    static const hist_t hist_callers[] =
    {
        NppHistogramEvenC1<CV_8U , nppiHistogramEven_8u_C1R , nppiHistogramEvenGetBufferSize_8u_C1R >::hist,
        0,
        NppHistogramEvenC1<CV_16U, nppiHistogramEven_16u_C1R, nppiHistogramEvenGetBufferSize_16u_C1R>::hist,
        NppHistogramEvenC1<CV_16S, nppiHistogramEven_16s_C1R, nppiHistogramEvenGetBufferSize_16s_C1R>::hist
    };

    hist_callers[src.depth()](src, hist, buf, histSize, lowerLevel, upperLevel, StreamAccessor::getStream(stream));
}

void cv::gpu::histEven(const GpuMat& src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream)
{
    GpuMat buf;
    histEven(src, hist, buf, histSize, lowerLevel, upperLevel, stream);
}

void cv::gpu::histEven(const GpuMat& src, GpuMat hist[4], GpuMat& buf, int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream)
{
    CV_Assert(src.type() == CV_8UC4 || src.type() == CV_16UC4 || src.type() == CV_16SC4 );

    typedef void (*hist_t)(const GpuMat& src, GpuMat hist[4], GpuMat& buf, int levels[4], int lowerLevel[4], int upperLevel[4], cudaStream_t stream);
    static const hist_t hist_callers[] =
    {
        NppHistogramEvenC4<CV_8U , nppiHistogramEven_8u_C4R , nppiHistogramEvenGetBufferSize_8u_C4R >::hist,
        0,
        NppHistogramEvenC4<CV_16U, nppiHistogramEven_16u_C4R, nppiHistogramEvenGetBufferSize_16u_C4R>::hist,
        NppHistogramEvenC4<CV_16S, nppiHistogramEven_16s_C4R, nppiHistogramEvenGetBufferSize_16s_C4R>::hist
    };

    hist_callers[src.depth()](src, hist, buf, histSize, lowerLevel, upperLevel, StreamAccessor::getStream(stream));
}

void cv::gpu::histRange(const GpuMat& src, GpuMat& hist, const GpuMat& levels, Stream& stream)
{
    GpuMat buf;
    histRange(src, hist, levels, buf, stream);
}

void cv::gpu::histRange(const GpuMat& src, GpuMat& hist, const GpuMat& levels, GpuMat& buf, Stream& stream)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_16UC1 || src.type() == CV_16SC1 || src.type() == CV_32FC1);

    typedef void (*hist_t)(const GpuMat& src, GpuMat& hist, const GpuMat& levels, GpuMat& buf, cudaStream_t stream);
    static const hist_t hist_callers[] =
    {
        NppHistogramRangeC1<CV_8U , nppiHistogramRange_8u_C1R , nppiHistogramRangeGetBufferSize_8u_C1R >::hist,
        0,
        NppHistogramRangeC1<CV_16U, nppiHistogramRange_16u_C1R, nppiHistogramRangeGetBufferSize_16u_C1R>::hist,
        NppHistogramRangeC1<CV_16S, nppiHistogramRange_16s_C1R, nppiHistogramRangeGetBufferSize_16s_C1R>::hist,
        0,
        NppHistogramRangeC1<CV_32F, nppiHistogramRange_32f_C1R, nppiHistogramRangeGetBufferSize_32f_C1R>::hist
    };

    hist_callers[src.depth()](src, hist, levels, buf, StreamAccessor::getStream(stream));
}

void cv::gpu::histRange(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], Stream& stream)
{
    GpuMat buf;
    histRange(src, hist, levels, buf, stream);
}

void cv::gpu::histRange(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], GpuMat& buf, Stream& stream)
{
    CV_Assert(src.type() == CV_8UC4 || src.type() == CV_16UC4 || src.type() == CV_16SC4 || src.type() == CV_32FC4);

    typedef void (*hist_t)(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], GpuMat& buf, cudaStream_t stream);
    static const hist_t hist_callers[] =
    {
        NppHistogramRangeC4<CV_8U , nppiHistogramRange_8u_C4R , nppiHistogramRangeGetBufferSize_8u_C4R >::hist,
        0,
        NppHistogramRangeC4<CV_16U, nppiHistogramRange_16u_C4R, nppiHistogramRangeGetBufferSize_16u_C4R>::hist,
        NppHistogramRangeC4<CV_16S, nppiHistogramRange_16s_C4R, nppiHistogramRangeGetBufferSize_16s_C4R>::hist,
        0,
        NppHistogramRangeC4<CV_32F, nppiHistogramRange_32f_C4R, nppiHistogramRangeGetBufferSize_32f_C4R>::hist
    };

    hist_callers[src.depth()](src, hist, levels, buf, StreamAccessor::getStream(stream));
}

namespace cv { namespace gpu { namespace device
{
    namespace hist
    {
        void histogram256_gpu(DevMem2Db src, int* hist, unsigned int* buf, cudaStream_t stream);

        const int PARTIAL_HISTOGRAM256_COUNT = 240;
        const int HISTOGRAM256_BIN_COUNT     = 256;

        void equalizeHist_gpu(DevMem2Db src, DevMem2Db dst, const int* lut, cudaStream_t stream);
    }
}}}

void cv::gpu::calcHist(const GpuMat& src, GpuMat& hist, Stream& stream)
{
    GpuMat buf;
    calcHist(src, hist, buf, stream);
}

void cv::gpu::calcHist(const GpuMat& src, GpuMat& hist, GpuMat& buf, Stream& stream)
{
    using namespace ::cv::gpu::device::hist;

    CV_Assert(src.type() == CV_8UC1);

    hist.create(1, 256, CV_32SC1);

    ensureSizeIsEnough(1, PARTIAL_HISTOGRAM256_COUNT * HISTOGRAM256_BIN_COUNT, CV_32SC1, buf);

    histogram256_gpu(src, hist.ptr<int>(), buf.ptr<unsigned int>(), StreamAccessor::getStream(stream));
}

void cv::gpu::equalizeHist(const GpuMat& src, GpuMat& dst, Stream& stream)
{
    GpuMat hist;
    GpuMat buf;
    equalizeHist(src, dst, hist, buf, stream);
}

void cv::gpu::equalizeHist(const GpuMat& src, GpuMat& dst, GpuMat& hist, Stream& stream)
{
    GpuMat buf;
    equalizeHist(src, dst, hist, buf, stream);
}

void cv::gpu::equalizeHist(const GpuMat& src, GpuMat& dst, GpuMat& hist, GpuMat& buf, Stream& s)
{
    using namespace ::cv::gpu::device::hist;

    CV_Assert(src.type() == CV_8UC1);

    dst.create(src.size(), src.type());

    int intBufSize;
    nppSafeCall( nppsIntegralGetBufferSize_32s(256, &intBufSize) );

    int bufSize = static_cast<int>(std::max(256 * 240 * sizeof(int), intBufSize + 256 * sizeof(int)));

    ensureSizeIsEnough(1, bufSize, CV_8UC1, buf);

    GpuMat histBuf(1, 256 * 240, CV_32SC1, buf.ptr());
    GpuMat intBuf(1, intBufSize, CV_8UC1, buf.ptr());
    GpuMat lut(1, 256, CV_32S, buf.ptr() + intBufSize);

    calcHist(src, hist, histBuf, s);

    cudaStream_t stream = StreamAccessor::getStream(s);

    NppStreamHandler h(stream);

    nppSafeCall( nppsIntegral_32s(hist.ptr<Npp32s>(), lut.ptr<Npp32s>(), 256, intBuf.ptr<Npp8u>()) );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );

    equalizeHist_gpu(src, dst, lut.ptr<int>(), stream);
}

////////////////////////////////////////////////////////////////////////
// cornerHarris & minEgenVal

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        void cornerHarris_gpu(int block_size, float k, DevMem2Df Dx, DevMem2Df Dy, DevMem2Df dst, int border_type, cudaStream_t stream);
        void cornerMinEigenVal_gpu(int block_size, DevMem2Df Dx, DevMem2Df Dy, DevMem2Df dst, int border_type, cudaStream_t stream);
    }
}}}

namespace
{
    void extractCovData(const GpuMat& src, GpuMat& Dx, GpuMat& Dy, GpuMat& buf, int blockSize, int ksize, int borderType, Stream& stream)
    {
        double scale = static_cast<double>(1 << ((ksize > 0 ? ksize : 3) - 1)) * blockSize;

        if (ksize < 0)
            scale *= 2.;

        if (src.depth() == CV_8U)
            scale *= 255.;

        scale = 1./scale;

        Dx.create(src.size(), CV_32F);
        Dy.create(src.size(), CV_32F);

        if (ksize > 0)
        {
            Sobel(src, Dx, CV_32F, 1, 0, buf, ksize, scale, borderType, -1, stream);
            Sobel(src, Dy, CV_32F, 0, 1, buf, ksize, scale, borderType, -1, stream);
        }
        else
        {
            Scharr(src, Dx, CV_32F, 1, 0, buf, scale, borderType, -1, stream);
            Scharr(src, Dy, CV_32F, 0, 1, buf, scale, borderType, -1, stream);
        }
    }
}

bool cv::gpu::tryConvertToGpuBorderType(int cpuBorderType, int& gpuBorderType)
{
    switch (cpuBorderType)
    {
    case cv::BORDER_REFLECT101:
        gpuBorderType = cv::gpu::BORDER_REFLECT101_GPU;
        return true;
    case cv::BORDER_REPLICATE:
        gpuBorderType = cv::gpu::BORDER_REPLICATE_GPU;
        return true;
    case cv::BORDER_CONSTANT:
        gpuBorderType = cv::gpu::BORDER_CONSTANT_GPU;
        return true;
    case cv::BORDER_REFLECT:
        gpuBorderType = cv::gpu::BORDER_REFLECT_GPU;
        return true;
    case cv::BORDER_WRAP:
        gpuBorderType = cv::gpu::BORDER_WRAP_GPU;
        return true;
    default:
        return false;
    };
    return false;
}

void cv::gpu::cornerHarris(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, double k, int borderType)
{
    GpuMat Dx, Dy;
    cornerHarris(src, dst, Dx, Dy, blockSize, ksize, k, borderType);
}

void cv::gpu::cornerHarris(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, int blockSize, int ksize, double k, int borderType)
{
    GpuMat buf;
    cornerHarris(src, dst, Dx, Dy, buf, blockSize, ksize, k, borderType);
}

void cv::gpu::cornerHarris(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, GpuMat& buf, int blockSize, int ksize, double k, int borderType, Stream& stream)
{
    using namespace cv::gpu::device::imgproc;

    CV_Assert(borderType == cv::BORDER_REFLECT101 || borderType == cv::BORDER_REPLICATE || borderType == cv::BORDER_REFLECT);

    int gpuBorderType;
    CV_Assert(tryConvertToGpuBorderType(borderType, gpuBorderType));

    extractCovData(src, Dx, Dy, buf, blockSize, ksize, borderType, stream);

    dst.create(src.size(), CV_32F);

    cornerHarris_gpu(blockSize, static_cast<float>(k), Dx, Dy, dst, gpuBorderType, StreamAccessor::getStream(stream));
}

void cv::gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, int borderType)
{
    GpuMat Dx, Dy;
    cornerMinEigenVal(src, dst, Dx, Dy, blockSize, ksize, borderType);
}

void cv::gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, int blockSize, int ksize, int borderType)
{
    GpuMat buf;
    cornerMinEigenVal(src, dst, Dx, Dy, buf, blockSize, ksize, borderType);
}

void cv::gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, GpuMat& buf, int blockSize, int ksize, int borderType, Stream& stream)
{
    using namespace ::cv::gpu::device::imgproc;

    CV_Assert(borderType == cv::BORDER_REFLECT101 || borderType == cv::BORDER_REPLICATE || borderType == cv::BORDER_REFLECT);

    int gpuBorderType;
    CV_Assert(tryConvertToGpuBorderType(borderType, gpuBorderType));

    extractCovData(src, Dx, Dy, buf, blockSize, ksize, borderType, stream);

    dst.create(src.size(), CV_32F);

    cornerMinEigenVal_gpu(blockSize, Dx, Dy, dst, gpuBorderType, StreamAccessor::getStream(stream));
}

//////////////////////////////////////////////////////////////////////////////
// mulSpectrums

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        void mulSpectrums(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, DevMem2D_<cufftComplex> c, cudaStream_t stream);

        void mulSpectrums_CONJ(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, DevMem2D_<cufftComplex> c, cudaStream_t stream);
    }
}}}

void cv::gpu::mulSpectrums(const GpuMat& a, const GpuMat& b, GpuMat& c, int flags, bool conjB, Stream& stream)
{
    using namespace ::cv::gpu::device::imgproc;

    typedef void (*Caller)(const PtrStep<cufftComplex>, const PtrStep<cufftComplex>, DevMem2D_<cufftComplex>, cudaStream_t stream);

    static Caller callers[] = { mulSpectrums, mulSpectrums_CONJ };

    CV_Assert(a.type() == b.type() && a.type() == CV_32FC2);
    CV_Assert(a.size() == b.size());

    c.create(a.size(), CV_32FC2);

    Caller caller = callers[(int)conjB];
    caller(a, b, c, StreamAccessor::getStream(stream));
}

//////////////////////////////////////////////////////////////////////////////
// mulAndScaleSpectrums

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        void mulAndScaleSpectrums(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, float scale, DevMem2D_<cufftComplex> c, cudaStream_t stream);

        void mulAndScaleSpectrums_CONJ(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, float scale, DevMem2D_<cufftComplex> c, cudaStream_t stream);
    }
}}}

void cv::gpu::mulAndScaleSpectrums(const GpuMat& a, const GpuMat& b, GpuMat& c, int flags, float scale, bool conjB, Stream& stream)
{
    using namespace ::cv::gpu::device::imgproc;

    typedef void (*Caller)(const PtrStep<cufftComplex>, const PtrStep<cufftComplex>, float scale, DevMem2D_<cufftComplex>, cudaStream_t stream);
    static Caller callers[] = { mulAndScaleSpectrums, mulAndScaleSpectrums_CONJ };

    CV_Assert(a.type() == b.type() && a.type() == CV_32FC2);
    CV_Assert(a.size() == b.size());

    c.create(a.size(), CV_32FC2);

    Caller caller = callers[(int)conjB];
    caller(a, b, scale, c, StreamAccessor::getStream(stream));
}

//////////////////////////////////////////////////////////////////////////////
// dft

void cv::gpu::dft(const GpuMat& src, GpuMat& dst, Size dft_size, int flags, Stream& stream)
{
#ifndef HAVE_CUFFT

    OPENCV_GPU_UNUSED(src);
    OPENCV_GPU_UNUSED(dst);
    OPENCV_GPU_UNUSED(dft_size);
    OPENCV_GPU_UNUSED(flags);
    OPENCV_GPU_UNUSED(stream);

    throw_nogpu();

#else

    CV_Assert(src.type() == CV_32F || src.type() == CV_32FC2);

    // We don't support unpacked output (in the case of real input)
    CV_Assert(!(flags & DFT_COMPLEX_OUTPUT));

    bool is_1d_input = (dft_size.height == 1) || (dft_size.width == 1);
    int is_row_dft = flags & DFT_ROWS;
    int is_scaled_dft = flags & DFT_SCALE;
    int is_inverse = flags & DFT_INVERSE;
    bool is_complex_input = src.channels() == 2;
    bool is_complex_output = !(flags & DFT_REAL_OUTPUT);

    // We don't support real-to-real transform
    CV_Assert(is_complex_input || is_complex_output);

    GpuMat src_data;

    // Make sure here we work with the continuous input,
    // as CUFFT can't handle gaps
    src_data = src;
    createContinuous(src.rows, src.cols, src.type(), src_data);
    if (src_data.data != src.data)
        src.copyTo(src_data);

    Size dft_size_opt = dft_size;
    if (is_1d_input && !is_row_dft)
    {
        // If the source matrix is single column handle it as single row
        dft_size_opt.width = std::max(dft_size.width, dft_size.height);
        dft_size_opt.height = std::min(dft_size.width, dft_size.height);
    }

    cufftType dft_type = CUFFT_R2C;
    if (is_complex_input)
        dft_type = is_complex_output ? CUFFT_C2C : CUFFT_C2R;

    CV_Assert(dft_size_opt.width > 1);

    cufftHandle plan;
    if (is_1d_input || is_row_dft)
        cufftPlan1d(&plan, dft_size_opt.width, dft_type, dft_size_opt.height);
    else
        cufftPlan2d(&plan, dft_size_opt.height, dft_size_opt.width, dft_type);

    cufftSafeCall( cufftSetStream(plan, StreamAccessor::getStream(stream)) );

    if (is_complex_input)
    {
        if (is_complex_output)
        {
            createContinuous(dft_size, CV_32FC2, dst);
            cufftSafeCall(cufftExecC2C(
                    plan, src_data.ptr<cufftComplex>(), dst.ptr<cufftComplex>(),
                    is_inverse ? CUFFT_INVERSE : CUFFT_FORWARD));
        }
        else
        {
            createContinuous(dft_size, CV_32F, dst);
            cufftSafeCall(cufftExecC2R(
                    plan, src_data.ptr<cufftComplex>(), dst.ptr<cufftReal>()));
        }
    }
    else
    {
        // We could swap dft_size for efficiency. Here we must reflect it
        if (dft_size == dft_size_opt)
            createContinuous(Size(dft_size.width / 2 + 1, dft_size.height), CV_32FC2, dst);
        else
            createContinuous(Size(dft_size.width, dft_size.height / 2 + 1), CV_32FC2, dst);

        cufftSafeCall(cufftExecR2C(
                plan, src_data.ptr<cufftReal>(), dst.ptr<cufftComplex>()));
    }

    cufftSafeCall(cufftDestroy(plan));

    if (is_scaled_dft)
        multiply(dst, Scalar::all(1. / dft_size.area()), dst, 1, -1, stream);

#endif
}

//////////////////////////////////////////////////////////////////////////////
// convolve

void cv::gpu::ConvolveBuf::create(Size image_size, Size templ_size)
{
    result_size = Size(image_size.width - templ_size.width + 1,
                       image_size.height - templ_size.height + 1);

    block_size = user_block_size;
    if (user_block_size.width == 0 || user_block_size.height == 0)
        block_size = estimateBlockSize(result_size, templ_size);

    dft_size.width = 1 << int(ceil(std::log(block_size.width + templ_size.width - 1.) / std::log(2.)));
    dft_size.height = 1 << int(ceil(std::log(block_size.height + templ_size.height - 1.) / std::log(2.)));

    // CUFFT has hard-coded kernels for power-of-2 sizes (up to 8192),
    // see CUDA Toolkit 4.1 CUFFT Library Programming Guide
    if (dft_size.width > 8192)
        dft_size.width = getOptimalDFTSize(block_size.width + templ_size.width - 1);
    if (dft_size.height > 8192)
        dft_size.height = getOptimalDFTSize(block_size.height + templ_size.height - 1);

    // To avoid wasting time doing small DFTs
    dft_size.width = std::max(dft_size.width, 512);
    dft_size.height = std::max(dft_size.height, 512);

    createContinuous(dft_size, CV_32F, image_block);
    createContinuous(dft_size, CV_32F, templ_block);
    createContinuous(dft_size, CV_32F, result_data);

    spect_len = dft_size.height * (dft_size.width / 2 + 1);
    createContinuous(1, spect_len, CV_32FC2, image_spect);
    createContinuous(1, spect_len, CV_32FC2, templ_spect);
    createContinuous(1, spect_len, CV_32FC2, result_spect);

    // Use maximum result matrix block size for the estimated DFT block size
    block_size.width = std::min(dft_size.width - templ_size.width + 1, result_size.width);
    block_size.height = std::min(dft_size.height - templ_size.height + 1, result_size.height);
}


Size cv::gpu::ConvolveBuf::estimateBlockSize(Size result_size, Size /*templ_size*/)
{
    int width = (result_size.width + 2) / 3;
    int height = (result_size.height + 2) / 3;
    width = std::min(width, result_size.width);
    height = std::min(height, result_size.height);    
    return Size(width, height);
}


void cv::gpu::convolve(const GpuMat& image, const GpuMat& templ, GpuMat& result, bool ccorr)
{
    ConvolveBuf buf;
    convolve(image, templ, result, ccorr, buf);
}

void cv::gpu::convolve(const GpuMat& image, const GpuMat& templ, GpuMat& result, bool ccorr, ConvolveBuf& buf, Stream& stream)
{
    using namespace ::cv::gpu::device::imgproc;

#ifndef HAVE_CUFFT
    throw_nogpu();
#else
    StaticAssert<sizeof(float) == sizeof(cufftReal)>::check();
    StaticAssert<sizeof(float) * 2 == sizeof(cufftComplex)>::check();

    CV_Assert(image.type() == CV_32F);
    CV_Assert(templ.type() == CV_32F);

    buf.create(image.size(), templ.size());
    result.create(buf.result_size, CV_32F);

    Size& block_size = buf.block_size;
    Size& dft_size = buf.dft_size;

    GpuMat& image_block = buf.image_block;
    GpuMat& templ_block = buf.templ_block;
    GpuMat& result_data = buf.result_data;

    GpuMat& image_spect = buf.image_spect;
    GpuMat& templ_spect = buf.templ_spect;
    GpuMat& result_spect = buf.result_spect;

    cufftHandle planR2C, planC2R;
    cufftSafeCall(cufftPlan2d(&planC2R, dft_size.height, dft_size.width, CUFFT_C2R));
    cufftSafeCall(cufftPlan2d(&planR2C, dft_size.height, dft_size.width, CUFFT_R2C));   

    cufftSafeCall( cufftSetStream(planR2C, StreamAccessor::getStream(stream)) );
    cufftSafeCall( cufftSetStream(planC2R, StreamAccessor::getStream(stream)) );

    GpuMat templ_roi(templ.size(), CV_32F, templ.data, templ.step);
    copyMakeBorder(templ_roi, templ_block, 0, templ_block.rows - templ_roi.rows, 0,
                   templ_block.cols - templ_roi.cols, 0, Scalar(), stream);

    cufftSafeCall(cufftExecR2C(planR2C, templ_block.ptr<cufftReal>(),
                               templ_spect.ptr<cufftComplex>()));

    // Process all blocks of the result matrix
    for (int y = 0; y < result.rows; y += block_size.height)
    {
        for (int x = 0; x < result.cols; x += block_size.width)
        {
            Size image_roi_size(std::min(x + dft_size.width, image.cols) - x,
                                std::min(y + dft_size.height, image.rows) - y);
            GpuMat image_roi(image_roi_size, CV_32F, (void*)(image.ptr<float>(y) + x),
                             image.step);
            copyMakeBorder(image_roi, image_block, 0, image_block.rows - image_roi.rows,
                           0, image_block.cols - image_roi.cols, 0, Scalar(), stream);

            cufftSafeCall(cufftExecR2C(planR2C, image_block.ptr<cufftReal>(),
                                       image_spect.ptr<cufftComplex>()));
            mulAndScaleSpectrums(image_spect, templ_spect, result_spect, 0,
                                 1.f / dft_size.area(), ccorr, stream);
            cufftSafeCall(cufftExecC2R(planC2R, result_spect.ptr<cufftComplex>(),
                                       result_data.ptr<cufftReal>()));

            Size result_roi_size(std::min(x + block_size.width, result.cols) - x,
                                 std::min(y + block_size.height, result.rows) - y);
            GpuMat result_roi(result_roi_size, result.type(),
                              (void*)(result.ptr<float>(y) + x), result.step);
            GpuMat result_block(result_roi_size, result_data.type(),
                                result_data.ptr(), result_data.step);

            if (stream)
                stream.enqueueCopy(result_block, result_roi);
            else
                result_block.copyTo(result_roi);
        }
    }

    cufftSafeCall(cufftDestroy(planR2C));
    cufftSafeCall(cufftDestroy(planC2R));
#endif
}


//////////////////////////////////////////////////////////////////////////////
// Canny

cv::gpu::CannyBuf::CannyBuf(const GpuMat& dx_, const GpuMat& dy_) : dx(dx_), dy(dy_)
{
    CV_Assert(dx_.type() == CV_32SC1 && dy_.type() == CV_32SC1 && dx_.size() == dy_.size());

    create(dx_.size(), -1);
}

void cv::gpu::CannyBuf::create(const Size& image_size, int apperture_size)
{
    ensureSizeIsEnough(image_size, CV_32SC1, dx);
    ensureSizeIsEnough(image_size, CV_32SC1, dy);

    if (apperture_size == 3)
    {
        ensureSizeIsEnough(image_size, CV_32SC1, dx_buf);
        ensureSizeIsEnough(image_size, CV_32SC1, dy_buf);
    }
    else if(apperture_size > 0)
    {
        if (!filterDX)
            filterDX = createDerivFilter_GPU(CV_8UC1, CV_32S, 1, 0, apperture_size, BORDER_REPLICATE);
        if (!filterDY)
            filterDY = createDerivFilter_GPU(CV_8UC1, CV_32S, 0, 1, apperture_size, BORDER_REPLICATE);
    }

    ensureSizeIsEnough(image_size.height + 2, image_size.width + 2, CV_32FC1, edgeBuf);

    ensureSizeIsEnough(1, image_size.width * image_size.height, CV_16UC2, trackBuf1);
    ensureSizeIsEnough(1, image_size.width * image_size.height, CV_16UC2, trackBuf2);
}

void cv::gpu::CannyBuf::release()
{
    dx.release();
    dy.release();
    dx_buf.release();
    dy_buf.release();
    edgeBuf.release();
    trackBuf1.release();
    trackBuf2.release();
}

namespace cv { namespace gpu { namespace device
{
    namespace canny
    {
        void calcSobelRowPass_gpu(PtrStepb src, PtrStepi dx_buf, PtrStepi dy_buf, int rows, int cols);

        void calcMagnitude_gpu(PtrStepi dx_buf, PtrStepi dy_buf, PtrStepi dx, PtrStepi dy, PtrStepf mag, int rows, int cols, bool L2Grad);
        void calcMagnitude_gpu(PtrStepi dx, PtrStepi dy, PtrStepf mag, int rows, int cols, bool L2Grad);

        void calcMap_gpu(PtrStepi dx, PtrStepi dy, PtrStepf mag, PtrStepi map, int rows, int cols, float low_thresh, float high_thresh);

        void edgesHysteresisLocal_gpu(PtrStepi map, ushort2* st1, int rows, int cols);

        void edgesHysteresisGlobal_gpu(PtrStepi map, ushort2* st1, ushort2* st2, int rows, int cols);

        void getEdges_gpu(PtrStepi map, PtrStepb dst, int rows, int cols);
    }
}}}

namespace
{
    void CannyCaller(CannyBuf& buf, GpuMat& dst, float low_thresh, float high_thresh)
    {
        using namespace ::cv::gpu::device::canny;

        calcMap_gpu(buf.dx, buf.dy, buf.edgeBuf, buf.edgeBuf, dst.rows, dst.cols, low_thresh, high_thresh);

        edgesHysteresisLocal_gpu(buf.edgeBuf, buf.trackBuf1.ptr<ushort2>(), dst.rows, dst.cols);

        edgesHysteresisGlobal_gpu(buf.edgeBuf, buf.trackBuf1.ptr<ushort2>(), buf.trackBuf2.ptr<ushort2>(), dst.rows, dst.cols);

        getEdges_gpu(buf.edgeBuf, dst, dst.rows, dst.cols);
    }
}

void cv::gpu::Canny(const GpuMat& src, GpuMat& dst, double low_thresh, double high_thresh, int apperture_size, bool L2gradient)
{
    CannyBuf buf(src.size(), apperture_size);
    Canny(src, buf, dst, low_thresh, high_thresh, apperture_size, L2gradient);
}

void cv::gpu::Canny(const GpuMat& src, CannyBuf& buf, GpuMat& dst, double low_thresh, double high_thresh, int apperture_size, bool L2gradient)
{
    using namespace ::cv::gpu::device::canny;

    CV_Assert(src.type() == CV_8UC1);

    if (!TargetArchs::builtWith(SHARED_ATOMICS) || !DeviceInfo().supports(SHARED_ATOMICS))
        CV_Error(CV_StsNotImplemented, "The device doesn't support shared atomics");

    if( low_thresh > high_thresh )
        std::swap( low_thresh, high_thresh);

    dst.create(src.size(), CV_8U);
    dst.setTo(Scalar::all(0));

    buf.create(src.size(), apperture_size);
    buf.edgeBuf.setTo(Scalar::all(0));

    if (apperture_size == 3)
    {
        calcSobelRowPass_gpu(src, buf.dx_buf, buf.dy_buf, src.rows, src.cols);

        calcMagnitude_gpu(buf.dx_buf, buf.dy_buf, buf.dx, buf.dy, buf.edgeBuf, src.rows, src.cols, L2gradient);
    }
    else
    {
        buf.filterDX->apply(src, buf.dx, Rect(0, 0, src.cols, src.rows));
        buf.filterDY->apply(src, buf.dy, Rect(0, 0, src.cols, src.rows));

        calcMagnitude_gpu(buf.dx, buf.dy, buf.edgeBuf, src.rows, src.cols, L2gradient);
    }

    CannyCaller(buf, dst, static_cast<float>(low_thresh), static_cast<float>(high_thresh));
}

void cv::gpu::Canny(const GpuMat& dx, const GpuMat& dy, GpuMat& dst, double low_thresh, double high_thresh, bool L2gradient)
{
    CannyBuf buf(dx, dy);
    Canny(dx, dy, buf, dst, low_thresh, high_thresh, L2gradient);
}

void cv::gpu::Canny(const GpuMat& dx, const GpuMat& dy, CannyBuf& buf, GpuMat& dst, double low_thresh, double high_thresh, bool L2gradient)
{
    using namespace ::cv::gpu::device::canny;

    CV_Assert(TargetArchs::builtWith(SHARED_ATOMICS) && DeviceInfo().supports(SHARED_ATOMICS));
    CV_Assert(dx.type() == CV_32SC1 && dy.type() == CV_32SC1 && dx.size() == dy.size());

    if( low_thresh > high_thresh )
        std::swap( low_thresh, high_thresh);

    dst.create(dx.size(), CV_8U);
    dst.setTo(Scalar::all(0));

    buf.dx = dx; buf.dy = dy;
    buf.create(dx.size(), -1);
    buf.edgeBuf.setTo(Scalar::all(0));

    calcMagnitude_gpu(dx, dy, buf.edgeBuf, dx.rows, dx.cols, L2gradient);

    CannyCaller(buf, dst, static_cast<float>(low_thresh), static_cast<float>(high_thresh));
}

#endif /* !defined (HAVE_CUDA) */


