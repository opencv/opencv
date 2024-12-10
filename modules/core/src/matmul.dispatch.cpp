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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
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
#include <opencv2/core/utils/logger.hpp>

#include "opencl_kernels_core.hpp"
#include "opencv2/core/opencl/runtime/opencl_clblas.hpp"
#include "opencv2/core/opencl/runtime/opencl_core.hpp"
#include "intel_gpu_gemm.inl.hpp"

#include "matmul.simd.hpp"
#include "matmul.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv
{

/****************************************************************************************\
*                                         GEMM                                           *
\****************************************************************************************/

#ifdef HAVE_CLAMDBLAS

static bool ocl_gemm_amdblas( InputArray matA, InputArray matB, double alpha,
                      InputArray matC, double beta, OutputArray matD, int flags )
{
    int type = matA.type(), esz = CV_ELEM_SIZE(type);
    bool haveC = matC.kind() != cv::_InputArray::NONE;
    Size sizeA = matA.size(), sizeB = matB.size(), sizeC = haveC ? matC.size() : Size(0, 0);
    bool atrans = (flags & GEMM_1_T) != 0, btrans = (flags & GEMM_2_T) != 0, ctrans = (flags & GEMM_3_T) != 0;

    if (atrans)
        sizeA = Size(sizeA.height, sizeA.width);
    if (btrans)
        sizeB = Size(sizeB.height, sizeB.width);
    if (haveC && ctrans)
        sizeC = Size(sizeC.height, sizeC.width);

    Size sizeD(sizeB.width, sizeA.height);

    CV_Assert( matB.type() == type && (!haveC || matC.type() == type) );
    CV_Assert( sizeA.width == sizeB.height && (!haveC || sizeC == sizeD) );

    matD.create(sizeD, type);
    if ( matA.offset() % esz != 0 || matA.step() % esz != 0 ||
         matB.offset() % esz != 0 || matB.step() % esz != 0 ||
         (haveC && (matC.offset() % esz != 0 || matC.step() % esz != 0)) )
        return false;

    UMat A = matA.getUMat(), B = matB.getUMat(), D = matD.getUMat();
    if (!ocl::internal::isCLBuffer(A) || !ocl::internal::isCLBuffer(B) || !ocl::internal::isCLBuffer(D))
    {
        return false;
    }
    if (haveC)
    {
        UMat C = matC.getUMat();
        if (!ocl::internal::isCLBuffer(C))
            return false;
    }
    if (haveC)
        ctrans ? transpose(matC, D) : matC.copyTo(D);
    else
        D.setTo(Scalar::all(0));

    int M = sizeD.height, N = sizeD.width, K = sizeA.width;
    int lda = (int)A.step / esz, ldb = (int)B.step / esz, ldc = (int)D.step / esz;
    int offa = (int)A.offset / esz, offb = (int)B.offset / esz, offc = (int)D.offset / esz;

    cl_command_queue clq = (cl_command_queue)ocl::Queue::getDefault().ptr();
    clblasTranspose transA = atrans ? clblasTrans : clblasNoTrans;
    clblasTranspose transB = btrans ? clblasTrans : clblasNoTrans;
    clblasOrder order = clblasRowMajor;
    clblasStatus status = clblasSuccess;

    if (type == CV_32FC1)
        status = clblasSgemm(order, transA, transB, M, N, K,
                             (cl_float)alpha, (const cl_mem)A.handle(ACCESS_READ), offa, lda,
                             (const cl_mem)B.handle(ACCESS_READ), offb, ldb,
                             (cl_float)beta, (cl_mem)D.handle(ACCESS_RW), offc, ldc,
                             1, &clq, 0, NULL, NULL);
    else if (type == CV_64FC1)
        status = clblasDgemm(order, transA, transB, M, N, K,
                             alpha, (const cl_mem)A.handle(ACCESS_READ), offa, lda,
                             (const cl_mem)B.handle(ACCESS_READ), offb, ldb,
                             beta, (cl_mem)D.handle(ACCESS_RW), offc, ldc,
                             1, &clq, 0, NULL, NULL);
    else if (type == CV_32FC2)
    {
         cl_float2 alpha_2 = { { (cl_float)alpha, 0 } };
         cl_float2 beta_2  = { { (cl_float)beta, 0 } };
         status = clblasCgemm(order, transA, transB, M, N, K,
                              alpha_2, (const cl_mem)A.handle(ACCESS_READ), offa, lda,
                              (const cl_mem)B.handle(ACCESS_READ), offb, ldb,
                              beta_2, (cl_mem)D.handle(ACCESS_RW), offc, ldc,
                              1, &clq, 0, NULL, NULL);
    }
    else if (type == CV_64FC2)
    {
        cl_double2 alpha_2 = { { alpha, 0 } };
        cl_double2 beta_2  = { { beta, 0 } };
        status = clblasZgemm(order, transA, transB, M, N, K,
                             alpha_2, (const cl_mem)A.handle(ACCESS_READ), offa, lda,
                             (const cl_mem)B.handle(ACCESS_READ), offb, ldb,
                             beta_2, (cl_mem)D.handle(ACCESS_RW), offc, ldc,
                             1, &clq, 0, NULL, NULL);
    }
    else
        CV_Error(Error::StsUnsupportedFormat, "");

    return status == clblasSuccess;
}

#endif

#ifdef HAVE_OPENCL
static bool ocl_gemm( InputArray matA, InputArray matB, double alpha,
                      InputArray matC, double beta, OutputArray matD, int flags )
{
    int type = matA.type();
    int depth = CV_MAT_DEPTH(type);
    int cn = CV_MAT_CN(type);

    CV_CheckTypeEQ(type, matB.type(), "");
    CV_CheckType(type, type == CV_32FC1 || type == CV_64FC1 || type == CV_32FC2 || type == CV_64FC2, "");

    const ocl::Device & dev = ocl::Device::getDefault();
    bool doubleSupport = dev.doubleFPConfig() > 0;

    if (!doubleSupport && depth == CV_64F)
        return false;

    bool haveC = matC.kind() != cv::_InputArray::NONE;
    Size sizeA = matA.size(), sizeB = matB.size(), sizeC = haveC ? matC.size() : Size(0, 0);
    bool atrans = (flags & GEMM_1_T) != 0, btrans = (flags & GEMM_2_T) != 0, ctrans = (flags & GEMM_3_T) != 0;

    if (haveC)
        CV_CheckTypeEQ(type, matC.type(), "");

    Size sizeD(((btrans) ? sizeB.height : sizeB.width),
               ((atrans) ? sizeA.width : sizeA.height));

    if (atrans)
        sizeA = Size(sizeA.height, sizeA.width);
    if (btrans)
        sizeB = Size(sizeB.height, sizeB.width);
    if (haveC && ctrans)
        sizeC = Size(sizeC.height, sizeC.width);

    CV_CheckEQ(sizeA.width, sizeB.height, "");
    if (haveC)
        CV_CheckEQ(sizeC, sizeD, "");

    UMat A = matA.getUMat();
    UMat B = matB.getUMat();

    matD.create(sizeD, type);
    UMat D = matD.getUMat();

    bool isPropagatedC2D = false;  // D content is updated with C / C.t()

    if (dev.intelSubgroupsSupport() && (depth == CV_32F) && cn == 1)
    {
        if (haveC && beta != 0.0)
        {
            ctrans ? transpose(matC, D) : matC.copyTo(D);
            isPropagatedC2D = true;
        }
        else
        {
            beta = 0.0;
        }

        bool res = intel_gpu_gemm(A, matA.size(),
                                  B, matB.size(),
                                  D, sizeD,
                                  alpha,
                                  beta,
                                  atrans, btrans,
                                  isPropagatedC2D);
        if (res)
            return true;
        // fallback on generic OpenCL code
    }

    if (sizeD.width < 8 || sizeD.height < 8)
        return false;

    String opts;

    int wg_size = (int)dev.maxWorkGroupSize();
    int sizeDmin = std::min(sizeD.width, sizeD.height);
    wg_size = std::min(wg_size, sizeDmin * sizeDmin);
    int block_size = (wg_size / (32*cn) < 32) ? (wg_size / (16*cn) < 16) ? (wg_size / (8*cn) < 8) ? 1 : 8 : 16 : 32;

    if (atrans)
        A = A.t();

    if (btrans)
        B = B.t();

    if (haveC && !isPropagatedC2D)
        ctrans ? transpose(matC, D) : matC.copyTo(D);

    int vectorWidths[] = { 4, 4, 2, 2, 1, 4, cn, -1 };
    int kercn = ocl::checkOptimalVectorWidth(vectorWidths, B, D);

    opts += format(" -D T=%s -D T1=%s -D WT=%s -D cn=%d -D kercn=%d -D LOCAL_SIZE=%d%s%s%s",
                      ocl::typeToStr(type), ocl::typeToStr(depth), ocl::typeToStr(CV_MAKETYPE(depth, kercn)),
                      cn, kercn, block_size,
                      (sizeA.width % block_size !=0) ? " -D NO_MULT" : "",
                      haveC ? " -D HAVE_C" : "",
                      doubleSupport ? " -D DOUBLE_SUPPORT" : "");

    ocl::Kernel k("gemm", cv::ocl::core::gemm_oclsrc, opts);
    if (k.empty())
        return false;

    if (depth == CV_64F)
        k.args(ocl::KernelArg::ReadOnlyNoSize(A),
               ocl::KernelArg::ReadOnlyNoSize(B, cn, kercn),
               ocl::KernelArg::ReadWrite(D, cn, kercn),
               sizeA.width, alpha, beta);
    else
        k.args(ocl::KernelArg::ReadOnlyNoSize(A),
               ocl::KernelArg::ReadOnlyNoSize(B, cn, kercn),
               ocl::KernelArg::ReadWrite(D, cn, kercn),
               sizeA.width, (float)alpha, (float)beta);

    size_t globalsize[2] = { (size_t)sizeD.width * cn / kercn, (size_t)sizeD.height};
    size_t localsize[2] = { (size_t)block_size, (size_t)block_size};

    return k.run(2, globalsize, block_size !=1 ? localsize : NULL, false);
}
#endif


namespace hal {

void gemm32f(const float* src1, size_t src1_step, const float* src2, size_t src2_step,
             float alpha, const float* src3, size_t src3_step, float beta, float* dst, size_t dst_step,
             int m_a, int n_a, int n_d, int flags)
{
    CV_INSTRUMENT_REGION();
    CALL_HAL(gemm32f, cv_hal_gemm32f, src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m_a, n_a, n_d, flags)
#ifdef CV_GEMM_BASELINE_ONLY
    CV_CPU_CALL_BASELINE(gemm32f, (src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m_a, n_a, n_d, flags));
#else
    CV_CPU_DISPATCH(gemm32f, (src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m_a, n_a, n_d, flags),
        CV_CPU_DISPATCH_MODES_ALL);
#endif
}

void gemm64f(const double* src1, size_t src1_step, const double* src2, size_t src2_step,
             double alpha, const double* src3, size_t src3_step, double beta, double* dst, size_t dst_step,
             int m_a, int n_a, int n_d, int flags)
{
    CV_INSTRUMENT_REGION();
    CALL_HAL(gemm64f, cv_hal_gemm64f, src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m_a, n_a, n_d, flags)
#ifdef CV_GEMM_BASELINE_ONLY
    CV_CPU_CALL_BASELINE(gemm64f, (src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m_a, n_a, n_d, flags));
#else
    CV_CPU_DISPATCH(gemm64f, (src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m_a, n_a, n_d, flags),
        CV_CPU_DISPATCH_MODES_ALL);
#endif
}

void gemm32fc(const float* src1, size_t src1_step, const float* src2, size_t src2_step,
              float alpha, const float* src3, size_t src3_step, float beta, float* dst, size_t dst_step,
              int m_a, int n_a, int n_d, int flags)
{
    CV_INSTRUMENT_REGION();
    CALL_HAL(gemm32fc, cv_hal_gemm32fc, src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m_a, n_a, n_d, flags)
#ifdef CV_GEMM_BASELINE_ONLY
    CV_CPU_CALL_BASELINE(gemm32fc, (src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m_a, n_a, n_d, flags));
#else
    CV_CPU_DISPATCH(gemm32fc, (src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m_a, n_a, n_d, flags),
        CV_CPU_DISPATCH_MODES_ALL);
#endif
}

void gemm64fc(const double* src1, size_t src1_step, const double* src2, size_t src2_step,
              double alpha, const double* src3, size_t src3_step, double beta, double* dst, size_t dst_step,
              int m_a, int n_a, int n_d, int flags)
{
    CV_INSTRUMENT_REGION();
    CALL_HAL(gemm64fc, cv_hal_gemm64fc, src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m_a, n_a, n_d, flags)
#ifdef CV_GEMM_BASELINE_ONLY
    CV_CPU_CALL_BASELINE(gemm64fc, (src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m_a, n_a, n_d, flags));
#else
    CV_CPU_DISPATCH(gemm64fc, (src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m_a, n_a, n_d, flags),
        CV_CPU_DISPATCH_MODES_ALL);
#endif
}

} // namespace hal

void gemm(InputArray matA, InputArray matB, double alpha,
          InputArray matC, double beta, OutputArray _matD, int flags)
{
#ifdef HAVE_CLAMDBLAS
    CV_OCL_RUN(ocl::haveAmdBlas() && matA.dims() <= 2 && matB.dims() <= 2 && matC.dims() <= 2 && _matD.isUMat() &&
        matA.cols() > 20 && matA.rows() > 20 && matB.cols() > 20, // since it works incorrect for small sizes
        ocl_gemm_amdblas(matA, matB, alpha, matC, beta, _matD, flags))
#endif

#ifdef HAVE_OPENCL
    CV_OCL_RUN(_matD.isUMat() && matA.dims() <= 2 && matB.dims() <= 2 && matC.dims() <= 2,
               ocl_gemm(matA, matB, alpha, matC, beta, _matD, flags))
#endif

    Mat A = matA.getMat(), B = matB.getMat(), C = beta != 0.0 ? matC.getMat() : Mat();
    Size a_size = A.size(), d_size;
    int len = 0, type = A.type();

    CV_Assert_N( type == B.type(), (type == CV_32FC1 || type == CV_64FC1 || type == CV_32FC2 || type == CV_64FC2) );

    switch( flags & (GEMM_1_T|GEMM_2_T) )
    {
    case 0:
        d_size = Size( B.cols, a_size.height );
        len = B.rows;
        CV_Assert( a_size.width == len );
        break;
    case 1:
        d_size = Size( B.cols, a_size.width );
        len = B.rows;
        CV_Assert( a_size.height == len );
        break;
    case 2:
        d_size = Size( B.rows, a_size.height );
        len = B.cols;
        CV_Assert( a_size.width == len );
        break;
    case 3:
        d_size = Size( B.rows, a_size.width );
        len = B.cols;
        CV_Assert( a_size.height == len );
        break;
    }

    if( !C.empty() )
    {
        CV_Assert_N( C.type() == type,
            (((flags&GEMM_3_T) == 0 && C.rows == d_size.height && C.cols == d_size.width) ||
             ((flags&GEMM_3_T) != 0 && C.rows == d_size.width && C.cols == d_size.height)));
    }

    _matD.create( d_size.height, d_size.width, type );
    Mat D = _matD.getMat();
    if( (flags & GEMM_3_T) != 0 && C.data == D.data )
    {
        transpose( C, C );
        flags &= ~GEMM_3_T;
    }

    Mat *DProxyPtr = &D, DProxy;
    if( D.data == A.data || D.data == B.data )
    {
        DProxy = Mat(d_size.height, d_size.width, D.type());
        DProxyPtr = &DProxy;
    }

    if( type == CV_32FC1 )
        hal::gemm32f(A.ptr<float>(), A.step, B.ptr<float>(), B.step, static_cast<float>(alpha),
                     C.ptr<float>(), C.step, static_cast<float>(beta),
                     DProxyPtr->ptr<float>(), DProxyPtr->step,
                     a_size.height, a_size.width, DProxyPtr->cols, flags);
    else if( type == CV_64FC1 )
        hal::gemm64f(A.ptr<double>(), A.step, B.ptr<double>(), B.step, alpha,
                     C.ptr<double>(), C.step, beta,
                     DProxyPtr->ptr<double>(), DProxyPtr->step,
                     a_size.height, a_size.width, DProxyPtr->cols, flags);
    else if( type == CV_32FC2 )
        hal::gemm32fc(A.ptr<float>(), A.step, B.ptr<float>(), B.step, static_cast<float>(alpha),
                      C.ptr<float>(), C.step, static_cast<float>(beta),
                      DProxyPtr->ptr<float>(), DProxyPtr->step,
                      a_size.height, a_size.width, DProxyPtr->cols, flags);
    else
    {
        CV_Assert( type == CV_64FC2 );
        hal::gemm64fc(A.ptr<double>(), A.step, B.ptr<double>(), B.step, alpha,
                      C.ptr<double>(), C.step, beta,
                      D.ptr<double>(), D.step,
                      a_size.height, a_size.width, DProxyPtr->cols, flags);
    }

    if(DProxyPtr != &D)
        DProxyPtr->copyTo(D);
}



/****************************************************************************************\
*                                        Transform                                       *
\****************************************************************************************/

static TransformFunc getTransformFunc(int depth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getTransformFunc, (depth),
        CV_CPU_DISPATCH_MODES_ALL);
}

static TransformFunc getDiagTransformFunc(int depth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getDiagTransformFunc, (depth),
        CV_CPU_DISPATCH_MODES_ALL);
}

void transform(InputArray _src, OutputArray _dst, InputArray _mtx)
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat(), m = _mtx.getMat();
    int depth = src.depth(), scn = src.channels(), dcn = m.rows;
    CV_Assert( scn == m.cols || scn + 1 == m.cols );
    bool isDiag = false;

    _dst.create( src.size(), CV_MAKETYPE(depth, dcn) );
    Mat dst = _dst.getMat();

    if (src.data == dst.data)  // inplace case
    {
        CV_Assert(scn == dcn);
        src = src.clone();  // TODO Add performance warning
    }

    int mtype = depth == CV_32S || depth == CV_64F ? CV_64F : CV_32F;
    AutoBuffer<double> _mbuf;
    double* mbuf;

    if( !m.isContinuous() || m.type() != mtype || m.cols != scn + 1 )
    {
        _mbuf.allocate(dcn*(scn+1));
        mbuf = _mbuf.data();
        Mat tmp(dcn, scn+1, mtype, mbuf);
        memset(tmp.ptr(), 0, tmp.total()*tmp.elemSize());
        if( m.cols == scn+1 )
            m.convertTo(tmp, mtype);
        else
        {
            Mat tmppart = tmp.colRange(0, m.cols);
            m.convertTo(tmppart, mtype);
        }
        m = tmp;
    }
    else
        mbuf = m.ptr<double>();

    if( scn == dcn )
    {
        int i, j;
        double eps = mtype == CV_32F ? FLT_EPSILON : DBL_EPSILON;

        if( scn == 1 )
        {
            double alpha, beta;
            if( mtype == CV_32F )
                alpha = m.at<float>(0), beta = m.at<float>(1);
            else
                alpha = m.at<double>(0), beta = m.at<double>(1);
            src.convertTo(dst, dst.type(), alpha, beta);
            return;
        }

        for( i = 0, isDiag = true; isDiag && i < scn; i++ )
        {
            for( j = 0; isDiag && j < scn; j++ )
            {
                double v = mtype == CV_32F ? m.at<float>(i, j) : m.at<double>(i, j);
                if( i != j && fabs(v) > eps )
                    isDiag = false;
            }
        }
    }

    TransformFunc func = isDiag ? getDiagTransformFunc(depth): getTransformFunc(depth);
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &dst, 0};
    uchar* ptrs[2] = {};
    NAryMatIterator it(arrays, ptrs);
    size_t i, total = it.size;

    for( i = 0; i < it.nplanes; i++, ++it )
        func( ptrs[0], ptrs[1], (uchar*)mbuf, (int)total, scn, dcn );
}



/****************************************************************************************\
*                                  Perspective Transform                                 *
\****************************************************************************************/

static TransformFunc getPerspectiveTransform(int depth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getPerspectiveTransform, (depth),
        CV_CPU_DISPATCH_MODES_ALL);
}

void perspectiveTransform(InputArray _src, OutputArray _dst, InputArray _mtx)
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat(), m = _mtx.getMat();
    int depth = src.depth(), scn = src.channels(), dcn = m.rows-1;
    CV_Assert( scn + 1 == m.cols );
    CV_Assert( depth == CV_32F || depth == CV_64F );

    _dst.create( src.size(), CV_MAKETYPE(depth, dcn) );
    Mat dst = _dst.getMat();

    const int mtype = CV_64F;
    AutoBuffer<double> _mbuf;
    double* mbuf = m.ptr<double>();

    if( !m.isContinuous() || m.type() != mtype )
    {
        _mbuf.allocate((dcn+1)*(scn+1));
        mbuf = _mbuf.data();
        Mat tmp(dcn+1, scn+1, mtype, mbuf);
        m.convertTo(tmp, mtype);
        m = tmp;
    }

    TransformFunc func = getPerspectiveTransform(depth);
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &dst, 0};
    uchar* ptrs[2] = {};
    NAryMatIterator it(arrays, ptrs);
    size_t i, total = it.size;

    for( i = 0; i < it.nplanes; i++, ++it )
        func( ptrs[0], ptrs[1], (uchar*)mbuf, (int)total, scn, dcn );
}

/****************************************************************************************\
*                                       ScaleAdd                                         *
\****************************************************************************************/

#ifdef HAVE_OPENCL

static bool ocl_scaleAdd( InputArray _src1, double alpha, InputArray _src2, OutputArray _dst, int type )
{
    const ocl::Device & d = ocl::Device::getDefault();

    bool doubleSupport = d.doubleFPConfig() > 0;
    Size size = _src1.size();
    int depth = CV_MAT_DEPTH(type);
    if ( (!doubleSupport && depth == CV_64F) || size != _src2.size() )
        return false;

    _dst.create(size, type);
    int cn = CV_MAT_CN(type), wdepth = std::max(depth, CV_32F);
    int kercn = ocl::predictOptimalVectorWidthMax(_src1, _src2, _dst),
        rowsPerWI = d.isIntel() ? 4 : 1;

    char cvt[2][50];
    ocl::Kernel k("KF", ocl::core::arithm_oclsrc,
                  format("-D OP_SCALE_ADD -D BINARY_OP -D dstT=%s -D DEPTH_dst=%d -D workT=%s -D convertToWT1=%s"
                         " -D srcT1=dstT -D srcT2=dstT -D convertToDT=%s -D workT1=%s"
                         " -D wdepth=%d%s -D rowsPerWI=%d",
                         ocl::typeToStr(CV_MAKE_TYPE(depth, kercn)), depth,
                         ocl::typeToStr(CV_MAKE_TYPE(wdepth, kercn)),
                         ocl::convertTypeStr(depth, wdepth, kercn, cvt[0], sizeof(cvt[0])),
                         ocl::convertTypeStr(wdepth, depth, kercn, cvt[1], sizeof(cvt[1])),
                         ocl::typeToStr(wdepth), wdepth,
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "", rowsPerWI));
    if (k.empty())
        return false;

    UMat src1 = _src1.getUMat(), src2 = _src2.getUMat(), dst = _dst.getUMat();

    ocl::KernelArg src1arg = ocl::KernelArg::ReadOnlyNoSize(src1),
            src2arg = ocl::KernelArg::ReadOnlyNoSize(src2),
            dstarg = ocl::KernelArg::WriteOnly(dst, cn, kercn);

    if (wdepth == CV_32F)
        k.args(src1arg, src2arg, dstarg, (float)alpha);
    else
        k.args(src1arg, src2arg, dstarg, alpha);

    size_t globalsize[2] = { (size_t)dst.cols * cn / kercn, ((size_t)dst.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

static ScaleAddFunc getScaleAddFunc(int depth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getScaleAddFunc, (depth),
        CV_CPU_DISPATCH_MODES_ALL);
}

void scaleAdd(InputArray _src1, double alpha, InputArray _src2, OutputArray _dst)
{
    CV_INSTRUMENT_REGION();

    int type = _src1.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert( type == _src2.type() );

    CV_OCL_RUN(_src1.dims() <= 2 && _src2.dims() <= 2 && _dst.isUMat(),
            ocl_scaleAdd(_src1, alpha, _src2, _dst, type))

    if( depth < CV_32F )
    {
        addWeighted(_src1, alpha, _src2, 1, 0, _dst, depth);
        return;
    }

    Mat src1 = _src1.getMat(), src2 = _src2.getMat();
    CV_Assert(src1.size == src2.size);

    _dst.create(src1.dims, src1.size, type);
    Mat dst = _dst.getMat();

    float falpha = (float)alpha;
    void* palpha = depth == CV_32F ? (void*)&falpha : (void*)&alpha;

    ScaleAddFunc func = getScaleAddFunc(depth);
    CV_Assert(func);

    if (src1.isContinuous() && src2.isContinuous() && dst.isContinuous())
    {
        size_t len = src1.total()*cn;
        func(src1.ptr(), src2.ptr(), dst.ptr(), (int)len, palpha);
        return;
    }

    const Mat* arrays[] = {&src1, &src2, &dst, 0};
    uchar* ptrs[3] = {};
    NAryMatIterator it(arrays, ptrs);
    size_t i, len = it.size*cn;

    for( i = 0; i < it.nplanes; i++, ++it )
        func( ptrs[0], ptrs[1], ptrs[2], (int)len, palpha );
}

/****************************************************************************************\
*                                 Covariation Matrix                                     *
\****************************************************************************************/

void calcCovarMatrix( const Mat* data, int nsamples, Mat& covar, Mat& _mean, int flags, int ctype )
{
    CV_INSTRUMENT_REGION();

    CV_Assert_N( data, nsamples > 0 );
    Size size = data[0].size();
    int sz = size.width * size.height, esz = (int)data[0].elemSize();
    int type = data[0].type();
    Mat mean;
    ctype = std::max(std::max(CV_MAT_DEPTH(ctype >= 0 ? ctype : type), _mean.depth()), CV_32F);

    if( (flags & cv::COVAR_USE_AVG) != 0 )
    {
        CV_Assert( _mean.size() == size );
        if( _mean.isContinuous() && _mean.type() == ctype )
            mean = _mean.reshape(1, 1);
        else
        {
            _mean.convertTo(mean, ctype);
            mean = mean.reshape(1, 1);
        }
    }

    Mat _data(nsamples, sz, type);

    for( int i = 0; i < nsamples; i++ )
    {
        CV_Assert_N( data[i].size() == size, data[i].type() == type );
        if( data[i].isContinuous() )
            memcpy( _data.ptr(i), data[i].ptr(), sz*esz );
        else
        {
            Mat dataRow(size.height, size.width, type, _data.ptr(i));
            data[i].copyTo(dataRow);
        }
    }

    calcCovarMatrix( _data, covar, mean, (flags & ~(cv::COVAR_ROWS|cv::COVAR_COLS)) | cv::COVAR_ROWS, ctype );
    if( (flags & cv::COVAR_USE_AVG) == 0 )
        _mean = mean.reshape(1, size.height);
}

void calcCovarMatrix( InputArray _src, OutputArray _covar, InputOutputArray _mean, int flags, int ctype )
{
    CV_INSTRUMENT_REGION();

    if(_src.kind() == _InputArray::STD_VECTOR_MAT || _src.kind() == _InputArray::STD_ARRAY_MAT)
    {
        std::vector<cv::Mat> src;
        _src.getMatVector(src);

        CV_Assert( src.size() > 0 );

        Size size = src[0].size();
        int type = src[0].type();

        ctype = std::max(std::max(CV_MAT_DEPTH(ctype >= 0 ? ctype : type), _mean.depth()), CV_32F);

        Mat _data(static_cast<int>(src.size()), size.area(), type);

        int i = 0;
        for(std::vector<cv::Mat>::iterator each = src.begin(); each != src.end(); ++each, ++i )
        {
            CV_Assert_N( (*each).size() == size, (*each).type() == type );
            Mat dataRow(size.height, size.width, type, _data.ptr(i));
            (*each).copyTo(dataRow);
        }

        Mat mean;
        if( (flags & cv::COVAR_USE_AVG) != 0 )
        {
            CV_Assert( _mean.size() == size );

            if( mean.type() != ctype )
            {
                mean = _mean.getMat();
                _mean.create(mean.size(), ctype);
                Mat tmp = _mean.getMat();
                mean.convertTo(tmp, ctype);
                mean = tmp;
            }

            mean = _mean.getMat().reshape(1, 1);
        }

        calcCovarMatrix( _data, _covar, mean, (flags & ~(cv::COVAR_ROWS|cv::COVAR_COLS)) | cv::COVAR_ROWS, ctype );

        if( (flags & cv::COVAR_USE_AVG) == 0 )
        {
            mean = mean.reshape(1, size.height);
            mean.copyTo(_mean);
        }
        return;
    }

    Mat data = _src.getMat(), mean;
    CV_Assert( ((flags & cv::COVAR_ROWS) != 0) ^ ((flags & cv::COVAR_COLS) != 0) );
    bool takeRows = (flags & cv::COVAR_ROWS) != 0;
    int type = data.type();
    int nsamples = takeRows ? data.rows : data.cols;
    CV_Assert( nsamples > 0 );
    Size size = takeRows ? Size(data.cols, 1) : Size(1, data.rows);

    if( (flags & cv::COVAR_USE_AVG) != 0 )
    {
        mean = _mean.getMat();
        ctype = std::max(std::max(CV_MAT_DEPTH(ctype >= 0 ? ctype : type), mean.depth()), CV_32F);
        CV_Assert( mean.size() == size );
        if( mean.type() != ctype )
        {
            _mean.create(mean.size(), ctype);
            Mat tmp = _mean.getMat();
            mean.convertTo(tmp, ctype);
            mean = tmp;
        }
    }
    else
    {
        ctype = std::max(CV_MAT_DEPTH(ctype >= 0 ? ctype : type), CV_32F);
        reduce( _src, _mean, takeRows ? 0 : 1, REDUCE_AVG, ctype );
        mean = _mean.getMat();
    }

    mulTransposed( data, _covar, ((flags & cv::COVAR_NORMAL) == 0) ^ takeRows,
        mean, (flags & cv::COVAR_SCALE) != 0 ? 1./nsamples : 1, ctype );
}



/****************************************************************************************\
*                                        Mahalanobis                                     *
\****************************************************************************************/

static MahalanobisImplFunc getMahalanobisImplFunc(int depth)
{
#ifdef CV_MAHALANOBIS_BASELINE_ONLY
    CV_CPU_CALL_BASELINE(getMahalanobisImplFunc, (depth));
#else
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getMahalanobisImplFunc, (depth),
        CV_CPU_DISPATCH_MODES_ALL);
#endif
}


double Mahalanobis(InputArray _v1, InputArray _v2, InputArray _icovar)
{
    CV_INSTRUMENT_REGION();

    Mat v1 = _v1.getMat(), v2 = _v2.getMat(), icovar = _icovar.getMat();
    int type = v1.type(), depth = v1.depth();
    Size sz = v1.size();
    int len = sz.width*sz.height*v1.channels();
    AutoBuffer<double> buf(len);

    CV_Assert_N( type == v2.type(), type == icovar.type(),
        sz == v2.size(), len == icovar.rows && len == icovar.cols );

    sz.width *= v1.channels();
    if( v1.isContinuous() && v2.isContinuous() )
    {
        sz.width *= sz.height;
        sz.height = 1;
    }

    MahalanobisImplFunc func = getMahalanobisImplFunc(depth);
    CV_Assert(func);

    double result = func(v1, v2, icovar, buf.data(), len);
    return std::sqrt(result);
}



/****************************************************************************************\
*                                        MulTransposed                                   *
\****************************************************************************************/

static MulTransposedFunc getMulTransposedFunc(int stype, int dtype, bool ata)
{
#ifdef CV_MULTRANSPOSED_BASELINE_ONLY
    CV_CPU_CALL_BASELINE(getMulTransposedFunc, (stype, dtype, ata));
#else
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getMulTransposedFunc, (stype, dtype, ata),
        CV_CPU_DISPATCH_MODES_ALL);
#endif
}

void mulTransposed(InputArray _src, OutputArray _dst, bool ata,
                   InputArray _delta, double scale, int dtype)
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat(), delta = _delta.getMat();
    const int gemm_level = 100; // boundary above which GEMM is faster.
    int stype = src.type();
    dtype = std::max(std::max(CV_MAT_DEPTH(dtype >= 0 ? dtype : stype), delta.depth()), CV_32F);
    CV_Assert( src.channels() == 1 );

    if( !delta.empty() )
    {
        CV_Assert_N( delta.channels() == 1,
            (delta.rows == src.rows || delta.rows == 1),
            (delta.cols == src.cols || delta.cols == 1));
        if( delta.type() != dtype )
            delta.convertTo(delta, dtype);
    }

    int dsize = ata ? src.cols : src.rows;
    _dst.create( dsize, dsize, dtype );
    Mat dst = _dst.getMat();

    if( src.data == dst.data || (stype == dtype &&
        (dst.cols >= gemm_level && dst.rows >= gemm_level &&
         src.cols >= gemm_level && src.rows >= gemm_level)))
    {
        Mat src2;
        const Mat* tsrc = &src;
        if( !delta.empty() )
        {
            if( delta.size() == src.size() )
                subtract( src, delta, src2 );
            else
            {
                repeat(delta, src.rows/delta.rows, src.cols/delta.cols, src2);
                subtract( src, src2, src2 );
            }
            tsrc = &src2;
        }
        gemm( *tsrc, *tsrc, scale, Mat(), 0, dst, ata ? GEMM_1_T : GEMM_2_T );
    }
    else
    {
        MulTransposedFunc func = getMulTransposedFunc(stype, dtype, ata);
        if( !func )
            CV_Error( cv::Error::StsUnsupportedFormat, "" );

        func( src, dst, delta, scale );
        completeSymm( dst, false );
    }
}

/****************************************************************************************\
*                                      Dot Product                                       *
\****************************************************************************************/

static double dotProd_8u(const uchar* src1, const uchar* src2, int len)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(dotProd_8u, (src1, src2, len),
        CV_CPU_DISPATCH_MODES_ALL);
}
static double dotProd_8s(const schar* src1, const schar* src2, int len)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(dotProd_8s, (src1, src2, len),
        CV_CPU_DISPATCH_MODES_ALL);
}
static double dotProd_16u(const ushort* src1, const ushort* src2, int len)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(dotProd_16u, (src1, src2, len),
        CV_CPU_DISPATCH_MODES_ALL);
}
static double dotProd_16s(const short* src1, const short* src2, int len)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(dotProd_16s, (src1, src2, len),
        CV_CPU_DISPATCH_MODES_ALL);
}
static double dotProd_32s(const int* src1, const int* src2, int len)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(dotProd_32s, (src1, src2, len),
        CV_CPU_DISPATCH_MODES_ALL);
}
static double dotProd_32f(const float* src1, const float* src2, int len)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(dotProd_32f, (src1, src2, len),
        CV_CPU_DISPATCH_MODES_ALL);
}
static double dotProd_64f(const double* src1, const double* src2, int len)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(dotProd_64f, (src1, src2, len),
        CV_CPU_DISPATCH_MODES_ALL);
}

typedef double (*DotProdFunc)(const uchar* src1, const uchar* src2, int len);

static DotProdFunc getDotProdFunc(int depth)
{
    static DotProdFunc dotProdTab[CV_DEPTH_MAX] =
    {
        (DotProdFunc)GET_OPTIMIZED(dotProd_8u), (DotProdFunc)GET_OPTIMIZED(dotProd_8s),
        (DotProdFunc)dotProd_16u, (DotProdFunc)dotProd_16s,
        (DotProdFunc)dotProd_32s, (DotProdFunc)GET_OPTIMIZED(dotProd_32f),
        (DotProdFunc)dotProd_64f, 0
    };

    return dotProdTab[depth];
}

double Mat::dot(InputArray _mat) const
{
    CV_INSTRUMENT_REGION();

    Mat mat = _mat.getMat();
    CV_Assert_N( mat.type() == type(), mat.size == size);

    int cn = channels();
    if (this->dims <= 2)
    {
        double product = 0;
        CALL_HAL_RET(dotProduct, cv_hal_dotProduct, product, this->data, this->step, mat.data, mat.step,
                     this->cols * cn, this->rows, this->depth());
    }

    DotProdFunc func = getDotProdFunc(depth());
    CV_Assert(func != 0 );

    if( isContinuous() && mat.isContinuous() )
    {
        size_t len = total()*cn;
        if( len == (size_t)(int)len )
            return func(data, mat.data, (int)len);
    }

    const Mat* arrays[] = {this, &mat, 0};
    uchar* ptrs[2] = {};
    NAryMatIterator it(arrays, ptrs);
    int len = (int)(it.size*cn);
    double r = 0;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        r += func( ptrs[0], ptrs[1], len );

    return r;
}


#ifdef HAVE_OPENCL

static bool ocl_dot( InputArray _src1, InputArray _src2, double & res )
{
    UMat src1 = _src1.getUMat().reshape(1), src2 = _src2.getUMat().reshape(1);

    int type = src1.type(), depth = CV_MAT_DEPTH(type),
            kercn = ocl::predictOptimalVectorWidth(src1, src2);
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;

    if ( !doubleSupport && depth == CV_64F )
        return false;

    int dbsize = ocl::Device::getDefault().maxComputeUnits();
    size_t wgs = ocl::Device::getDefault().maxWorkGroupSize();
    int ddepth = std::max(CV_32F, depth);

    int wgs2_aligned = 1;
    while (wgs2_aligned < (int)wgs)
        wgs2_aligned <<= 1;
    wgs2_aligned >>= 1;

    char cvt[50];
    ocl::Kernel k("reduce", ocl::core::reduce_oclsrc,
                  format("-D srcT=%s -D srcT1=%s -D dstT=%s -D dstTK=%s -D ddepth=%d -D convertToDT=%s -D OP_DOT "
                         "-D WGS=%d -D WGS2_ALIGNED=%d%s%s%s -D kercn=%d",
                         ocl::typeToStr(CV_MAKE_TYPE(depth, kercn)), ocl::typeToStr(depth),
                         ocl::typeToStr(ddepth), ocl::typeToStr(CV_MAKE_TYPE(ddepth, kercn)),
                         ddepth, ocl::convertTypeStr(depth, ddepth, kercn, cvt, sizeof(cvt)),
                         (int)wgs, wgs2_aligned, doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                         _src1.isContinuous() ? " -D HAVE_SRC_CONT" : "",
                         _src2.isContinuous() ? " -D HAVE_SRC2_CONT" : "", kercn));
    if (k.empty())
        return false;

    UMat db(1, dbsize, ddepth);

    ocl::KernelArg src1arg = ocl::KernelArg::ReadOnlyNoSize(src1),
            src2arg = ocl::KernelArg::ReadOnlyNoSize(src2),
            dbarg = ocl::KernelArg::PtrWriteOnly(db);

    k.args(src1arg, src1.cols, (int)src1.total(), dbsize, dbarg, src2arg);

    size_t globalsize = dbsize * wgs;
    if (k.run(1, &globalsize, &wgs, true))
    {
        res = sum(db.getMat(ACCESS_READ))[0];
        return true;
    }
    return false;
}

#endif

double UMat::dot(InputArray m) const
{
    CV_INSTRUMENT_REGION();

    CV_Assert(m.sameSize(*this) && m.type() == type());

#ifdef HAVE_OPENCL
    double r = 0;
    CV_OCL_RUN_(dims <= 2, ocl_dot(*this, m, r), r)
#endif

    return getMat(ACCESS_READ).dot(m);
}

}  // namespace cv::


#ifndef OPENCV_EXCLUDE_C_API
/****************************************************************************************\
*                                    Earlier API                                         *
\****************************************************************************************/

CV_IMPL void cvGEMM( const CvArr* Aarr, const CvArr* Barr, double alpha,
                     const CvArr* Carr, double beta, CvArr* Darr, int flags )
{
    cv::Mat A = cv::cvarrToMat(Aarr), B = cv::cvarrToMat(Barr);
    cv::Mat C, D = cv::cvarrToMat(Darr);

    if( Carr )
        C = cv::cvarrToMat(Carr);

    CV_Assert_N( (D.rows == ((flags & CV_GEMM_A_T) == 0 ? A.rows : A.cols)),
               (D.cols == ((flags & CV_GEMM_B_T) == 0 ? B.cols : B.rows)),
               D.type() == A.type() );

    gemm( A, B, alpha, C, beta, D, flags );
}


CV_IMPL void
cvTransform( const CvArr* srcarr, CvArr* dstarr,
             const CvMat* transmat, const CvMat* shiftvec )
{
    cv::Mat m = cv::cvarrToMat(transmat), src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    if( shiftvec )
    {
        cv::Mat v = cv::cvarrToMat(shiftvec).reshape(1,m.rows),
            _m(m.rows, m.cols + 1, m.type()), m1 = _m.colRange(0,m.cols), v1 = _m.col(m.cols);
        m.convertTo(m1, m1.type());
        v.convertTo(v1, v1.type());
        m = _m;
    }

    CV_Assert_N( dst.depth() == src.depth(), dst.channels() == m.rows );
    cv::transform( src, dst, m );
}


CV_IMPL void
cvPerspectiveTransform( const CvArr* srcarr, CvArr* dstarr, const CvMat* mat )
{
    cv::Mat m = cv::cvarrToMat(mat), src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert_N( dst.type() == src.type(), dst.channels() == m.rows-1 );
    cv::perspectiveTransform( src, dst, m );
}


CV_IMPL void cvScaleAdd( const CvArr* srcarr1, CvScalar scale,
                         const CvArr* srcarr2, CvArr* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);

    CV_Assert_N( src1.size == dst.size, src1.type() == dst.type() );
    cv::scaleAdd( src1, scale.val[0], cv::cvarrToMat(srcarr2), dst );
}


CV_IMPL void
cvCalcCovarMatrix( const CvArr** vecarr, int count,
                   CvArr* covarr, CvArr* avgarr, int flags )
{
    cv::Mat cov0 = cv::cvarrToMat(covarr), cov = cov0, mean0, mean;
    CV_Assert_N( vecarr != 0, count >= 1 );

    if( avgarr )
        mean = mean0 = cv::cvarrToMat(avgarr);

    if( (flags & cv::COVAR_COLS) != 0 || (flags & cv::COVAR_ROWS) != 0 )
    {

        cv::Mat data = cv::cvarrToMat(vecarr[0]);
        cv::calcCovarMatrix( data, cov, mean, flags, cov.type() );
    }
    else
    {
        std::vector<cv::Mat> data(count);
        for( int i = 0; i < count; i++ )
            data[i] = cv::cvarrToMat(vecarr[i]);
        cv::calcCovarMatrix( &data[0], count, cov, mean, flags, cov.type() );
    }

    if( mean.data != mean0.data && mean0.data )
        mean.convertTo(mean0, mean0.type());

    if( cov.data != cov0.data )
        cov.convertTo(cov0, cov0.type());
}


CV_IMPL double
cvMahalanobis( const CvArr* srcAarr, const CvArr* srcBarr, const CvArr* matarr )
{
    return cv::Mahalanobis(cv::cvarrToMat(srcAarr),
        cv::cvarrToMat(srcBarr), cv::cvarrToMat(matarr));
}

CV_IMPL void
cvMulTransposed( const CvArr* srcarr, CvArr* dstarr,
                 int order, const CvArr* deltaarr, double scale )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst0 = cv::cvarrToMat(dstarr), dst = dst0, delta;
    if( deltaarr )
        delta = cv::cvarrToMat(deltaarr);
    cv::mulTransposed( src, dst, order != 0, delta, scale, dst.type());
    if( dst.data != dst0.data )
        dst.convertTo(dst0, dst0.type());
}

CV_IMPL double cvDotProduct( const CvArr* srcAarr, const CvArr* srcBarr )
{
    return cv::cvarrToMat(srcAarr).dot(cv::cvarrToMat(srcBarr));
}


CV_IMPL void
cvCalcPCA( const CvArr* data_arr, CvArr* avg_arr, CvArr* eigenvals, CvArr* eigenvects, int flags )
{
    cv::Mat data = cv::cvarrToMat(data_arr), mean0 = cv::cvarrToMat(avg_arr);
    cv::Mat evals0 = cv::cvarrToMat(eigenvals), evects0 = cv::cvarrToMat(eigenvects);
    cv::Mat mean = mean0, evals = evals0, evects = evects0;

    cv::PCA pca;
    pca.mean = mean;
    pca.eigenvalues = evals;
    pca.eigenvectors = evects;

    pca(data, (flags & CV_PCA_USE_AVG) ? mean : cv::Mat(),
        flags, !evals.empty() ? evals.rows + evals.cols - 1 : 0);

    if( pca.mean.size() == mean.size() )
        pca.mean.convertTo( mean, mean.type() );
    else
    {
        cv::Mat temp; pca.mean.convertTo( temp, mean.type() );
        transpose( temp, mean );
    }

    evals = pca.eigenvalues;
    evects = pca.eigenvectors;
    int ecount0 = evals0.cols + evals0.rows - 1;
    int ecount = evals.cols + evals.rows - 1;

    CV_Assert_N( (evals0.cols == 1 || evals0.rows == 1),
                ecount0 <= ecount,
                evects0.cols == evects.cols,
                evects0.rows == ecount0 );

    cv::Mat temp = evals0;
    if( evals.rows == 1 )
        evals.colRange(0, ecount0).convertTo(temp, evals0.type());
    else
        evals.rowRange(0, ecount0).convertTo(temp, evals0.type());
    if( temp.data != evals0.data )
        transpose(temp, evals0);
    evects.rowRange(0, ecount0).convertTo( evects0, evects0.type() );

    // otherwise some datatype's or size's were incorrect, so the output arrays have been reallocated
    CV_Assert( mean0.data == mean.data );
}


CV_IMPL void
cvProjectPCA( const CvArr* data_arr, const CvArr* avg_arr,
              const CvArr* eigenvects, CvArr* result_arr )
{
    cv::Mat data = cv::cvarrToMat(data_arr), mean = cv::cvarrToMat(avg_arr);
    cv::Mat evects = cv::cvarrToMat(eigenvects), dst0 = cv::cvarrToMat(result_arr), dst = dst0;

    cv::PCA pca;
    pca.mean = mean;
    int n;
    if( mean.rows == 1 )
    {
        CV_Assert_N(dst.cols <= evects.rows, dst.rows == data.rows);
        n = dst.cols;
    }
    else
    {
        CV_Assert_N(dst.rows <= evects.rows, dst.cols == data.cols);
        n = dst.rows;
    }
    pca.eigenvectors = evects.rowRange(0, n);

    cv::Mat result = pca.project(data);
    if( result.cols != dst.cols )
        result = result.reshape(1, 1);
    result.convertTo(dst, dst.type());

    CV_Assert(dst0.data == dst.data);
}


CV_IMPL void
cvBackProjectPCA( const CvArr* proj_arr, const CvArr* avg_arr,
                  const CvArr* eigenvects, CvArr* result_arr )
{
    cv::Mat data = cv::cvarrToMat(proj_arr), mean = cv::cvarrToMat(avg_arr);
    cv::Mat evects = cv::cvarrToMat(eigenvects), dst0 = cv::cvarrToMat(result_arr), dst = dst0;

    cv::PCA pca;
    pca.mean = mean;
    int n;
    if( mean.rows == 1 )
    {
        CV_Assert_N(data.cols <= evects.rows, dst.rows == data.rows);
        n = data.cols;
    }
    else
    {
        CV_Assert_N(data.rows <= evects.rows, dst.cols == data.cols);
        n = data.rows;
    }
    pca.eigenvectors = evects.rowRange(0, n);

    cv::Mat result = pca.backProject(data);
    result.convertTo(dst, dst.type());

    CV_Assert(dst0.data == dst.data);
}

#endif  // OPENCV_EXCLUDE_C_API

/* End of file. */
