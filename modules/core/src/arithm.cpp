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

/* ////////////////////////////////////////////////////////////////////
//
//  Arithmetic and logical operations: +, -, *, /, &, |, ^, ~, abs ...
//
// */

#include "precomp.hpp"
#include "arithm_expr.hpp"   // the new element-wise engine (cv::ew)
#include "opencl_kernels_core.hpp"

namespace cv
{

/****************************************************************************************\
*                                   logical operations                                   *
\****************************************************************************************/

enum { OCL_OP_ADD=0, OCL_OP_SUB=1, OCL_OP_RSUB=2, OCL_OP_ABSDIFF=3, OCL_OP_MUL=4,
       OCL_OP_MUL_SCALE=5, OCL_OP_DIV_SCALE=6, OCL_OP_RECIP_SCALE=7, OCL_OP_ADDW=8,
       OCL_OP_AND=9, OCL_OP_OR=10, OCL_OP_XOR=11, OCL_OP_NOT=12, OCL_OP_MIN=13, OCL_OP_MAX=14,
       OCL_OP_RDIV_SCALE=15 };

// The unified entry point for every element-wise binary op (empty + UMat/OpenCL + CPU engine). Defined
// lower down (near cv::add); forward-declared here so cv::min/cv::max (above it) can use it too.
static void arithm_op(ew::TOp op, InputArray src1, InputArray src2, OutputArray dst,
                      InputArray mask, int dtype, int oclop, bool muldiv, const Scalar& params = Scalar(1));

#ifdef HAVE_OPENCL

static const char* oclop2str[] = { "OP_ADD", "OP_SUB", "OP_RSUB", "OP_ABSDIFF",
    "OP_MUL", "OP_MUL_SCALE", "OP_DIV_SCALE", "OP_RECIP_SCALE",
    "OP_ADDW", "OP_AND", "OP_OR", "OP_XOR", "OP_NOT", "OP_MIN", "OP_MAX", "OP_RDIV_SCALE", 0 };

static bool ocl_binary_op(InputArray _src1, InputArray _src2, OutputArray _dst,
                          InputArray _mask, bool bitwise, int oclop, bool haveScalar )
{
    bool haveMask = !_mask.empty();
    int srctype = _src1.type();
    int srcdepth = CV_MAT_DEPTH(srctype);
    int cn = CV_MAT_CN(srctype);

    const ocl::Device d = ocl::Device::getDefault();
    bool doubleSupport = d.doubleFPConfig() > 0;
    if( oclop < 0 || ((haveMask || haveScalar) && cn > 4) ||
            (!doubleSupport && srcdepth == CV_64F && !bitwise))
        return false;

    char opts[1024];
    int kercn = haveMask || haveScalar ? cn : ocl::predictOptimalVectorWidth(_src1, _src2, _dst);
    int scalarcn = kercn == 3 ? 4 : kercn;
    int rowsPerWI = d.isIntel() ? 4 : 1;

    const int dstDepth = srcdepth;
    const int dstType = CV_MAKETYPE(dstDepth, kercn);
    const int dstType1 = CV_MAKETYPE(dstDepth, 1);
    const int scalarType = CV_MAKETYPE(srcdepth, scalarcn);

    snprintf(opts, sizeof(opts), "-D %s%s -D %s%s -D dstT=%s -D DEPTH_dst=%d -D dstT_C1=%s -D workST=%s -D cn=%d -D rowsPerWI=%d",
            haveMask ? "MASK_" : "", haveScalar ? "UNARY_OP" : "BINARY_OP", oclop2str[oclop],
            doubleSupport ? " -D DOUBLE_SUPPORT" : "",
            bitwise ? ocl::memopTypeToStr(dstType) : ocl::typeToStr(dstType),
            dstDepth,
            bitwise ? ocl::memopTypeToStr(dstType1) : ocl::typeToStr(dstType1),
            bitwise ? ocl::memopTypeToStr(scalarType) : ocl::typeToStr(scalarType),
            kercn, rowsPerWI);

    ocl::Kernel k("KF", ocl::core::arithm_oclsrc, opts);
    if (k.empty())
        return false;

    UMat src1 = _src1.getUMat(), src2;
    UMat dst = _dst.getUMat(), mask = _mask.getUMat();

    ocl::KernelArg src1arg = ocl::KernelArg::ReadOnlyNoSize(src1, cn, kercn);
    ocl::KernelArg dstarg = haveMask ? ocl::KernelArg::ReadWrite(dst, cn, kercn) :
                                       ocl::KernelArg::WriteOnly(dst, cn, kercn);
    ocl::KernelArg maskarg = ocl::KernelArg::ReadOnlyNoSize(mask, 1);

    if( haveScalar )
    {
        size_t esz = CV_ELEM_SIZE1(srctype)*scalarcn;
        double buf[4] = {0,0,0,0};

        if( oclop != OCL_OP_NOT )
        {
            Mat src2sc = _src2.getMat();
            convertAndUnrollScalar(src2sc, srctype, (uchar*)buf, 1);
        }

        ocl::KernelArg scalararg = ocl::KernelArg(ocl::KernelArg::CONSTANT, 0, 0, 0, buf, esz);

        if( !haveMask )
            k.args(src1arg, dstarg, scalararg);
        else
            k.args(src1arg, maskarg, dstarg, scalararg);
    }
    else
    {
        src2 = _src2.getUMat();
        ocl::KernelArg src2arg = ocl::KernelArg::ReadOnlyNoSize(src2, cn, kercn);

        if( !haveMask )
            k.args(src1arg, src2arg, dstarg);
        else
            k.args(src1arg, src2arg, maskarg, dstarg);
    }

    size_t globalsize[] = { (size_t)src1.cols * cn / kercn, ((size_t)src1.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, 0, false);
}

#endif

static void binary_op( InputArray _src1, InputArray _src2, OutputArray _dst,
                       InputArray _mask, const BinaryFuncC* tab,
                       bool bitwise, int oclop )
{
    const _InputArray *psrc1 = &_src1, *psrc2 = &_src2;
    _InputArray::KindFlag kind1 = psrc1->kind(), kind2 = psrc2->kind();
    int type1 = psrc1->type(), depth1 = CV_MAT_DEPTH(type1), cn = CV_MAT_CN(type1);
    int type2 = psrc2->type(), depth2 = CV_MAT_DEPTH(type2), cn2 = CV_MAT_CN(type2);
    int dims1 = psrc1->dims(), dims2 = psrc2->dims();
    Size sz1 = dims1 <= 2 ? psrc1->size() : Size();
    Size sz2 = dims2 <= 2 ? psrc2->size() : Size();
#ifdef HAVE_OPENCL
    bool use_opencl = (kind1 == _InputArray::UMAT || kind2 == _InputArray::UMAT) &&
            dims1 <= 2 && dims2 <= 2;
#endif
    bool haveMask = !_mask.empty(), haveScalar = false;
    BinaryFuncC func;

    if( dims1 <= 2 && dims2 <= 2 && kind1 == kind2 && sz1 == sz2 && type1 == type2 && !haveMask )
    {
        _dst.createSameSize(*psrc1, type1);
        CV_OCL_RUN(use_opencl,
                   ocl_binary_op(*psrc1, *psrc2, _dst, _mask, bitwise, oclop, false))

        if( bitwise )
        {
            func = *tab;
            cn = (int)CV_ELEM_SIZE(type1);
        }
        else
        {
            func = tab[depth1];
        }
        CV_Assert(func);

        Mat src1 = psrc1->getMat(), src2 = psrc2->getMat(), dst = _dst.getMat();
        Size sz = getContinuousSize2D(src1, src2, dst);
        size_t len = sz.width*(size_t)cn;
        if (len < INT_MAX)  // FIXIT similar code below doesn't have that check
        {
            sz.width = (int)len;
            func(src1.ptr(), src1.step, src2.ptr(), src2.step, dst.ptr(), dst.step, sz.width, sz.height, 0);
            return;
        }
    }

    if( oclop == OCL_OP_NOT )
        haveScalar = true;
    else if( (kind1 == _InputArray::MATX) + (kind2 == _InputArray::MATX) == 1 ||
        !psrc1->sameSize(*psrc2) || type1 != type2 )
    {
        if( checkScalar(*psrc1, type2, kind1, kind2) )
        {
            // src1 is a scalar; swap it with src2
            swap(psrc1, psrc2);
            swap(type1, type2);
            swap(depth1, depth2);
            swap(cn, cn2);
            swap(sz1, sz2);
        }
        else if( !checkScalar(*psrc2, type1, kind2, kind1) )
            CV_Error( cv::Error::StsUnmatchedSizes,
                      "The operation is neither 'array op array' (where arrays have the same size and type), "
                      "nor 'array op scalar', nor 'scalar op array'" );
        haveScalar = true;
    }
    else
    {
        CV_Assert( psrc1->sameSize(*psrc2) && type1 == type2 );
    }

    size_t esz = CV_ELEM_SIZE(type1);
    size_t blocksize0 = (BLOCK_SIZE + esz-1)/esz;
    BinaryFunc copymask = 0;
    bool reallocate = false;

    if( haveMask )
    {
        int mtype = _mask.type();
        CV_Assert( (mtype == CV_8U || mtype == CV_8S || mtype == CV_Bool) && _mask.sameSize(*psrc1));
        copymask = getCopyMaskFunc(esz);
        reallocate = !_dst.sameSize(*psrc1) || _dst.type() != type1;
    }

    AutoBuffer<uchar> _buf;
    uchar *scbuf = 0, *maskbuf = 0;

    _dst.createSameSize(*psrc1, type1);
    // if this is mask operation and dst has been reallocated,
    // we have to clear the destination
    if( haveMask && reallocate )
        _dst.setTo(0.);

    CV_OCL_RUN(use_opencl,
               ocl_binary_op(*psrc1, *psrc2, _dst, _mask, bitwise, oclop, haveScalar))


    Mat src1 = psrc1->getMat(), src2 = psrc2->getMat();
    Mat dst = _dst.getMat(), mask = _mask.getMat();

    if( bitwise )
    {
        func = *tab;
        cn = (int)esz;
    }
    else
        func = tab[depth1];
    CV_Assert(func);

    if( !haveScalar )
    {
        const Mat* arrays[] = { &src1, &src2, &dst, &mask, 0 };
        uchar* ptrs[4] = {};

        NAryMatIterator it(arrays, ptrs);
        size_t total = it.size, blocksize = total;

        if( blocksize*cn > INT_MAX )
            blocksize = INT_MAX/cn;

        if( haveMask )
        {
            blocksize = std::min(blocksize, blocksize0);
            _buf.allocate(blocksize*esz);
            maskbuf = _buf.data();
        }

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            for( size_t j = 0; j < total; j += blocksize )
            {
                int bsz = (int)MIN(total - j, blocksize);

                func( ptrs[0], 0, ptrs[1], 0, haveMask ? maskbuf : ptrs[2], 0, bsz*cn, 1, 0 );
                if( haveMask )
                {
                    copymask( maskbuf, 0, ptrs[3], 0, ptrs[2], 0, Size(bsz, 1), &esz );
                    ptrs[3] += bsz;
                }

                bsz *= (int)esz;
                ptrs[0] += bsz; ptrs[1] += bsz; ptrs[2] += bsz;
            }
        }
    }
    else
    {
        const Mat* arrays[] = { &src1, &dst, &mask, 0 };
        uchar* ptrs[3] = {};

        NAryMatIterator it(arrays, ptrs);
        size_t total = it.size, blocksize = std::min(total, blocksize0);

        _buf.allocate(blocksize*(haveMask ? 2 : 1)*esz + 32);
        scbuf = _buf.data();
        maskbuf = alignPtr(scbuf + blocksize*esz, 16);

        convertAndUnrollScalar( src2, src1.type(), scbuf, blocksize);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            for( size_t j = 0; j < total; j += blocksize )
            {
                int bsz = (int)MIN(total - j, blocksize);

                func( ptrs[0], 0, scbuf, 0, haveMask ? maskbuf : ptrs[1], 0, bsz*cn, 1, 0 );
                if( haveMask )
                {
                    copymask( maskbuf, 0, ptrs[2], 0, ptrs[1], 0, Size(bsz, 1), &esz );
                    ptrs[2] += bsz;
                }

                bsz *= (int)esz;
                ptrs[0] += bsz; ptrs[1] += bsz;
            }
        }
    }
}

static BinaryFuncC* getMaxTab()
{
    static BinaryFuncC maxTab[CV_DEPTH_MAX] =
    {
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::max8u),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::max8s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::max16u),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::max16s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::max32s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::max32f),
        (BinaryFuncC)cv::hal::max64f,
        (BinaryFuncC)cv::hal::max16f,
        (BinaryFuncC)cv::hal::max16bf,
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::max8u), // bool
        (BinaryFuncC)cv::hal::max64u,
        (BinaryFuncC)cv::hal::max64s,
        (BinaryFuncC)cv::hal::max32u,
        0
    };

    return maxTab;
}

static BinaryFuncC* getMinTab()
{
    static BinaryFuncC minTab[CV_DEPTH_MAX] =
    {
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::min8u),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::min8s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::min16u),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::min16s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::min32s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::min32f),
        (BinaryFuncC)cv::hal::min64f,
        (BinaryFuncC)cv::hal::min16f,
        (BinaryFuncC)cv::hal::min16bf,
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::min8u), // bool
        (BinaryFuncC)cv::hal::min64u,
        (BinaryFuncC)cv::hal::min64s,
        (BinaryFuncC)cv::hal::min32u,
        0
    };

    return minTab;
}

}

void cv::bitwise_and(InputArray a, InputArray b, OutputArray c, InputArray mask)
{
    CV_INSTRUMENT_REGION();

    BinaryFuncC f = (BinaryFuncC)GET_OPTIMIZED(cv::hal::and8u);
    binary_op(a, b, c, mask, &f, true, OCL_OP_AND);
}

void cv::bitwise_or(InputArray a, InputArray b, OutputArray c, InputArray mask)
{
    CV_INSTRUMENT_REGION();

    BinaryFuncC f = (BinaryFuncC)GET_OPTIMIZED(cv::hal::or8u);
    binary_op(a, b, c, mask, &f, true, OCL_OP_OR);
}

void cv::bitwise_xor(InputArray a, InputArray b, OutputArray c, InputArray mask)
{
    CV_INSTRUMENT_REGION();

    BinaryFuncC f = (BinaryFuncC)GET_OPTIMIZED(cv::hal::xor8u);
    binary_op(a, b, c, mask, &f, true, OCL_OP_XOR);
}

void cv::bitwise_not(InputArray a, OutputArray c, InputArray mask)
{
    CV_INSTRUMENT_REGION();

    BinaryFuncC f = (BinaryFuncC)GET_OPTIMIZED(cv::hal::not8u);
    binary_op(a, a, c, mask, &f, true, OCL_OP_NOT);
}

void cv::max( InputArray src1, InputArray src2, OutputArray dst )
{
    CV_INSTRUMENT_REGION();

    arithm_op(ew::OP_MAX, src1, src2, dst, noArray(), -1, OCL_OP_MAX, false);
}

void cv::min( InputArray src1, InputArray src2, OutputArray dst )
{
    CV_INSTRUMENT_REGION();

    arithm_op(ew::OP_MIN, src1, src2, dst, noArray(), -1, OCL_OP_MIN, false);
}

void cv::max(const Mat& src1, const Mat& src2, Mat& dst)
{
    CV_INSTRUMENT_REGION();

    OutputArray _dst(dst);
    binary_op(src1, src2, _dst, noArray(), getMaxTab(), false, OCL_OP_MAX );
}

void cv::min(const Mat& src1, const Mat& src2, Mat& dst)
{
    CV_INSTRUMENT_REGION();

    OutputArray _dst(dst);
    binary_op(src1, src2, _dst, noArray(), getMinTab(), false, OCL_OP_MIN );
}

void cv::max(const UMat& src1, const UMat& src2, UMat& dst)
{
    CV_INSTRUMENT_REGION();

    OutputArray _dst(dst);
    binary_op(src1, src2, _dst, noArray(), getMaxTab(), false, OCL_OP_MAX );
}

void cv::min(const UMat& src1, const UMat& src2, UMat& dst)
{
    CV_INSTRUMENT_REGION();

    OutputArray _dst(dst);
    binary_op(src1, src2, _dst, noArray(), getMinTab(), false, OCL_OP_MIN );
}


/****************************************************************************************\
*                                      add/subtract                                      *
\****************************************************************************************/

namespace cv
{

static int actualScalarDepth(const double* data, int len)
{
    int i = 0, minval = INT_MAX, maxval = INT_MIN;
    for(; i < len; ++i)
    {
        int ival = cvRound(data[i]);
        if( ival != data[i] )
            break;
        minval = MIN(minval, ival);
        maxval = MAX(maxval, ival);
    }
    return i < len ? CV_64F :
        minval >= 0 && maxval <= (int)UCHAR_MAX ? CV_8U :
        minval >= (int)SCHAR_MIN && maxval <= (int)SCHAR_MAX ? CV_8S :
        minval >= 0 && maxval <= (int)USHRT_MAX ? CV_16U :
        minval >= (int)SHRT_MIN && maxval <= (int)SHRT_MAX ? CV_16S :
        CV_32S;
}

static int coerceTypes(int depth1, int depth2, bool muldiv)
{
    return depth1 == depth2 ? depth1 :
        ((depth1 <= CV_32S) & (depth2 <= CV_32S)) != 0 ?
        (((int)!muldiv & (depth1 <= CV_8S) & (depth2 <= CV_8S)) != 0 ? CV_16S : CV_32S) :
        ((CV_ELEM_SIZE1(depth1) > 4) | (CV_ELEM_SIZE1(depth2) > 4)) != 0 ? CV_64F : CV_32F;
}

#ifdef HAVE_OPENCL

static bool ocl_arithm_op(InputArray _src1, InputArray _src2, OutputArray _dst,
                          InputArray _mask, int wtype,
                          void* usrdata, int oclop,
                          bool haveScalar )
{
    const ocl::Device d = ocl::Device::getDefault();
    bool doubleSupport = d.doubleFPConfig() > 0;
    int type1 = _src1.type(), depth1 = CV_MAT_DEPTH(type1), cn = CV_MAT_CN(type1);
    bool haveMask = !_mask.empty();

    if ( (haveMask || haveScalar) && cn > 4 )
        return false;

    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype), wdepth = std::max(CV_32S, CV_MAT_DEPTH(wtype));
    if (!doubleSupport)
        wdepth = std::min(wdepth, CV_32F);

    wtype = CV_MAKETYPE(wdepth, cn);
    int type2 = haveScalar ? wtype : _src2.type(), depth2 = CV_MAT_DEPTH(type2);
    if (!doubleSupport && (depth2 == CV_64F || depth1 == CV_64F))
        return false;

    int kercn = haveMask || haveScalar ? cn : ocl::predictOptimalVectorWidth(_src1, _src2, _dst);
    int scalarcn = kercn == 3 ? 4 : kercn, rowsPerWI = d.isIntel() ? 4 : 1;

    char cvtstr[4][50], opts[1024];
    snprintf(opts, sizeof(opts), "-D %s%s -D %s -D srcT1=%s -D srcT1_C1=%s -D srcT2=%s -D srcT2_C1=%s "
            "-D dstT=%s -D DEPTH_dst=%d -D dstT_C1=%s -D workT=%s -D workST=%s -D scaleT=%s -D wdepth=%d -D convertToWT1=%s "
            "-D convertToWT2=%s -D convertToDT=%s%s -D cn=%d -D rowsPerWI=%d -D convertFromU=%s",
            (haveMask ? "MASK_" : ""), (haveScalar ? "UNARY_OP" : "BINARY_OP"),
            oclop2str[oclop], ocl::typeToStr(CV_MAKETYPE(depth1, kercn)),
            ocl::typeToStr(depth1), ocl::typeToStr(CV_MAKETYPE(depth2, kercn)),
            ocl::typeToStr(depth2), ocl::typeToStr(CV_MAKETYPE(ddepth, kercn)), ddepth,
            ocl::typeToStr(ddepth), ocl::typeToStr(CV_MAKETYPE(wdepth, kercn)),
            ocl::typeToStr(CV_MAKETYPE(wdepth, scalarcn)),
            ocl::typeToStr(wdepth), wdepth,
            ocl::convertTypeStr(depth1, wdepth, kercn, cvtstr[0], sizeof(cvtstr[0])),
            ocl::convertTypeStr(depth2, wdepth, kercn, cvtstr[1], sizeof(cvtstr[1])),
            ocl::convertTypeStr(wdepth, ddepth, kercn, cvtstr[2], sizeof(cvtstr[2])),
            doubleSupport ? " -D DOUBLE_SUPPORT" : "", kercn, rowsPerWI,
            oclop == OCL_OP_ABSDIFF && wdepth == CV_32S && ddepth == wdepth ?
            ocl::convertTypeStr(CV_8U, ddepth, kercn, cvtstr[3], sizeof(cvtstr[3])) : "noconvert");

    size_t usrdata_esz = CV_ELEM_SIZE(wdepth);
    const uchar* usrdata_p = (const uchar*)usrdata;
    const double* usrdata_d = (const double*)usrdata;
    float usrdata_f[3];
    int i, n = oclop == OCL_OP_MUL_SCALE || oclop == OCL_OP_DIV_SCALE ||
        oclop == OCL_OP_RDIV_SCALE || oclop == OCL_OP_RECIP_SCALE ? 1 : oclop == OCL_OP_ADDW ? 3 : 0;
    if( usrdata && n > 0 && wdepth == CV_32F )
    {
        for( i = 0; i < n; i++ )
            usrdata_f[i] = (float)usrdata_d[i];
        usrdata_p = (const uchar*)usrdata_f;
    }

    ocl::Kernel k("KF", ocl::core::arithm_oclsrc, opts);
    if (k.empty())
        return false;

    UMat src1 = _src1.getUMat(), src2;
    UMat dst = _dst.getUMat(), mask = _mask.getUMat();

    ocl::KernelArg src1arg = ocl::KernelArg::ReadOnlyNoSize(src1, cn, kercn);
    ocl::KernelArg dstarg = haveMask ? ocl::KernelArg::ReadWrite(dst, cn, kercn) :
                                       ocl::KernelArg::WriteOnly(dst, cn, kercn);
    ocl::KernelArg maskarg = ocl::KernelArg::ReadOnlyNoSize(mask, 1);

    if( haveScalar )
    {
        size_t esz = CV_ELEM_SIZE1(wtype)*scalarcn;
        double buf[4]={0,0,0,0};
        Mat src2sc = _src2.getMat();

        if( !src2sc.empty() )
            convertAndUnrollScalar(src2sc, wtype, (uchar*)buf, 1);
        ocl::KernelArg scalararg = ocl::KernelArg(ocl::KernelArg::CONSTANT, 0, 0, 0, buf, esz);

        if( !haveMask )
        {
            if(n == 0)
                k.args(src1arg, dstarg, scalararg);
            else if(n == 1)
                k.args(src1arg, dstarg, scalararg,
                       ocl::KernelArg(ocl::KernelArg::CONSTANT, 0, 0, 0, usrdata_p, usrdata_esz));
            else
                CV_Error(Error::StsNotImplemented, "unsupported number of extra parameters");
        }
        else
            k.args(src1arg, maskarg, dstarg, scalararg);
    }
    else
    {
        src2 = _src2.getUMat();
        ocl::KernelArg src2arg = ocl::KernelArg::ReadOnlyNoSize(src2, cn, kercn);

        if( !haveMask )
        {
            if (n == 0)
                k.args(src1arg, src2arg, dstarg);
            else if (n == 1)
                k.args(src1arg, src2arg, dstarg,
                       ocl::KernelArg(ocl::KernelArg::CONSTANT, 0, 0, 0, usrdata_p, usrdata_esz));
            else if (n == 3)
                k.args(src1arg, src2arg, dstarg,
                       ocl::KernelArg(ocl::KernelArg::CONSTANT, 0, 0, 0, usrdata_p, usrdata_esz),
                       ocl::KernelArg(ocl::KernelArg::CONSTANT, 0, 0, 0, usrdata_p + usrdata_esz, usrdata_esz),
                       ocl::KernelArg(ocl::KernelArg::CONSTANT, 0, 0, 0, usrdata_p + usrdata_esz*2, usrdata_esz));
            else
                CV_Error(Error::StsNotImplemented, "unsupported number of extra parameters");
        }
        else
            k.args(src1arg, src2arg, maskarg, dstarg);
    }

    size_t globalsize[] = { (size_t)src1.cols * cn / kercn, ((size_t)src1.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

// OpenCL branch of the arithmetic ops, extracted from the former arithm_op (whose CPU compute is now
// the element-wise engine). Returns true if an OpenCL kernel handled the op, false to let the caller
// fall through to the CPU engine. min/max use the bitwise-style kernel; every other op the arithm
// kernel with a coerced working type. `usrdata` carries the mul/div/addWeighted scale block, `oclop`
// the kernel id. Only reached for a UMat dst; sizes/types that don't fit the scalar/same-size pattern
// (e.g. a broadcast) decline (return false) so the engine handles them on the CPU.
static bool arithm_op_ocl(InputArray _src1, InputArray _src2, OutputArray _dst,
                          InputArray _mask, int dtype, int oclop, bool muldiv, void* usrdata)
{
#ifndef HAVE_OPENCL
    CV_UNUSED(_src1); CV_UNUSED(_src2); CV_UNUSED(_dst); CV_UNUSED(_mask);
    CV_UNUSED(dtype); CV_UNUSED(oclop); CV_UNUSED(muldiv); CV_UNUSED(usrdata);
    return false;
#else
    const _InputArray *psrc1 = &_src1, *psrc2 = &_src2;
    _InputArray::KindFlag kind1 = psrc1->kind(), kind2 = psrc2->kind();
    bool haveMask = !_mask.empty();
    bool reallocate = false;
    int type1 = psrc1->type(), depth1 = CV_MAT_DEPTH(type1), cn = CV_MAT_CN(type1);
    int type2 = psrc2->type(), depth2 = CV_MAT_DEPTH(type2), cn2 = CV_MAT_CN(type2);
    int wtype, dims1 = psrc1->dims(), dims2 = psrc2->dims();
    Size sz1 = dims1 <= 2 ? psrc1->size() : Size();
    Size sz2 = dims2 <= 2 ? psrc2->size() : Size();

    if (!(OCL_PERFORMANCE_CHECK(_dst.isUMat()) && dims1 <= 2 && dims2 <= 2))
        return false;

    // min/max: the bitwise-style OpenCL kernel (same-type element-wise, no working-type coercion).
    if (oclop == OCL_OP_MIN || oclop == OCL_OP_MAX)
    {
        bool haveScalar = false;
        if ((kind1 == _InputArray::MATX) + (kind2 == _InputArray::MATX) == 1 ||
            !psrc1->sameSize(*psrc2) || type1 != type2)
        {
            if (checkScalar(*psrc1, type2, kind1, kind2))
            { std::swap(psrc1, psrc2); std::swap(type1, type2); }
            else if (!checkScalar(*psrc2, type1, kind2, kind1))
                return false;
            haveScalar = true;
        }
        _dst.createSameSize(*psrc1, type1);
        return ocl_binary_op(*psrc1, *psrc2, _dst, _mask, false, oclop, haveScalar);
    }

    // reciprocal (scale/src): the OpenCL kernel is unary on src - use src2 for both operands (the CPU
    // engine received a 0-dim `1` numerator as src1, which the recip kernel would ignore). [TODO.VP: tidy]
    if (oclop == OCL_OP_RECIP_SCALE)
    {
        psrc1 = psrc2; kind1 = kind2; type1 = type2; depth1 = depth2; cn = cn2; dims1 = dims2; sz1 = sz2;
    }

    // add/sub/mul/div/absdiff/addWeighted/recip: the arithm OpenCL kernel with a coerced work type.
    bool src1Scalar = checkScalar(*psrc1, type2, kind1, kind2);
    bool src2Scalar = checkScalar(*psrc2, type1, kind2, kind1);
    bool haveScalar = false;

    if (dims1 != dims2 || sz1 != sz2 || cn != cn2 ||
        (kind1 == _InputArray::MATX && (sz1 == Size(1,4) || sz1 == Size(1,1))) ||
        (kind2 == _InputArray::MATX && (sz2 == Size(1,4) || sz2 == Size(1,1))))
    {
        if ((type1 == CV_64F && (sz1.height == 1 || sz1.height == 4)) && src1Scalar)
        {
            // src1 is a scalar; swap it with src2
            std::swap(psrc1, psrc2); std::swap(sz1, sz2); std::swap(type1, type2);
            std::swap(depth1, depth2); std::swap(cn, cn2); std::swap(dims1, dims2);
            if (oclop == OCL_OP_SUB) oclop = OCL_OP_RSUB;
            if (oclop == OCL_OP_DIV_SCALE) oclop = OCL_OP_RDIV_SCALE;
        }
        else if (!src2Scalar)
            return false;   // array op array with mismatched size/cn: engine broadcasts on the CPU
        haveScalar = true;
        CV_Assert((type2 == CV_64F || type2 == CV_32F) && (sz2.height == 1 || sz2.height == 4));

        if (!muldiv)
        {
            Mat sc = psrc2->getMat();
            depth2 = actualScalarDepth(sc.ptr<double>(), sz2 == Size(1, 1) ? cn2 : cn);
            if (depth2 == CV_64F && CV_ELEM_SIZE1(depth1) < 8)
                depth2 = CV_32F;
        }
        else
            depth2 = CV_64F;
    }

    if (dtype < 0)
    {
        if (_dst.fixedType())
            dtype = _dst.type();
        else
        {
            if (!haveScalar && type1 != type2)
                CV_Error(cv::Error::StsBadArg,
                     "When the input arrays in add/subtract/multiply/divide functions have different types, "
                     "the output array type must be explicitly specified");
            dtype = type1;
        }
    }
    dtype = CV_MAT_DEPTH(dtype);

    if (depth1 == depth2 && dtype == depth1)
        wtype = dtype;
    else if (!muldiv)
    {
        wtype = coerceTypes(depth1, depth2, false);
        wtype = coerceTypes(wtype, dtype, false);
        if (dtype < CV_32F && (depth1 < CV_32F || depth2 < CV_32F))
            wtype = CV_32S;
    }
    else
    {
        wtype = coerceTypes(depth1, depth2, true);
        wtype = coerceTypes(wtype, dtype, true);
    }

    // The scaled OpenCL kernels (mul/div by a scale, reciprocal, addWeighted) compute in float; the old
    // same-type fast path forced max(depth, CV_32F). coerceTypes keeps same-type integer, so bump here.
    if (oclop == OCL_OP_MUL_SCALE || oclop == OCL_OP_DIV_SCALE || oclop == OCL_OP_RDIV_SCALE ||
        oclop == OCL_OP_RECIP_SCALE || oclop == OCL_OP_ADDW)
        wtype = std::max(wtype, (int)CV_32F);

    dtype = CV_MAKETYPE(dtype, cn);
    wtype = CV_MAKETYPE(wtype, cn);

    if (haveMask)
    {
        int mtype = _mask.type();
        CV_Assert((mtype == CV_8UC1 || mtype == CV_8SC1 || mtype == CV_Bool) && _mask.sameSize(*psrc1));
        reallocate = !_dst.sameSize(*psrc1) || _dst.type() != dtype;
    }

    _dst.createSameSize(*psrc1, dtype);
    if (reallocate)
        _dst.setTo(0.);

    return ocl_arithm_op(*psrc1, *psrc2, _dst, _mask, wtype, usrdata, oclop, haveScalar);
#endif
}

// Element-wise binary op (ADD/SUB/...) on two CPU operands - each an array Mat or a MATX scalar -
// through the new broadcasting engine. Handles mixed input types (emitBinary inserts the casts) and
// broadcasting (a scalar rides as a 0-dim per-channel CONST; arrays broadcast via outputShape/exec).
// Returns false (declining) for UMat, a write-mask, or two scalar operands, so the caller falls
// through to arithm_op. Incompatible shapes make the engine throw (matching cv::add's error).
// Does the output array already have exactly this shape+type (=> create() reuses it, no realloc)?
// Works for Mat and UMat alike (shape/type queries, no data-pointer peeking).
static bool outArrayMatches(const _OutputArray& a, const MatShape& shp, int type)
{
    if (a.empty() || a.type() != type)
        return false;
    int sz[MatShape::MAX_DIMS];
    int nd = a.sizend(sz);
    if (nd != (int)shp.size())
        return false;
    for (int i = 0; i < nd; i++)
        if (sz[i] != shp[i])
            return false;
    return true;
}

// The one entry point for every element-wise binary op (add/sub/mul/div/min/max/absdiff/addWeighted/
// reciprocal). Handles empty inputs, the UMat/OpenCL branch (arithm_op_ocl) and the CPU element-wise
// engine. `oclop` selects the OpenCL kernel; `muldiv` drives the OpenCL working-type coercion + scale;
// `params` are the op scalars (params[0]=mul/div scale; {alpha,beta,gamma} for addWeighted).
static void arithm_op(ew::TOp op, InputArray src1, InputArray src2, OutputArray dst,
                      InputArray mask, int dtype, int oclop, bool muldiv, const Scalar& params)
{
    CV_Assert(src1.empty() == src2.empty());
    if (src1.empty() && src2.empty())          // empty inputs -> empty result
    {
        dst.release();
        if (dtype >= 0)
            dst.create(0, 0, dtype);
        return;
    }

    // UMat -> classic OpenCL kernel. It declines (false) when OpenCL can't apply (no device, dims>2, a
    // broadcast, cn>4 masked, ...) - then the CPU engine below handles it (mapping the UMat via getMat).
    if (src1.isUMat() || src2.isUMat() || dst.isUMat())
    {
        double abg[3] = { params[0], params[1], params[2] }, scale = params[0];
        void* usrdata = (op == ew::OP_ADDW) ? (void*)abg : (muldiv ? (void*)&scale : nullptr);
        if (arithm_op_ocl(src1, src2, dst, mask, dtype, oclop, muldiv, usrdata))
            return;
    }

    const bool haveMask = !mask.empty();
    const int cn1 = src1.channels(), cn2 = src2.channels();
    bool s1 = isScalarArg(src1, cn2), s2 = isScalarArg(src2, cn1);
    if (s1 && s2)
        s1 = s2 = false;    // two scalars: treat both as tiny array operands (element-wise + broadcast),
                            // matching arithm_op (Scalar+Scalar -> 4x1, number+number -> 1x1, ...)

    // Direct Mat pointers - NO header copy / refcount atomics (the InputArrays outlive this call, so
    // their data stays alive). A MAT operand is used in place via getObj(); a non-MAT one (UMat, or a
    // two-scalar operand handled as an array) is mapped once into a local. EXCEPTION - in-place with a
    // realloc: if dst IS one of the sources, dst.create() may free that source's data mid-op, so take a
    // header copy (increfs, keeping the old data alive) of the sources in that case only.
    const void* dstObj = (dst.kind() == _InputArray::MAT) ? dst.getObj() : nullptr;
    const bool aliased = dstObj && (dstObj == src1.getObj() || dstObj == src2.getObj());
    Mat m1loc, m2loc, mloc;
    auto asMat = [&](InputArray a, Mat& loc) -> const Mat* {
        if (a.kind() == _InputArray::MAT && !aliased) return (const Mat*)a.getObj();
        loc = a.getMat(); return &loc;
    };
    const Mat* pm1 = s1 ? nullptr : asMat(src1, m1loc);
    const Mat* pm2 = s2 ? nullptr : asMat(src2, m2loc);
    const int adepth = s1 ? pm2->depth() : pm1->depth();   // the (first) array operand's depth
    // result depth: explicit dtype wins; else a fixed-type dst dictates it (matches arithm_op); else
    // the array operand's own depth.
    const int rdepth = dtype >= 0 ? CV_MAT_DEPTH(dtype)
                     : (dst.fixedType() ? dst.depth() : adepth);

    // operand order = (src1, src2): an array becomes an INPUT, a scalar a flexible per-channel CONST
    // read straight from the caller's inline storage (getObj()), no Mat / convertTo.
    ew::TExpr p;
    auto addOperand = [&](InputArray src, bool isScalar, const Mat* m, int otherCn) -> int {
        if (!isScalar)
            return p.addInput(m->depth());
        const Size sz = src.getSz();
        const int scn0 = sz.width * sz.height, cn = otherCn;
        const int scn = (cn == scn0 || (cn < 4 && scn0 == 4)) ? cn
                      : (scn0 == 1) ? 1 : 0;
        CV_Assert(scn > 0);
        return p.addConst(ew::EW_DEPTH_NONE, src.depth(), src.getObj(), scn);
    };
    const int a0 = addOperand(src1, s1, pm1, cn2);
    const int a1 = addOperand(src2, s2, pm2, cn1);

    // Write-mask: a single-channel 1-byte array (u8/s8/bool) the size of the output spatial shape,
    // added as another broadcast input. The arithmetic result lands in a temp; OP_COPY_MASK then
    // overwrites only mask!=0 positions of the (pre-existing) output => dst = mask ? result : dst.
    int sMask = 0;
    const Mat* pmask = nullptr;
    if (haveMask)
    {
        CV_Assert(mask.type() == CV_8U || mask.type() == CV_8S || mask.type() == CV_Bool);
        pmask = asMat(mask, mloc);
        sMask = p.addInput(pmask->depth());
    }
    const int sOut = p.addOutput(rdepth);               // output BEFORE emit -> moveToOutput drops the
    const int r = p.emitBinary(op, a0, a1, rdepth, params);   // result temp (mask-free => none); scale
                                                        // rides params[0] (mul/div), {a,b,g} for addW
    if (haveMask)
        p.addInsn(ew::OP_COPY_MASK, r, sMask, 0, sOut);
    else
        p.moveToOutput(r, sOut);
    p.compile();

    const Mat* inputs[3]; int ni = 0;                   // inputs in program (input-index) order
    if (!s1) inputs[ni++] = pm1;
    if (!s2) inputs[ni++] = pm2;
    if (haveMask) inputs[ni++] = pmask;

    // Fast path (the common case): a plain Mat dst, no mask. Hand exec the dst Mat directly - it sizes
    // and fills it in place, exactly like the raw engine, with no extra outputShape/create/getMat.
    if (!haveMask && dst.kind() == _InputArray::MAT)
    {
        p.exec(inputs, (Mat*)dst.getObj());
        return;
    }

    // General path: pre-create the output at the broadcast shape (needed for a UMat dst, uploaded back
    // on release; and for the read-modify mask). With a mask, a freshly (re)allocated dst is zeroed so
    // mask==0 reads 0 (matches arithm_op); a reused dst keeps its prior content there - detect reuse by
    // shape+type BEFORE create (correct for a reused UMat dst too).
    MatShape oshape; int ocn;
    p.outputShape(inputs, oshape, ocn);
    const int otype = CV_MAKETYPE(rdepth, ocn);
    const bool reused = haveMask && outArrayMatches(dst, oshape, otype);
    dst.create(oshape, otype);
    Mat dstloc;
    Mat* pdst = (dst.kind() == _InputArray::MAT) ? (Mat*)dst.getObj() : &(dstloc = dst.getMat());
    if (haveMask && !reused)
        pdst->setZero();                          // setZero (not setTo, which caps at 4 channels)
    p.exec(inputs, pdst);
}

}

void cv::add( InputArray src1, InputArray src2, OutputArray dst,
          InputArray mask, int dtype )
{
    CV_INSTRUMENT_REGION();

    arithm_op(ew::OP_ADD, src1, src2, dst, mask, dtype, OCL_OP_ADD, false);
}

void cv::subtract( InputArray src1, InputArray src2, OutputArray dst,
                   InputArray mask, int dtype )
{
    CV_INSTRUMENT_REGION();

    arithm_op(ew::OP_SUB, src1, src2, dst, mask, dtype, OCL_OP_SUB, false);
}

void cv::absdiff( InputArray src1, InputArray src2, OutputArray dst )
{
    CV_INSTRUMENT_REGION();

    arithm_op(ew::OP_ABSDIFF, src1, src2, dst, noArray(), -1, OCL_OP_ABSDIFF, false);
}

void cv::copyTo(InputArray _src, OutputArray _dst, InputArray _mask)
{
    CV_INSTRUMENT_REGION();

    _src.copyTo(_dst, _mask);
}

/****************************************************************************************\
*                                    multiply/divide                                     *
\****************************************************************************************/

namespace cv
{

void multiply(InputArray src1, InputArray src2,
              OutputArray dst, double scale, int dtype)
{
    CV_INSTRUMENT_REGION();

    const int oclop = std::abs(scale - 1.0) < DBL_EPSILON ? OCL_OP_MUL : OCL_OP_MUL_SCALE;
    arithm_op(ew::OP_MUL, src1, src2, dst, noArray(), dtype, oclop, /*muldiv*/ true, Scalar(scale));
}

void divide(InputArray src1, InputArray src2,
                OutputArray dst, double scale, int dtype)
{
    CV_INSTRUMENT_REGION();

    arithm_op(ew::OP_DIV, src1, src2, dst, noArray(), dtype, OCL_OP_DIV_SCALE, /*muldiv*/ true, Scalar(scale));
}

void divide(double scale, InputArray src2,
                OutputArray dst, int dtype)
{
    CV_INSTRUMENT_REGION();

    if (src2.empty())
    {
        dst.release();
        return;
    }

    // scale / src2 == scale * 1 / src2: feed a 0-dim integer `1` as the numerator so the CPU engine
    // reuses the normal divide (scale*a/b). Integer (not f64) so div-by-zero on an integer src still
    // yields 0 (the div guard triggers only when BOTH operands are integer). The result depth follows
    // src2 (not the numerator): pass it explicitly for dtype<0 (a fixed-type dst still wins). The UMat
    // path uses the dedicated reciprocal kernel (OCL_OP_RECIP_SCALE).
    int one = 1;
    Mat numerator(MatShape::scalar(), CV_32S, &one);
    const int rtype = (dtype < 0 && !dst.fixedType()) ? src2.depth() : dtype;
    arithm_op(ew::OP_DIV, numerator, src2, dst, noArray(), rtype, OCL_OP_RECIP_SCALE, /*muldiv*/ true, Scalar(scale));
}

UMat UMat::mul(InputArray m, double scale) const
{
    UMat dst;
    multiply(*this, m, dst, scale);
    return dst;
}

/****************************************************************************************\
*                                      addWeighted                                       *
\****************************************************************************************/

}

void cv::addWeighted( InputArray src1, double alpha, InputArray src2,
                      double beta, double gamma, OutputArray dst, int dtype )
{
    CV_INSTRUMENT_REGION();

    arithm_op(ew::OP_ADDW, src1, src2, dst, noArray(), dtype, OCL_OP_ADDW, /*muldiv*/ true,
              Scalar(alpha, beta, gamma));
}


/****************************************************************************************\
*                                          compare                                       *
\****************************************************************************************/

namespace cv
{

static BinaryFuncC getCmpFunc(int depth)
{
    static BinaryFuncC cmpTab[CV_DEPTH_MAX] =
    {
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::cmp8u),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::cmp8s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::cmp16u),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::cmp16s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::cmp32s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::cmp32f),
        (BinaryFuncC)cv::hal::cmp64f,
        (BinaryFuncC)cv::hal::cmp16f,
        (BinaryFuncC)cv::hal::cmp16bf,
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::cmp8u),
        (BinaryFuncC)cv::hal::cmp64u,
        (BinaryFuncC)cv::hal::cmp64s,
        (BinaryFuncC)cv::hal::cmp32u,
        0
    };

    return cmpTab[depth];
}

static double getMinVal(int depth)
{
    static const double tab[CV_DEPTH_MAX] =
    {
        0, -128, 0, -32768, INT_MIN, -FLT_MAX, -DBL_MAX,
        -65504, -FLT_MAX, 0, 0, (double)INT64_MIN, 0
    };
    return tab[depth];
}

static double getMaxVal(int depth)
{
    static const double tab[CV_DEPTH_MAX] = {
        255, 127, 65535, 32767, INT_MAX, FLT_MAX, DBL_MAX,
        65504, FLT_MAX, 255, (double)UINT64_MAX, (double)INT64_MAX, (double)UINT32_MAX, 0
    };
    return tab[depth];
}

#ifdef HAVE_OPENCL

static bool ocl_compare(InputArray _src1, InputArray _src2, OutputArray _dst, int op, bool haveScalar)
{
    const ocl::Device& dev = ocl::Device::getDefault();
    bool doubleSupport = dev.doubleFPConfig() > 0;
    int type1 = _src1.type(), depth1 = CV_MAT_DEPTH(type1), cn = CV_MAT_CN(type1),
            type2 = _src2.type(), depth2 = CV_MAT_DEPTH(type2);

    if (!doubleSupport && depth1 == CV_64F)
        return false;

    if (!haveScalar && (!_src1.sameSize(_src2) || type1 != type2))
            return false;

    int kercn = haveScalar ? cn : ocl::predictOptimalVectorWidth(_src1, _src2, _dst), rowsPerWI = dev.isIntel() ? 4 : 1;
    // Workaround for bug with "?:" operator in AMD OpenCL compiler
    if (depth1 >= CV_16U)
        kercn = 1;

    int scalarcn = kercn == 3 ? 4 : kercn;
    const char * const operationMap[] = { "==", ">", ">=", "<", "<=", "!=" };
    char cvt[50];

    String opts = format("-D %s -D srcT1=%s -D dstT=%s -D DEPTH_dst=%d -D workT=srcT1 -D cn=%d"
                         " -D convertToDT=%s -D OP_CMP -D CMP_OPERATOR=%s -D srcT1_C1=%s"
                         " -D srcT2_C1=%s -D dstT_C1=%s -D workST=%s -D rowsPerWI=%d%s",
                         haveScalar ? "UNARY_OP" : "BINARY_OP",
                         ocl::typeToStr(CV_MAKE_TYPE(depth1, kercn)),
                         ocl::typeToStr(CV_8UC(kercn)), CV_8U, kercn,
                         ocl::convertTypeStr(depth1, CV_8U, kercn, cvt, sizeof(cvt)),
                         operationMap[op], ocl::typeToStr(depth1),
                         ocl::typeToStr(depth1), ocl::typeToStr(CV_8U),
                         ocl::typeToStr(CV_MAKE_TYPE(depth1, scalarcn)), rowsPerWI,
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "");

    ocl::Kernel k("KF", ocl::core::arithm_oclsrc, opts);
    if (k.empty())
        return false;

    UMat src1 = _src1.getUMat();
    Size size = src1.size();
    _dst.create(size, CV_8UC(cn));
    UMat dst = _dst.getUMat();

    if (haveScalar)
    {
        size_t esz = CV_ELEM_SIZE1(type1) * scalarcn;
        double buf[4] = { 0, 0, 0, 0 };
        Mat src2 = _src2.getMat();

        if( depth1 > CV_32S )
            convertAndUnrollScalar( src2, depth1, (uchar *)buf, kercn );
        else
        {
            double fval = 0;
            getConvertFunc(depth2, CV_64F)(src2.ptr(), 1, 0, 1, (uchar *)&fval, 1, Size(1, 1), 0);
            if( fval < getMinVal(depth1) )
                return dst.setTo(Scalar::all(op == CMP_GT || op == CMP_GE || op == CMP_NE ? 255 : 0)), true;

            if( fval > getMaxVal(depth1) )
                return dst.setTo(Scalar::all(op == CMP_LT || op == CMP_LE || op == CMP_NE ? 255 : 0)), true;

            int ival = cvRound(fval);
            if( fval != ival )
            {
                if( op == CMP_LT || op == CMP_GE )
                    ival = cvCeil(fval);
                else if( op == CMP_LE || op == CMP_GT )
                    ival = cvFloor(fval);
                else
                    return dst.setTo(Scalar::all(op == CMP_NE ? 255 : 0)), true;
            }
            convertAndUnrollScalar(Mat(1, 1, CV_32S, &ival), depth1, (uchar *)buf, kercn);
        }

        ocl::KernelArg scalararg = ocl::KernelArg(ocl::KernelArg::CONSTANT, 0, 0, 0, buf, esz);

        k.args(ocl::KernelArg::ReadOnlyNoSize(src1, cn, kercn),
               ocl::KernelArg::WriteOnly(dst, cn, kercn), scalararg);
    }
    else
    {
        UMat src2 = _src2.getUMat();

        k.args(ocl::KernelArg::ReadOnlyNoSize(src1),
               ocl::KernelArg::ReadOnlyNoSize(src2),
               ocl::KernelArg::WriteOnly(dst, cn, kercn));
    }

    size_t globalsize[2] = { (size_t)dst.cols * cn / kercn, ((size_t)dst.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

}

void cv::compare(InputArray _src1, InputArray _src2, OutputArray _dst, int op)
{
    CV_INSTRUMENT_REGION();

    CV_Assert( op == CMP_LT || op == CMP_LE || op == CMP_EQ ||
               op == CMP_NE || op == CMP_GE || op == CMP_GT );

    CV_Assert(_src1.empty() == _src2.empty());
    if (_src1.empty() && _src2.empty())
    {
        _dst.release();
        return;
    }

    bool haveScalar = false;

    if ((_src1.isMatx() + _src2.isMatx()) == 1
            || !_src1.sameSize(_src2)
            || _src1.type() != _src2.type())
    {
        bool is_src1_scalar = checkScalar(_src1, _src2.type(), _src1.kind(), _src2.kind());
        bool is_src2_scalar = checkScalar(_src2, _src1.type(), _src2.kind(), _src1.kind());

        if (is_src1_scalar && !is_src2_scalar)
        {
            op = op == CMP_LT ? CMP_GT : op == CMP_LE ? CMP_GE :
                op == CMP_GE ? CMP_LE : op == CMP_GT ? CMP_LT : op;
            // src1 is a scalar; swap it with src2
            compare(_src2, _src1, _dst, op);
            return;
        }
        else if(is_src1_scalar == is_src2_scalar)
            CV_Error( cv::Error::StsUnmatchedSizes,
                     "The operation is neither 'array op array' (where arrays have the same size and the same type), "
                     "nor 'array op scalar', nor 'scalar op array'" );
        haveScalar = true;
    }

    CV_OCL_RUN(_src1.dims() <= 2 && _src2.dims() <= 2 && OCL_PERFORMANCE_CHECK(_dst.isUMat()),
               ocl_compare(_src1, _src2, _dst, op, haveScalar))

    _InputArray::KindFlag kind1 = _src1.kind(), kind2 = _src2.kind();
    Mat src1 = _src1.getMat(), src2 = _src2.getMat();
    int depth1 = src1.depth(), depth2 = src2.depth();

    if( kind1 == kind2 && src1.dims <= 2 && src2.dims <= 2 && src1.size() == src2.size() && src1.type() == src2.type() )
    {
        int cn = src1.channels();
        _dst.createSameSize(src1, CV_8UC(cn));
        Mat dst = _dst.getMat();
        Size sz = getContinuousSize2D(src1, src2, dst, src1.channels());
        BinaryFuncC cmpFn = getCmpFunc(depth1);
        CV_Assert(cmpFn);
        cmpFn(src1.ptr(), src1.step, src2.ptr(), src2.step, dst.ptr(), dst.step, sz.width, sz.height, &op);
        return;
    }

    int cn = src1.channels();

    _dst.create(src1.size, CV_8UC(cn));
    src1 = src1.reshape(1); src2 = src2.reshape(1);
    Mat dst = _dst.getMat().reshape(1);

    size_t esz = std::max(src1.elemSize(), (size_t)1);
    size_t blocksize0 = (size_t)(BLOCK_SIZE + esz-1)/esz;
    BinaryFuncC func = getCmpFunc(depth1);
    CV_Assert(func);

    if( !haveScalar )
    {
        const Mat* arrays[] = { &src1, &src2, &dst, 0 };
        uchar* ptrs[3] = {};

        NAryMatIterator it(arrays, ptrs);
        size_t total = it.size;

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func( ptrs[0], 0, ptrs[1], 0, ptrs[2], 0, (int)total, 1, &op );
    }
    else
    {
        const Mat* arrays[] = { &src1, &dst, 0 };
        uchar* ptrs[2] = {};

        NAryMatIterator it(arrays, ptrs);
        size_t total = it.size, blocksize = std::min(total, blocksize0);

        AutoBuffer<uchar> _buf(blocksize*esz);
        uchar *buf = _buf.data();

        if( ((depth1 == CV_16F) | (depth1 == CV_16BF) |
             (depth1 == CV_32F) | (depth1 == CV_64F)) != 0 )
            convertAndUnrollScalar( src2, depth1, buf, blocksize );
        else
        {
            double fval=0;
            BinaryFunc cvtFn = getConvertFunc(depth2, CV_64F);
            CV_Assert(cvtFn);
            cvtFn(src2.ptr(), 1, 0, 1, (uchar*)&fval, 1, Size(1,1), 0);
            if( fval < getMinVal(depth1) )
            {
                dst = Scalar::all(op == CMP_GT || op == CMP_GE || op == CMP_NE ? 255 : 0);
                return;
            }

            if( fval > getMaxVal(depth1) )
            {
                dst = Scalar::all(op == CMP_LT || op == CMP_LE || op == CMP_NE ? 255 : 0);
                return;
            }

            double ival = round(fval);
            if( fval != ival )
            {
                if( op == CMP_LT || op == CMP_GE )
                    ival = ceil(fval);
                else if( op == CMP_LE || op == CMP_GT )
                    ival = floor(fval);
                else
                {
                    dst = Scalar::all(op == CMP_NE ? 255 : 0);
                    return;
                }
            }
            convertAndUnrollScalar(Mat(1, 1, CV_64F, &ival), depth1, buf, blocksize);
        }

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            for( size_t j = 0; j < total; j += blocksize )
            {
                int bsz = (int)MIN(total - j, blocksize);
                func( ptrs[0], 0, buf, 0, ptrs[1], 0, bsz, 1, &op);
                ptrs[0] += bsz*esz;
                ptrs[1] += bsz;
            }
        }
    }
}

/****************************************************************************************\
*                                        inRange                                         *
\****************************************************************************************/

namespace cv
{

template <typename T>
struct InRange_SIMD
{
    int operator () (const T * src1, const T * src2, const T * src3, uchar * dst, int len) const
    {
        for (int x = 0; x < len; x++)
            dst[x] = (src2[x] <= src1[x] && src1[x] <= src3[x]) ? 255 : 0;
        return len;
    }
};

#if (CV_SIMD || CV_SIMD_SCALABLE)

template <>
struct InRange_SIMD<uchar>
{
    int operator () (const uchar * src1, const uchar * src2, const uchar * src3,
        uchar * dst, int len) const
    {
        int x = 0;
        const int width = VTraits<v_uint8>::vlanes();

        for (; x <= len - width; x += width)
        {
            v_uint8 values = vx_load(src1 + x);
            v_uint8 low = vx_load(src2 + x);
            v_uint8 high = vx_load(src3 + x);

            v_store(dst + x, v_and(v_ge(values, low), v_ge(high, values)));
        }
        vx_cleanup();
        return x;
    }
};

template <>
struct InRange_SIMD<schar>
{
    int operator () (const schar * src1, const schar * src2, const schar * src3,
        uchar * dst, int len) const
    {
        int x = 0;
        const int width = VTraits<v_int8>::vlanes();

        for (; x <= len - width; x += width)
        {
            v_int8 values = vx_load(src1 + x);
            v_int8 low = vx_load(src2 + x);
            v_int8 high = vx_load(src3 + x);

            v_store((schar*)(dst + x), v_and(v_ge(values, low), v_ge(high, values)));
        }
        vx_cleanup();
        return x;
    }
};

template <>
struct InRange_SIMD<ushort>
{
    int operator () (const ushort * src1, const ushort * src2, const ushort * src3,
        uchar * dst, int len) const
    {
        int x = 0;
        const int width = VTraits<v_uint16>::vlanes() * 2;

        for (; x <= len - width; x += width)
        {
            v_uint16 values1 = vx_load(src1 + x);
            v_uint16 low1 = vx_load(src2 + x);
            v_uint16 high1 = vx_load(src3 + x);

            v_uint16 values2 = vx_load(src1 + x + VTraits<v_uint16>::vlanes());
            v_uint16 low2 = vx_load(src2 + x + VTraits<v_uint16>::vlanes());
            v_uint16 high2 = vx_load(src3 + x + VTraits<v_uint16>::vlanes());

            v_store(dst + x, v_pack(v_and(v_ge(values1, low1), v_ge(high1, values1)), v_and(v_ge(values2, low2), v_ge(high2, values2))));
        }
        vx_cleanup();
        return x;
    }
};

template <>
struct InRange_SIMD<short>
{
    int operator () (const short * src1, const short * src2, const short * src3,
        uchar * dst, int len) const
    {
        int x = 0;
        const int width = (int)VTraits<v_int16>::vlanes() * 2;

        for (; x <= len - width; x += width)
        {
            v_int16 values1 = vx_load(src1 + x);
            v_int16 low1 = vx_load(src2 + x);
            v_int16 high1 = vx_load(src3 + x);

            v_int16 values2 = vx_load(src1 + x + VTraits<v_int16>::vlanes());
            v_int16 low2 = vx_load(src2 + x + VTraits<v_int16>::vlanes());
            v_int16 high2 = vx_load(src3 + x + VTraits<v_int16>::vlanes());

            v_store((schar*)(dst + x), v_pack(v_and(v_ge(values1, low1), v_ge(high1, values1)), v_and(v_ge(values2, low2), v_ge(high2, values2))));
        }
        vx_cleanup();
        return x;
    }
};

template <>
struct InRange_SIMD<int>
{
    int operator () (const int * src1, const int * src2, const int * src3,
        uchar * dst, int len) const
    {
        int x = 0;
        const int width = (int)VTraits<v_int32>::vlanes() * 2;

        for (; x <= len - width; x += width)
        {
            v_int32 values1 = vx_load(src1 + x);
            v_int32 low1 = vx_load(src2 + x);
            v_int32 high1 = vx_load(src3 + x);

            v_int32 values2 = vx_load(src1 + x + VTraits<v_int32>::vlanes());
            v_int32 low2 = vx_load(src2 + x + VTraits<v_int32>::vlanes());
            v_int32 high2 = vx_load(src3 + x + VTraits<v_int32>::vlanes());

            v_pack_store(dst + x, v_reinterpret_as_u16(v_pack(v_and(v_ge(values1, low1), v_ge(high1, values1)), v_and(v_ge(values2, low2), v_ge(high2, values2)))));
        }
        vx_cleanup();
        return x;
    }
};

template <>
struct InRange_SIMD<float>
{
    int operator () (const float * src1, const float * src2, const float * src3,
        uchar * dst, int len) const
    {
        int x = 0;
        const int width = (int)VTraits<v_float32>::vlanes() * 2;

        for (; x <= len - width; x += width)
        {
            v_float32 values1 = vx_load(src1 + x);
            v_float32 low1 = vx_load(src2 + x);
            v_float32 high1 = vx_load(src3 + x);

            v_float32 values2 = vx_load(src1 + x + VTraits<v_float32>::vlanes());
            v_float32 low2 = vx_load(src2 + x + VTraits<v_float32>::vlanes());
            v_float32 high2 = vx_load(src3 + x + VTraits<v_float32>::vlanes());

            v_pack_store(dst + x, v_pack(v_and(v_reinterpret_as_u32(v_ge(values1, low1)), v_reinterpret_as_u32(v_ge(high1, values1))),
                                         v_and(v_reinterpret_as_u32(v_ge(values2, low2)), v_reinterpret_as_u32(v_ge(high2, values2)))));
        }
        vx_cleanup();
        return x;
    }
};

template <>
struct InRange_SIMD<hfloat>
{
    int operator () (const hfloat * src1, const hfloat * src2, const hfloat * src3,
        uchar * dst, int len) const
    {
        int x = 0;
        const int width = (int)VTraits<v_float32>::vlanes()*2;

        for (; x <= len - width; x += width)
        {
            v_float32 values1 = vx_load_expand(src1 + x);
            v_float32 low1 = vx_load_expand(src2 + x);
            v_float32 high1 = vx_load_expand(src3 + x);

            v_float32 values2 = vx_load_expand(src1 + x + VTraits<v_float32>::vlanes());
            v_float32 low2 = vx_load_expand(src2 + x + VTraits<v_float32>::vlanes());
            v_float32 high2 = vx_load_expand(src3 + x + VTraits<v_float32>::vlanes());

            v_pack_store(dst + x, v_pack(v_and(v_reinterpret_as_u32(v_ge(values1, low1)), v_reinterpret_as_u32(v_ge(high1, values1))),
                                         v_and(v_reinterpret_as_u32(v_ge(values2, low2)), v_reinterpret_as_u32(v_ge(high2, values2)))));
        }
        vx_cleanup();
        return x;
    }
};

template <>
struct InRange_SIMD<bfloat>
{
    int operator () (const bfloat * src1, const bfloat * src2, const bfloat * src3,
        uchar * dst, int len) const
    {
        int x = 0;
        const int width = (int)VTraits<v_float32>::vlanes()*2;

        for (; x <= len - width; x += width)
        {
            v_float32 values1 = vx_load_expand(src1 + x);
            v_float32 low1 = vx_load_expand(src2 + x);
            v_float32 high1 = vx_load_expand(src3 + x);

            v_float32 values2 = vx_load_expand(src1 + x + VTraits<v_float32>::vlanes());
            v_float32 low2 = vx_load_expand(src2 + x + VTraits<v_float32>::vlanes());
            v_float32 high2 = vx_load_expand(src3 + x + VTraits<v_float32>::vlanes());

            v_pack_store(dst + x, v_pack(v_and(v_reinterpret_as_u32(v_ge(values1, low1)), v_reinterpret_as_u32(v_ge(high1, values1))),
                                         v_and(v_reinterpret_as_u32(v_ge(values2, low2)), v_reinterpret_as_u32(v_ge(high2, values2)))));
        }
        vx_cleanup();
        return x;
    }
};

template <>
struct InRange_SIMD<unsigned>
{
    int operator () (const unsigned * src1, const unsigned * src2, const unsigned * src3,
        uchar * dst, int len) const
    {
        int x = 0;
        const int width = (int)VTraits<v_uint32>::vlanes() * 2;

        for (; x <= len - width; x += width)
        {
            v_uint32 values1 = vx_load(src1 + x);
            v_uint32 low1 = vx_load(src2 + x);
            v_uint32 high1 = vx_load(src3 + x);

            v_uint32 values2 = vx_load(src1 + x + VTraits<v_uint32>::vlanes());
            v_uint32 low2 = vx_load(src2 + x + VTraits<v_uint32>::vlanes());
            v_uint32 high2 = vx_load(src3 + x + VTraits<v_uint32>::vlanes());

            v_pack_store(dst + x, v_reinterpret_as_u16(v_pack(v_and(v_ge(values1, low1), v_ge(high1, values1)), v_and(v_ge(values2, low2), v_ge(high2, values2)))));
        }
        vx_cleanup();
        return x;
    }
};

#if CV_SIMD_64F

template <>
struct InRange_SIMD<double>
{
    int operator () (const double * src1, const double * src2, const double * src3,
        uchar * dst, int len) const
    {
        int x = 0;
        const int step = VTraits<v_float64>::vlanes();
        const int width = step * 4;

        for (; x <= len - width; x += width)
        {
            v_float64 v1 = vx_load(src1 + x);
            v_float64 l1 = vx_load(src2 + x);
            v_float64 h1 = vx_load(src3 + x);
            v_uint64 m1 = v_reinterpret_as_u64(v_and(v_ge(v1, l1), v_ge(h1, v1)));

            v_float64 v2 = vx_load(src1 + x + step);
            v_float64 l2 = vx_load(src2 + x + step);
            v_float64 h2 = vx_load(src3 + x + step);
            v_uint64 m2 = v_reinterpret_as_u64(v_and(v_ge(v2, l2), v_ge(h2, v2)));

            v_float64 v3 = vx_load(src1 + x + step * 2);
            v_float64 l3 = vx_load(src2 + x + step * 2);
            v_float64 h3 = vx_load(src3 + x + step * 2);
            v_uint64 m3 = v_reinterpret_as_u64(v_and(v_ge(v3, l3), v_ge(h3, v3)));

            v_float64 v4 = vx_load(src1 + x + step * 3);
            v_float64 l4 = vx_load(src2 + x + step * 3);
            v_float64 h4 = vx_load(src3 + x + step * 3);
            v_uint64 m4 = v_reinterpret_as_u64(v_and(v_ge(v4, l4), v_ge(h4, v4)));


            v_pack_store(dst + x, v_pack(v_pack(m1, m2), v_pack(m3, m4)));
        }
        vx_cleanup();
        return x;
    }
};

#endif

#endif

template <typename T>
static void inRange_(const T* src1, size_t step1, const T* src2, size_t step2,
         const T* src3, size_t step3, uchar* dst, size_t step,
         Size size)
{
    step1 /= sizeof(src1[0]);
    step2 /= sizeof(src2[0]);
    step3 /= sizeof(src3[0]);

    InRange_SIMD<T> vop;

    for( ; size.height--; src1 += step1, src2 += step2, src3 += step3, dst += step )
    {
        int x = vop(src1, src2, src3, dst, size.width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            int t0, t1;
            t0 = src2[x] <= src1[x] && src1[x] <= src3[x];
            t1 = src2[x+1] <= src1[x+1] && src1[x+1] <= src3[x+1];
            dst[x] = (uchar)-t0; dst[x+1] = (uchar)-t1;
            t0 = src2[x+2] <= src1[x+2] && src1[x+2] <= src3[x+2];
            t1 = src2[x+3] <= src1[x+3] && src1[x+3] <= src3[x+3];
            dst[x+2] = (uchar)-t0; dst[x+3] = (uchar)-t1;
        }
        #endif
        for( ; x < size.width; x++ )
            dst[x] = (uchar)-(src2[x] <= src1[x] && src1[x] <= src3[x]);
    }
}


static void inRange8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                      const uchar* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange8s(const schar* src1, size_t step1, const schar* src2, size_t step2,
                      const schar* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange16u(const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                       const ushort* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange16s(const short* src1, size_t step1, const short* src2, size_t step2,
                       const short* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange32u(const unsigned* src1, size_t step1, const unsigned* src2, size_t step2,
                       const unsigned* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange32s(const int* src1, size_t step1, const int* src2, size_t step2,
                       const int* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange64u(const uint64* src1, size_t step1, const uint64* src2, size_t step2,
                       const uint64* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange64s(const int64* src1, size_t step1, const int64* src2, size_t step2,
                       const int64* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange32f(const float* src1, size_t step1, const float* src2, size_t step2,
                       const float* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange64f(const double* src1, size_t step1, const double* src2, size_t step2,
                       const double* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange16f(const hfloat* src1, size_t step1, const hfloat* src2, size_t step2,
                       const hfloat* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange16bf(const bfloat* src1, size_t step1, const bfloat* src2, size_t step2,
                        const bfloat* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRangeReduce(const uchar* src, uchar* dst, size_t len, int cn)
{
    int k = cn % 4 ? cn % 4 : 4;
    size_t i, j;
    if( k == 1 )
        for( i = j = 0; i < len; i++, j += cn )
            dst[i] = src[j];
    else if( k == 2 )
        for( i = j = 0; i < len; i++, j += cn )
            dst[i] = src[j] & src[j+1];
    else if( k == 3 )
        for( i = j = 0; i < len; i++, j += cn )
            dst[i] = src[j] & src[j+1] & src[j+2];
    else
        for( i = j = 0; i < len; i++, j += cn )
            dst[i] = src[j] & src[j+1] & src[j+2] & src[j+3];

    for( ; k < cn; k += 4 )
    {
        for( i = 0, j = k; i < len; i++, j += cn )
            dst[i] &= src[j] & src[j+1] & src[j+2] & src[j+3];
    }
}

typedef void (*InRangeFunc)( const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                             const uchar* src3, size_t step3, uchar* dst, size_t step, Size sz );

static InRangeFunc getInRangeFunc(int depth)
{
    static InRangeFunc inRangeTab[CV_DEPTH_MAX] =
    {
        (InRangeFunc)GET_OPTIMIZED(inRange8u),
        (InRangeFunc)GET_OPTIMIZED(inRange8s),
        (InRangeFunc)GET_OPTIMIZED(inRange16u),
        (InRangeFunc)GET_OPTIMIZED(inRange16s),
        (InRangeFunc)GET_OPTIMIZED(inRange32s),
        (InRangeFunc)GET_OPTIMIZED(inRange32f),
        (InRangeFunc)GET_OPTIMIZED(inRange64f),
        (InRangeFunc)inRange16f,
        (InRangeFunc)inRange16bf,
        0,
        (InRangeFunc)GET_OPTIMIZED(inRange64u),
        (InRangeFunc)GET_OPTIMIZED(inRange64s),
        (InRangeFunc)GET_OPTIMIZED(inRange32u),
        0,
    };

    return inRangeTab[depth];
}

#ifdef HAVE_OPENCL

static bool ocl_inRange( InputArray _src, InputArray _lowerb,
                         InputArray _upperb, OutputArray _dst )
{
    const ocl::Device & d = ocl::Device::getDefault();
    _InputArray::KindFlag skind = _src.kind(), lkind = _lowerb.kind(), ukind = _upperb.kind();
    Size ssize = _src.size(), lsize = _lowerb.size(), usize = _upperb.size();
    int stype = _src.type(), ltype = _lowerb.type(), utype = _upperb.type();
    int sdepth = CV_MAT_DEPTH(stype), ldepth = CV_MAT_DEPTH(ltype), udepth = CV_MAT_DEPTH(utype);
    int cn = CV_MAT_CN(stype), rowsPerWI = d.isIntel() ? 4 : 1;
    bool lbScalar = false, ubScalar = false;

    if( (lkind == _InputArray::MATX && skind != _InputArray::MATX) ||
        ssize != lsize || stype != ltype )
    {
        if( !checkScalar(_lowerb, stype, lkind, skind) )
            CV_Error( cv::Error::StsUnmatchedSizes,
                     "The lower boundary is neither an array of the same size and same type as src, nor a scalar");
        lbScalar = true;
    }

    if( (ukind == _InputArray::MATX && skind != _InputArray::MATX) ||
        ssize != usize || stype != utype )
    {
        if( !checkScalar(_upperb, stype, ukind, skind) )
            CV_Error( cv::Error::StsUnmatchedSizes,
                     "The upper boundary is neither an array of the same size and same type as src, nor a scalar");
        ubScalar = true;
    }

    if (lbScalar != ubScalar)
        return false;

    bool doubleSupport = d.doubleFPConfig() > 0,
            haveScalar = lbScalar && ubScalar;

    if ( (!doubleSupport && sdepth == CV_64F) ||
         (!haveScalar && (sdepth != ldepth || sdepth != udepth)) )
        return false;

    int kercn = haveScalar ? cn : std::max(std::min(ocl::predictOptimalVectorWidth(_src, _lowerb, _upperb, _dst), 4), cn);
    if (kercn % cn != 0)
        kercn = cn;
    int colsPerWI = kercn / cn;
    String opts = format("%s-D CN=%d -D SRC_T=%s -D SRC_T1=%s -D DST_T=%s -D KERCN=%d -D DEPTH=%d%s -D COLS_PER_WI=%d",
                           haveScalar ? "-D HAVE_SCALAR " : "", cn, ocl::typeToStr(CV_MAKE_TYPE(sdepth, kercn)),
                           ocl::typeToStr(sdepth), ocl::typeToStr(CV_8UC(colsPerWI)), kercn, sdepth,
                           doubleSupport ? " -D DOUBLE_SUPPORT" : "", colsPerWI);

    ocl::Kernel ker("inrange", ocl::core::inrange_oclsrc, opts);
    if (ker.empty())
        return false;

    _dst.create(ssize, CV_8UC1);
    UMat src = _src.getUMat(), dst = _dst.getUMat(), lscalaru, uscalaru;
    Mat lscalar, uscalar;

    if (lbScalar && ubScalar)
    {
        lscalar = _lowerb.getMat();
        uscalar = _upperb.getMat();

        size_t esz = src.elemSize();
        size_t blocksize = 36;

        AutoBuffer<uchar> _buf(blocksize*(((int)lbScalar + (int)ubScalar)*esz + cn) + 2*cn*sizeof(int) + 128);
        uchar *buf = alignPtr(_buf.data() + blocksize*cn, 16);

        if( ldepth != sdepth && sdepth < CV_32S )
        {
            int* ilbuf = (int*)alignPtr(buf + blocksize*esz, 16);
            int* iubuf = ilbuf + cn;

            BinaryFunc sccvtfunc = getConvertFunc(ldepth, CV_32S);
            sccvtfunc(lscalar.ptr(), 1, 0, 1, (uchar*)ilbuf, 1, Size(cn, 1), 0);
            sccvtfunc(uscalar.ptr(), 1, 0, 1, (uchar*)iubuf, 1, Size(cn, 1), 0);
            int minval = cvRound(getMinVal(sdepth)), maxval = cvRound(getMaxVal(sdepth));

            for( int k = 0; k < cn; k++ )
            {
                if( ilbuf[k] > iubuf[k] || ilbuf[k] > maxval || iubuf[k] < minval )
                    ilbuf[k] = minval+1, iubuf[k] = minval;
            }
            lscalar = Mat(cn, 1, CV_32S, ilbuf);
            uscalar = Mat(cn, 1, CV_32S, iubuf);
        }

        lscalar.convertTo(lscalar, stype);
        uscalar.convertTo(uscalar, stype);
    }
    else
    {
        lscalaru = _lowerb.getUMat();
        uscalaru = _upperb.getUMat();
    }

    ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
            dstarg = ocl::KernelArg::WriteOnly(dst, 1, colsPerWI);

    if (haveScalar)
    {
        lscalar.copyTo(lscalaru);
        uscalar.copyTo(uscalaru);

        ker.args(srcarg, dstarg, ocl::KernelArg::PtrReadOnly(lscalaru),
               ocl::KernelArg::PtrReadOnly(uscalaru), rowsPerWI);
    }
    else
        ker.args(srcarg, dstarg, ocl::KernelArg::ReadOnlyNoSize(lscalaru),
               ocl::KernelArg::ReadOnlyNoSize(uscalaru), rowsPerWI);

    size_t globalsize[2] = { (size_t)ssize.width / colsPerWI, ((size_t)ssize.height + rowsPerWI - 1) / rowsPerWI };
    return ker.run(2, globalsize, NULL, false);
}

#endif

}

void cv::inRange(InputArray _src, InputArray _lowerb,
                 InputArray _upperb, OutputArray _dst)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(! _src.empty());

    CV_OCL_RUN(_src.dims() <= 2 && _lowerb.dims() <= 2 &&
               _upperb.dims() <= 2 && OCL_PERFORMANCE_CHECK(_dst.isUMat()),
               ocl_inRange(_src, _lowerb, _upperb, _dst))

    _InputArray::KindFlag skind = _src.kind(), lkind = _lowerb.kind(), ukind = _upperb.kind();
    Mat src = _src.getMat(), lb = _lowerb.getMat(), ub = _upperb.getMat();

    bool lbScalar = false, ubScalar = false;

    if( (lkind == _InputArray::MATX && skind != _InputArray::MATX) ||
        src.size != lb.size || src.type() != lb.type() )
    {
        if( !checkScalar(lb, src.type(), lkind, skind) )
            CV_Error( cv::Error::StsUnmatchedSizes,
                     "The lower boundary is neither an array of the same size and same type as src, nor a scalar");
        lbScalar = true;
    }

    if( (ukind == _InputArray::MATX && skind != _InputArray::MATX) ||
        src.size != ub.size || src.type() != ub.type() )
    {
        if( !checkScalar(ub, src.type(), ukind, skind) )
            CV_Error( cv::Error::StsUnmatchedSizes,
                     "The upper boundary is neither an array of the same size and same type as src, nor a scalar");
        ubScalar = true;
    }

    CV_Assert(lbScalar == ubScalar);

    int cn = src.channels(), depth = src.depth();

    size_t esz = src.elemSize();
    size_t blocksize0 = (size_t)(BLOCK_SIZE + esz-1)/esz;

    _dst.createSameSize(_src, CV_8UC1);
    Mat dst = _dst.getMat();
    InRangeFunc func = getInRangeFunc(depth);

    const Mat* arrays_sc[] = { &src, &dst, 0 };
    const Mat* arrays_nosc[] = { &src, &dst, &lb, &ub, 0 };
    uchar* ptrs[4] = {};

    NAryMatIterator it(lbScalar && ubScalar ? arrays_sc : arrays_nosc, ptrs);
    size_t total = it.size, blocksize = std::min(total, blocksize0);

    AutoBuffer<uchar> _buf(blocksize*(((int)lbScalar + (int)ubScalar)*esz + cn) + 2*cn*sizeof(int) + 128);
    uchar *buf = _buf.data(), *mbuf = buf, *lbuf = 0, *ubuf = 0;
    buf = alignPtr(buf + blocksize*cn, 16);

    if( lbScalar && ubScalar )
    {
        lbuf = buf;
        ubuf = buf = alignPtr(buf + blocksize*esz, 16);

        CV_Assert( lb.type() == ub.type() );
        int scdepth = lb.depth();

        if( scdepth != depth && depth < CV_32S )
        {
            int* ilbuf = (int*)alignPtr(buf + blocksize*esz, 16);
            int* iubuf = ilbuf + cn;

            BinaryFunc sccvtfunc = getConvertFunc(scdepth, CV_32S);
            sccvtfunc(lb.ptr(), 1, 0, 1, (uchar*)ilbuf, 1, Size(cn, 1), 0);
            sccvtfunc(ub.ptr(), 1, 0, 1, (uchar*)iubuf, 1, Size(cn, 1), 0);
            int minval = cvRound(getMinVal(depth)), maxval = cvRound(getMaxVal(depth));

            for( int k = 0; k < cn; k++ )
            {
                if( ilbuf[k] > iubuf[k] || ilbuf[k] > maxval || iubuf[k] < minval )
                    ilbuf[k] = minval+1, iubuf[k] = minval;
            }
            lb = Mat(cn, 1, CV_32S, ilbuf);
            ub = Mat(cn, 1, CV_32S, iubuf);
        }

        convertAndUnrollScalar( lb, src.type(), lbuf, blocksize );
        convertAndUnrollScalar( ub, src.type(), ubuf, blocksize );

        if (depth == CV_8U && src.dims <= 2) {
            uint8_t lb_scalar = lbuf[0];
            uint8_t ub_scalar = ubuf[0];
            CALL_HAL(inRange_u8, cv_hal_inRange8u, src.data, src.step, dst.data, dst.step, dst.depth(), src.cols, src.rows, src.channels(),
                lb_scalar, ub_scalar);
        } else if (depth == CV_32F && src.dims <= 2) {
            double lb_scalar = lb.ptr<double>(0)[0];
            double ub_scalar = ub.ptr<double>(0)[0];
            CALL_HAL(inRange_f32, cv_hal_inRange32f, src.data, src.step, dst.data, dst.step, dst.depth(), src.cols, src.rows, src.channels(),
                lb_scalar, ub_scalar);
        }
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( size_t j = 0; j < total; j += blocksize )
        {
            int bsz = (int)MIN(total - j, blocksize);
            size_t delta = bsz*esz;
            uchar *lptr = lbuf, *uptr = ubuf;
            if( !lbScalar )
            {
                lptr = ptrs[2];
                ptrs[2] += delta;
            }
            if( !ubScalar )
            {
                int idx = !lbScalar ? 3 : 2;
                uptr = ptrs[idx];
                ptrs[idx] += delta;
            }
            func( ptrs[0], 0, lptr, 0, uptr, 0, cn == 1 ? ptrs[1] : mbuf, 0, Size(bsz*cn, 1));
            if( cn > 1 )
                inRangeReduce(mbuf, ptrs[1], bsz, cn);
            ptrs[0] += delta;
            ptrs[1] += bsz;
        }
    }
}
/* End of file. */
