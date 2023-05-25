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
            CV_Error( CV_StsUnmatchedSizes,
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
        CV_Assert( (mtype == CV_8U || mtype == CV_8S) && _mask.sameSize(*psrc1));
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
    static BinaryFuncC maxTab[] =
    {
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::max8u), (BinaryFuncC)GET_OPTIMIZED(cv::hal::max8s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::max16u), (BinaryFuncC)GET_OPTIMIZED(cv::hal::max16s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::max32s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::max32f), (BinaryFuncC)cv::hal::max64f,
        0
    };

    return maxTab;
}

static BinaryFuncC* getMinTab()
{
    static BinaryFuncC minTab[] =
    {
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::min8u), (BinaryFuncC)GET_OPTIMIZED(cv::hal::min8s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::min16u), (BinaryFuncC)GET_OPTIMIZED(cv::hal::min16s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::min32s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::min32f), (BinaryFuncC)cv::hal::min64f,
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

    binary_op(src1, src2, dst, noArray(), getMaxTab(), false, OCL_OP_MAX );
}

void cv::min( InputArray src1, InputArray src2, OutputArray dst )
{
    CV_INSTRUMENT_REGION();

    binary_op(src1, src2, dst, noArray(), getMinTab(), false, OCL_OP_MIN );
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

static void arithm_op(InputArray _src1, InputArray _src2, OutputArray _dst,
                      InputArray _mask, int dtype, BinaryFuncC* tab, bool muldiv=false,
                      void* usrdata=0, int oclop=-1 )
{
    const _InputArray *psrc1 = &_src1, *psrc2 = &_src2;
    _InputArray::KindFlag kind1 = psrc1->kind(), kind2 = psrc2->kind();
    bool haveMask = !_mask.empty();
    bool reallocate = false;
    int type1 = psrc1->type(), depth1 = CV_MAT_DEPTH(type1), cn = CV_MAT_CN(type1);
    int type2 = psrc2->type(), depth2 = CV_MAT_DEPTH(type2), cn2 = CV_MAT_CN(type2);
    int wtype, dims1 = psrc1->dims(), dims2 = psrc2->dims();
    Size sz1 = dims1 <= 2 ? psrc1->size() : Size();
    Size sz2 = dims2 <= 2 ? psrc2->size() : Size();
#ifdef HAVE_OPENCL
    bool use_opencl = OCL_PERFORMANCE_CHECK(_dst.isUMat()) && dims1 <= 2 && dims2 <= 2;
#endif
    bool src1Scalar = checkScalar(*psrc1, type2, kind1, kind2);
    bool src2Scalar = checkScalar(*psrc2, type1, kind2, kind1);

    if( (kind1 == kind2 || cn == 1) && sz1 == sz2 && dims1 <= 2 && dims2 <= 2 && type1 == type2 &&
        !haveMask && ((!_dst.fixedType() && (dtype < 0 || CV_MAT_DEPTH(dtype) == depth1)) ||
                       (_dst.fixedType() && _dst.type() == type1)) &&
        (src1Scalar == src2Scalar) )
    {
        _dst.createSameSize(*psrc1, type1);
        CV_OCL_RUN(use_opencl,
            ocl_arithm_op(*psrc1, *psrc2, _dst, _mask,
                          (!usrdata ? type1 : std::max(depth1, CV_32F)),
                          usrdata, oclop, false))

        Mat src1 = psrc1->getMat(), src2 = psrc2->getMat(), dst = _dst.getMat();
        Size sz = getContinuousSize2D(src1, src2, dst, src1.channels());
        tab[depth1](src1.ptr(), src1.step, src2.ptr(), src2.step, dst.ptr(), dst.step, sz.width, sz.height, usrdata);
        return;
    }

    bool haveScalar = false, swapped12 = false;

    if( dims1 != dims2 || sz1 != sz2 || cn != cn2 ||
        (kind1 == _InputArray::MATX && (sz1 == Size(1,4) || sz1 == Size(1,1))) ||
        (kind2 == _InputArray::MATX && (sz2 == Size(1,4) || sz2 == Size(1,1))) )
    {
        if ((type1 == CV_64F && (sz1.height == 1 || sz1.height == 4)) &&
            checkScalar(*psrc1, type2, kind1, kind2))
        {
            // src1 is a scalar; swap it with src2
            swap(psrc1, psrc2);
            swap(sz1, sz2);
            swap(type1, type2);
            swap(depth1, depth2);
            swap(cn, cn2);
            swap(dims1, dims2);
            swapped12 = true;
            if( oclop == OCL_OP_SUB )
                oclop = OCL_OP_RSUB;
            if ( oclop == OCL_OP_DIV_SCALE )
                oclop = OCL_OP_RDIV_SCALE;
        }
        else if( !checkScalar(*psrc2, type1, kind2, kind1) )
            CV_Error( CV_StsUnmatchedSizes,
                     "The operation is neither 'array op array' "
                     "(where arrays have the same size and the same number of channels), "
                     "nor 'array op scalar', nor 'scalar op array'" );
        haveScalar = true;
        CV_Assert(type2 == CV_64F && (sz2.height == 1 || sz2.height == 4));

        if (!muldiv)
        {
            Mat sc = psrc2->getMat();
            depth2 = actualScalarDepth(sc.ptr<double>(), sz2 == Size(1, 1) ? cn2 : cn);
            if( depth2 == CV_64F && (depth1 < CV_32S || depth1 == CV_32F) )
                depth2 = CV_32F;
        }
        else
            depth2 = CV_64F;
    }

    if( dtype < 0 )
    {
        if( _dst.fixedType() )
            dtype = _dst.type();
        else
        {
            if( !haveScalar && type1 != type2 )
                CV_Error(CV_StsBadArg,
                     "When the input arrays in add/subtract/multiply/divide functions have different types, "
                     "the output array type must be explicitly specified");
            dtype = type1;
        }
    }
    dtype = CV_MAT_DEPTH(dtype);

    if( depth1 == depth2 && dtype == depth1 )
        wtype = dtype;
    else if( !muldiv )
    {
        wtype = depth1 <= CV_8S && depth2 <= CV_8S ? CV_16S :
                depth1 <= CV_32S && depth2 <= CV_32S ? CV_32S : std::max(depth1, depth2);
        wtype = std::max(wtype, dtype);

        // when the result of addition should be converted to an integer type,
        // and just one of the input arrays is floating-point, it makes sense to convert that input to integer type before the operation,
        // instead of converting the other input to floating-point and then converting the operation result back to integers.
        if( dtype < CV_32F && (depth1 < CV_32F || depth2 < CV_32F) )
            wtype = CV_32S;
    }
    else
    {
        wtype = std::max(depth1, std::max(depth2, CV_32F));
        wtype = std::max(wtype, dtype);
    }

    dtype = CV_MAKETYPE(dtype, cn);
    wtype = CV_MAKETYPE(wtype, cn);

    if( haveMask )
    {
        int mtype = _mask.type();
        CV_Assert( (mtype == CV_8UC1 || mtype == CV_8SC1) && _mask.sameSize(*psrc1) );
        reallocate = !_dst.sameSize(*psrc1) || _dst.type() != dtype;
    }

    _dst.createSameSize(*psrc1, dtype);
    if( reallocate )
        _dst.setTo(0.);

    CV_OCL_RUN(use_opencl,
               ocl_arithm_op(*psrc1, *psrc2, _dst, _mask, wtype,
               usrdata, oclop, haveScalar))

    BinaryFunc cvtsrc1 = type1 == wtype ? 0 : getConvertFunc(type1, wtype);
    BinaryFunc cvtsrc2 = type2 == type1 ? cvtsrc1 : type2 == wtype ? 0 : getConvertFunc(type2, wtype);
    BinaryFunc cvtdst = dtype == wtype ? 0 : getConvertFunc(wtype, dtype);

    size_t esz1 = CV_ELEM_SIZE(type1), esz2 = CV_ELEM_SIZE(type2);
    size_t dsz = CV_ELEM_SIZE(dtype), wsz = CV_ELEM_SIZE(wtype);
    size_t blocksize0 = (size_t)(BLOCK_SIZE + wsz-1)/wsz;
    BinaryFunc copymask = getCopyMaskFunc(dsz);
    Mat src1 = psrc1->getMat(), src2 = psrc2->getMat(), dst = _dst.getMat(), mask = _mask.getMat();

    AutoBuffer<uchar> _buf;
    uchar *buf, *maskbuf = 0, *buf1 = 0, *buf2 = 0, *wbuf = 0;
    size_t bufesz = (cvtsrc1 ? wsz : 0) +
                    (cvtsrc2 || haveScalar ? wsz : 0) +
                    (cvtdst ? wsz : 0) +
                    (haveMask ? dsz : 0);
    BinaryFuncC func = tab[CV_MAT_DEPTH(wtype)];
    CV_Assert(func);

    if( !haveScalar )
    {
        const Mat* arrays[] = { &src1, &src2, &dst, &mask, 0 };
        uchar* ptrs[4] = {};

        NAryMatIterator it(arrays, ptrs);
        size_t total = it.size, blocksize = total;

        if( haveMask || cvtsrc1 || cvtsrc2 || cvtdst )
            blocksize = std::min(blocksize, blocksize0);

        _buf.allocate(bufesz*blocksize + 64);
        buf = _buf.data();
        if( cvtsrc1 )
            buf1 = buf, buf = alignPtr(buf + blocksize*wsz, 16);
        if( cvtsrc2 )
            buf2 = buf, buf = alignPtr(buf + blocksize*wsz, 16);
        wbuf = maskbuf = buf;
        if( cvtdst )
            buf = alignPtr(buf + blocksize*wsz, 16);
        if( haveMask )
            maskbuf = buf;

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            for( size_t j = 0; j < total; j += blocksize )
            {
                int bsz = (int)MIN(total - j, blocksize);
                Size bszn(bsz*cn, 1);
                const uchar *sptr1 = ptrs[0], *sptr2 = ptrs[1];
                uchar* dptr = ptrs[2];
                if( cvtsrc1 )
                {
                    cvtsrc1( sptr1, 1, 0, 1, buf1, 1, bszn, 0 );
                    sptr1 = buf1;
                }
                if( ptrs[0] == ptrs[1] )
                    sptr2 = sptr1;
                else if( cvtsrc2 )
                {
                    cvtsrc2( sptr2, 1, 0, 1, buf2, 1, bszn, 0 );
                    sptr2 = buf2;
                }

                if( !haveMask && !cvtdst )
                    func( sptr1, 1, sptr2, 1, dptr, 1, bszn.width, bszn.height, usrdata );
                else
                {
                    func( sptr1, 1, sptr2, 1, wbuf, 0, bszn.width, bszn.height, usrdata );
                    if( !haveMask )
                        cvtdst( wbuf, 1, 0, 1, dptr, 1, bszn, 0 );
                    else if( !cvtdst )
                    {
                        copymask( wbuf, 1, ptrs[3], 1, dptr, 1, Size(bsz, 1), &dsz );
                        ptrs[3] += bsz;
                    }
                    else
                    {
                        cvtdst( wbuf, 1, 0, 1, maskbuf, 1, bszn, 0 );
                        copymask( maskbuf, 1, ptrs[3], 1, dptr, 1, Size(bsz, 1), &dsz );
                        ptrs[3] += bsz;
                    }
                }
                ptrs[0] += bsz*esz1; ptrs[1] += bsz*esz2; ptrs[2] += bsz*dsz;
            }
        }
    }
    else
    {
        const Mat* arrays[] = { &src1, &dst, &mask, 0 };
        uchar* ptrs[3] = {};

        NAryMatIterator it(arrays, ptrs);
        size_t total = it.size, blocksize = std::min(total, blocksize0);

        _buf.allocate(bufesz*blocksize + 64);
        buf = _buf.data();
        if( cvtsrc1 )
            buf1 = buf, buf = alignPtr(buf + blocksize*wsz, 16);
        buf2 = buf; buf = alignPtr(buf + blocksize*wsz, 16);
        wbuf = maskbuf = buf;
        if( cvtdst )
            buf = alignPtr(buf + blocksize*wsz, 16);
        if( haveMask )
            maskbuf = buf;

        convertAndUnrollScalar( src2, wtype, buf2, blocksize);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            for( size_t j = 0; j < total; j += blocksize )
            {
                int bsz = (int)MIN(total - j, blocksize);
                Size bszn(bsz*cn, 1);
                const uchar *sptr1 = ptrs[0];
                const uchar* sptr2 = buf2;
                uchar* dptr = ptrs[1];

                if( cvtsrc1 )
                {
                    cvtsrc1( sptr1, 1, 0, 1, buf1, 1, bszn, 0 );
                    sptr1 = buf1;
                }

                if( swapped12 )
                    std::swap(sptr1, sptr2);

                if( !haveMask && !cvtdst )
                    func( sptr1, 1, sptr2, 1, dptr, 1, bszn.width, bszn.height, usrdata );
                else
                {
                    func( sptr1, 1, sptr2, 1, wbuf, 1, bszn.width, bszn.height, usrdata );
                    if( !haveMask )
                        cvtdst( wbuf, 1, 0, 1, dptr, 1, bszn, 0 );
                    else if( !cvtdst )
                    {
                        copymask( wbuf, 1, ptrs[2], 1, dptr, 1, Size(bsz, 1), &dsz );
                        ptrs[2] += bsz;
                    }
                    else
                    {
                        cvtdst( wbuf, 1, 0, 1, maskbuf, 1, bszn, 0 );
                        copymask( maskbuf, 1, ptrs[2], 1, dptr, 1, Size(bsz, 1), &dsz );
                        ptrs[2] += bsz;
                    }
                }
                ptrs[0] += bsz*esz1; ptrs[1] += bsz*dsz;
            }
        }
    }
}

static BinaryFuncC* getAddTab()
{
    static BinaryFuncC addTab[] =
    {
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::add8u), (BinaryFuncC)GET_OPTIMIZED(cv::hal::add8s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::add16u), (BinaryFuncC)GET_OPTIMIZED(cv::hal::add16s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::add32s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::add32f), (BinaryFuncC)cv::hal::add64f,
        0
    };

    return addTab;
}

static BinaryFuncC* getSubTab()
{
    static BinaryFuncC subTab[] =
    {
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::sub8u), (BinaryFuncC)GET_OPTIMIZED(cv::hal::sub8s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::sub16u), (BinaryFuncC)GET_OPTIMIZED(cv::hal::sub16s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::sub32s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::sub32f), (BinaryFuncC)cv::hal::sub64f,
        0
    };

    return subTab;
}

static BinaryFuncC* getAbsDiffTab()
{
    static BinaryFuncC absDiffTab[] =
    {
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::absdiff8u), (BinaryFuncC)GET_OPTIMIZED(cv::hal::absdiff8s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::absdiff16u), (BinaryFuncC)GET_OPTIMIZED(cv::hal::absdiff16s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::absdiff32s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::absdiff32f), (BinaryFuncC)cv::hal::absdiff64f,
        0
    };

    return absDiffTab;
}

}

void cv::add( InputArray src1, InputArray src2, OutputArray dst,
          InputArray mask, int dtype )
{
    CV_INSTRUMENT_REGION();

    arithm_op(src1, src2, dst, mask, dtype, getAddTab(), false, 0, OCL_OP_ADD );
}

void cv::subtract( InputArray _src1, InputArray _src2, OutputArray _dst,
               InputArray mask, int dtype )
{
    CV_INSTRUMENT_REGION();

    arithm_op(_src1, _src2, _dst, mask, dtype, getSubTab(), false, 0, OCL_OP_SUB );
}

void cv::absdiff( InputArray src1, InputArray src2, OutputArray dst )
{
    CV_INSTRUMENT_REGION();

    arithm_op(src1, src2, dst, noArray(), -1, getAbsDiffTab(), false, 0, OCL_OP_ABSDIFF);
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

static BinaryFuncC* getMulTab()
{
    static BinaryFuncC mulTab[] =
    {
        (BinaryFuncC)cv::hal::mul8u, (BinaryFuncC)cv::hal::mul8s, (BinaryFuncC)cv::hal::mul16u,
        (BinaryFuncC)cv::hal::mul16s, (BinaryFuncC)cv::hal::mul32s, (BinaryFuncC)cv::hal::mul32f,
        (BinaryFuncC)cv::hal::mul64f, 0
    };

    return mulTab;
}

static BinaryFuncC* getDivTab()
{
    static BinaryFuncC divTab[] =
    {
        (BinaryFuncC)cv::hal::div8u, (BinaryFuncC)cv::hal::div8s, (BinaryFuncC)cv::hal::div16u,
        (BinaryFuncC)cv::hal::div16s, (BinaryFuncC)cv::hal::div32s, (BinaryFuncC)cv::hal::div32f,
        (BinaryFuncC)cv::hal::div64f, 0
    };

    return divTab;
}

static BinaryFuncC* getRecipTab()
{
    static BinaryFuncC recipTab[] =
    {
        (BinaryFuncC)cv::hal::recip8u, (BinaryFuncC)cv::hal::recip8s, (BinaryFuncC)cv::hal::recip16u,
        (BinaryFuncC)cv::hal::recip16s, (BinaryFuncC)cv::hal::recip32s, (BinaryFuncC)cv::hal::recip32f,
        (BinaryFuncC)cv::hal::recip64f, 0
    };

    return recipTab;
}

void multiply(InputArray src1, InputArray src2,
                  OutputArray dst, double scale, int dtype)
{
    CV_INSTRUMENT_REGION();

    arithm_op(src1, src2, dst, noArray(), dtype, getMulTab(),
              true, &scale, std::abs(scale - 1.0) < DBL_EPSILON ? OCL_OP_MUL : OCL_OP_MUL_SCALE);
}

void divide(InputArray src1, InputArray src2,
                OutputArray dst, double scale, int dtype)
{
    CV_INSTRUMENT_REGION();

    arithm_op(src1, src2, dst, noArray(), dtype, getDivTab(), true, &scale, OCL_OP_DIV_SCALE);
}

void divide(double scale, InputArray src2,
                OutputArray dst, int dtype)
{
    CV_INSTRUMENT_REGION();

    arithm_op(src2, src2, dst, noArray(), dtype, getRecipTab(), true, &scale, OCL_OP_RECIP_SCALE);
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

static BinaryFuncC* getAddWeightedTab()
{
    static BinaryFuncC addWeightedTab[] =
    {
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::addWeighted8u), (BinaryFuncC)GET_OPTIMIZED(cv::hal::addWeighted8s), (BinaryFuncC)GET_OPTIMIZED(cv::hal::addWeighted16u),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::addWeighted16s), (BinaryFuncC)GET_OPTIMIZED(cv::hal::addWeighted32s), (BinaryFuncC)cv::hal::addWeighted32f,
        (BinaryFuncC)cv::hal::addWeighted64f, 0
    };

    return addWeightedTab;
}

}

void cv::addWeighted( InputArray src1, double alpha, InputArray src2,
                      double beta, double gamma, OutputArray dst, int dtype )
{
    CV_INSTRUMENT_REGION();

    double scalars[] = {alpha, beta, gamma};
    arithm_op(src1, src2, dst, noArray(), dtype, getAddWeightedTab(), true, scalars, OCL_OP_ADDW);
}


/****************************************************************************************\
*                                          compare                                       *
\****************************************************************************************/

namespace cv
{

static BinaryFuncC getCmpFunc(int depth)
{
    static BinaryFuncC cmpTab[] =
    {
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::cmp8u), (BinaryFuncC)GET_OPTIMIZED(cv::hal::cmp8s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::cmp16u), (BinaryFuncC)GET_OPTIMIZED(cv::hal::cmp16s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::cmp32s),
        (BinaryFuncC)GET_OPTIMIZED(cv::hal::cmp32f), (BinaryFuncC)cv::hal::cmp64f,
        0
    };

    return cmpTab[depth];
}

static double getMinVal(int depth)
{
    static const double tab[] = {0, -128, 0, -32768, INT_MIN, -FLT_MAX, -DBL_MAX, 0};
    return tab[depth];
}

static double getMaxVal(int depth)
{
    static const double tab[] = {255, 127, 65535, 32767, INT_MAX, FLT_MAX, DBL_MAX, 0};
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
            CV_Error( CV_StsUnmatchedSizes,
                     "The operation is neither 'array op array' (where arrays have the same size and the same type), "
                     "nor 'array op scalar', nor 'scalar op array'" );
        haveScalar = true;
    }

    CV_OCL_RUN(_src1.dims() <= 2 && _src2.dims() <= 2 && OCL_PERFORMANCE_CHECK(_dst.isUMat()),
               ocl_compare(_src1, _src2, _dst, op, haveScalar))

    _InputArray::KindFlag kind1 = _src1.kind(), kind2 = _src2.kind();
    Mat src1 = _src1.getMat(), src2 = _src2.getMat();

    int depth1 = src1.depth(), depth2 = src2.depth();
    if (depth1 == CV_16F || depth2 == CV_16F)
        CV_Error(Error::StsNotImplemented, "Unsupported depth value CV_16F");

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

    _dst.create(src1.dims, src1.size, CV_8UC(cn));
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

        if( depth1 > CV_32S )
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

            int ival = cvRound(fval);
            if( fval != ival )
            {
                if( op == CMP_LT || op == CMP_GE )
                    ival = cvCeil(fval);
                else if( op == CMP_LE || op == CMP_GT )
                    ival = cvFloor(fval);
                else
                {
                    dst = Scalar::all(op == CMP_NE ? 255 : 0);
                    return;
                }
            }
            convertAndUnrollScalar(Mat(1, 1, CV_32S, &ival), depth1, buf, blocksize);
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
    int operator () (const T *, const T *, const T *, uchar *, int) const
    {
        return 0;
    }
};

#if CV_SIMD

template <>
struct InRange_SIMD<uchar>
{
    int operator () (const uchar * src1, const uchar * src2, const uchar * src3,
        uchar * dst, int len) const
    {
        int x = 0;
        const int width = v_uint8::nlanes;

        for (; x <= len - width; x += width)
        {
            v_uint8 values = vx_load(src1 + x);
            v_uint8 low = vx_load(src2 + x);
            v_uint8 high = vx_load(src3 + x);

            v_store(dst + x, (values >= low) & (high >= values));
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
        const int width = v_int8::nlanes;

        for (; x <= len - width; x += width)
        {
            v_int8 values = vx_load(src1 + x);
            v_int8 low = vx_load(src2 + x);
            v_int8 high = vx_load(src3 + x);

            v_store((schar*)(dst + x), (values >= low) & (high >= values));
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
        const int width = v_uint16::nlanes * 2;

        for (; x <= len - width; x += width)
        {
            v_uint16 values1 = vx_load(src1 + x);
            v_uint16 low1 = vx_load(src2 + x);
            v_uint16 high1 = vx_load(src3 + x);

            v_uint16 values2 = vx_load(src1 + x + v_uint16::nlanes);
            v_uint16 low2 = vx_load(src2 + x + v_uint16::nlanes);
            v_uint16 high2 = vx_load(src3 + x + v_uint16::nlanes);

            v_store(dst + x, v_pack((values1 >= low1) & (high1 >= values1), (values2 >= low2) & (high2 >= values2)));
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
        const int width = (int)v_int16::nlanes * 2;

        for (; x <= len - width; x += width)
        {
            v_int16 values1 = vx_load(src1 + x);
            v_int16 low1 = vx_load(src2 + x);
            v_int16 high1 = vx_load(src3 + x);

            v_int16 values2 = vx_load(src1 + x + v_int16::nlanes);
            v_int16 low2 = vx_load(src2 + x + v_int16::nlanes);
            v_int16 high2 = vx_load(src3 + x + v_int16::nlanes);

            v_store((schar*)(dst + x), v_pack((values1 >= low1) & (high1 >= values1), (values2 >= low2) & (high2 >= values2)));
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
        const int width = (int)v_int32::nlanes * 2;

        for (; x <= len - width; x += width)
        {
            v_int32 values1 = vx_load(src1 + x);
            v_int32 low1 = vx_load(src2 + x);
            v_int32 high1 = vx_load(src3 + x);

            v_int32 values2 = vx_load(src1 + x + v_int32::nlanes);
            v_int32 low2 = vx_load(src2 + x + v_int32::nlanes);
            v_int32 high2 = vx_load(src3 + x + v_int32::nlanes);

            v_pack_store(dst + x, v_reinterpret_as_u16(v_pack((values1 >= low1) & (high1 >= values1), (values2 >= low2) & (high2 >= values2))));
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
        const int width = (int)v_float32::nlanes * 2;

        for (; x <= len - width; x += width)
        {
            v_float32 values1 = vx_load(src1 + x);
            v_float32 low1 = vx_load(src2 + x);
            v_float32 high1 = vx_load(src3 + x);

            v_float32 values2 = vx_load(src1 + x + v_float32::nlanes);
            v_float32 low2 = vx_load(src2 + x + v_float32::nlanes);
            v_float32 high2 = vx_load(src3 + x + v_float32::nlanes);

            v_pack_store(dst + x, v_pack(v_reinterpret_as_u32(values1 >= low1) & v_reinterpret_as_u32(high1 >= values1),
                                         v_reinterpret_as_u32(values2 >= low2) & v_reinterpret_as_u32(high2 >= values2)));
        }
        vx_cleanup();
        return x;
    }
};

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

static void inRange32s(const int* src1, size_t step1, const int* src2, size_t step2,
                       const int* src3, size_t step3, uchar* dst, size_t step, Size size)
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
    static InRangeFunc inRangeTab[] =
    {
        (InRangeFunc)GET_OPTIMIZED(inRange8u), (InRangeFunc)GET_OPTIMIZED(inRange8s), (InRangeFunc)GET_OPTIMIZED(inRange16u),
        (InRangeFunc)GET_OPTIMIZED(inRange16s), (InRangeFunc)GET_OPTIMIZED(inRange32s), (InRangeFunc)GET_OPTIMIZED(inRange32f),
        (InRangeFunc)inRange64f, 0
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
            CV_Error( CV_StsUnmatchedSizes,
                     "The lower boundary is neither an array of the same size and same type as src, nor a scalar");
        lbScalar = true;
    }

    if( (ukind == _InputArray::MATX && skind != _InputArray::MATX) ||
        ssize != usize || stype != utype )
    {
        if( !checkScalar(_upperb, stype, ukind, skind) )
            CV_Error( CV_StsUnmatchedSizes,
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
    String opts = format("%s-D cn=%d -D srcT=%s -D srcT1=%s -D dstT=%s -D kercn=%d -D depth=%d%s -D colsPerWI=%d",
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
            CV_Error( CV_StsUnmatchedSizes,
                     "The lower boundary is neither an array of the same size and same type as src, nor a scalar");
        lbScalar = true;
    }

    if( (ukind == _InputArray::MATX && skind != _InputArray::MATX) ||
        src.size != ub.size || src.type() != ub.type() )
    {
        if( !checkScalar(ub, src.type(), ukind, skind) )
            CV_Error( CV_StsUnmatchedSizes,
                     "The upper boundary is neither an array of the same size and same type as src, nor a scalar");
        ubScalar = true;
    }

    CV_Assert(lbScalar == ubScalar);

    int cn = src.channels(), depth = src.depth();

    size_t esz = src.elemSize();
    size_t blocksize0 = (size_t)(BLOCK_SIZE + esz-1)/esz;

    _dst.create(src.dims, src.size, CV_8UC1);
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


#ifndef OPENCV_EXCLUDE_C_API

/****************************************************************************************\
*                                Earlier API: cvAdd etc.                                 *
\****************************************************************************************/

CV_IMPL void
cvNot( const CvArr* srcarr, CvArr* dstarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.size == dst.size && src.type() == dst.type() );
    cv::bitwise_not( src, dst );
}


CV_IMPL void
cvAnd( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_and( src1, src2, dst, mask );
}


CV_IMPL void
cvOr( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_or( src1, src2, dst, mask );
}


CV_IMPL void
cvXor( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_xor( src1, src2, dst, mask );
}


CV_IMPL void
cvAndS( const CvArr* srcarr, CvScalar s, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src.size == dst.size && src.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_and( src, (const cv::Scalar&)s, dst, mask );
}


CV_IMPL void
cvOrS( const CvArr* srcarr, CvScalar s, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src.size == dst.size && src.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_or( src, (const cv::Scalar&)s, dst, mask );
}


CV_IMPL void
cvXorS( const CvArr* srcarr, CvScalar s, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src.size == dst.size && src.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_xor( src, (const cv::Scalar&)s, dst, mask );
}


CV_IMPL void cvAdd( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.channels() == dst.channels() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::add( src1, src2, dst, mask, dst.type() );
}


CV_IMPL void cvSub( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.channels() == dst.channels() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::subtract( src1, src2, dst, mask, dst.type() );
}


CV_IMPL void cvAddS( const CvArr* srcarr1, CvScalar value, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.channels() == dst.channels() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::add( src1, (const cv::Scalar&)value, dst, mask, dst.type() );
}


CV_IMPL void cvSubRS( const CvArr* srcarr1, CvScalar value, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.channels() == dst.channels() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::subtract( (const cv::Scalar&)value, src1, dst, mask, dst.type() );
}


CV_IMPL void cvMul( const CvArr* srcarr1, const CvArr* srcarr2,
                    CvArr* dstarr, double scale )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.channels() == dst.channels() );
    cv::multiply( src1, src2, dst, scale, dst.type() );
}


CV_IMPL void cvDiv( const CvArr* srcarr1, const CvArr* srcarr2,
                    CvArr* dstarr, double scale )
{
    cv::Mat src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src2.size == dst.size && src2.channels() == dst.channels() );

    if( srcarr1 )
        cv::divide( cv::cvarrToMat(srcarr1), src2, dst, scale, dst.type() );
    else
        cv::divide( scale, src2, dst, dst.type() );
}


CV_IMPL void
cvAddWeighted( const CvArr* srcarr1, double alpha,
               const CvArr* srcarr2, double beta,
               double gamma, CvArr* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.channels() == dst.channels() );
    cv::addWeighted( src1, alpha, src2, beta, gamma, dst, dst.type() );
}


CV_IMPL  void
cvAbsDiff( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::absdiff( src1, cv::cvarrToMat(srcarr2), dst );
}


CV_IMPL void
cvAbsDiffS( const CvArr* srcarr1, CvArr* dstarr, CvScalar scalar )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::absdiff( src1, (const cv::Scalar&)scalar, dst );
}


CV_IMPL void
cvInRange( const void* srcarr1, const void* srcarr2,
           const void* srcarr3, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && dst.type() == CV_8U );

    cv::inRange( src1, cv::cvarrToMat(srcarr2), cv::cvarrToMat(srcarr3), dst );
}


CV_IMPL void
cvInRangeS( const void* srcarr1, CvScalar lowerb, CvScalar upperb, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && dst.type() == CV_8U );

    cv::inRange( src1, (const cv::Scalar&)lowerb, (const cv::Scalar&)upperb, dst );
}


CV_IMPL void
cvCmp( const void* srcarr1, const void* srcarr2, void* dstarr, int cmp_op )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && dst.type() == CV_8U );

    cv::compare( src1, cv::cvarrToMat(srcarr2), dst, cmp_op );
}


CV_IMPL void
cvCmpS( const void* srcarr1, double value, void* dstarr, int cmp_op )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && dst.type() == CV_8U );

    cv::compare( src1, value, dst, cmp_op );
}


CV_IMPL void
cvMin( const void* srcarr1, const void* srcarr2, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::min( src1, cv::cvarrToMat(srcarr2), dst );
}


CV_IMPL void
cvMax( const void* srcarr1, const void* srcarr2, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::max( src1, cv::cvarrToMat(srcarr2), dst );
}


CV_IMPL void
cvMinS( const void* srcarr1, double value, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::min( src1, value, dst );
}


CV_IMPL void
cvMaxS( const void* srcarr1, double value, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::max( src1, value, dst );
}

#endif  // OPENCV_EXCLUDE_C_API
/* End of file. */
