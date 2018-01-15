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

#include "precomp.hpp"

#include "opencl_kernels_core.hpp"

#include "convert.hpp"

#include "opencv2/core/openvx/ovx_defs.hpp"

#ifdef __APPLE__
#undef CV_NEON
#define CV_NEON 0
#endif

#define CV_SPLIT_MERGE_MAX_BLOCK_SIZE(cn) ((INT_MAX/4)/cn) // HAL implementation accepts 'int' len, so INT_MAX doesn't work here

/****************************************************************************************\
*                                       split & merge                                    *
\****************************************************************************************/

typedef void (*SplitFunc)(const uchar* src, uchar** dst, int len, int cn);

static SplitFunc getSplitFunc(int depth)
{
    static SplitFunc splitTab[] =
    {
        (SplitFunc)GET_OPTIMIZED(cv::hal::split8u), (SplitFunc)GET_OPTIMIZED(cv::hal::split8u), (SplitFunc)GET_OPTIMIZED(cv::hal::split16u), (SplitFunc)GET_OPTIMIZED(cv::hal::split16u),
        (SplitFunc)GET_OPTIMIZED(cv::hal::split32s), (SplitFunc)GET_OPTIMIZED(cv::hal::split32s), (SplitFunc)GET_OPTIMIZED(cv::hal::split64s), 0
    };

    return splitTab[depth];
}

typedef void (*MergeFunc)(const uchar** src, uchar* dst, int len, int cn);

static MergeFunc getMergeFunc(int depth)
{
    static MergeFunc mergeTab[] =
    {
        (MergeFunc)GET_OPTIMIZED(cv::hal::merge8u), (MergeFunc)GET_OPTIMIZED(cv::hal::merge8u), (MergeFunc)GET_OPTIMIZED(cv::hal::merge16u), (MergeFunc)GET_OPTIMIZED(cv::hal::merge16u),
        (MergeFunc)GET_OPTIMIZED(cv::hal::merge32s), (MergeFunc)GET_OPTIMIZED(cv::hal::merge32s), (MergeFunc)GET_OPTIMIZED(cv::hal::merge64s), 0
    };

    return mergeTab[depth];
}

#ifdef HAVE_IPP

namespace cv {
static bool ipp_split(const Mat& src, Mat* mv, int channels)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP()

    if(channels != 3 && channels != 4)
        return false;

    if(src.dims <= 2)
    {
        IppiSize size       = ippiSize(src.size());
        void    *dstPtrs[4] = {NULL};
        size_t   dstStep    = mv[0].step;
        for(int i = 0; i < channels; i++)
        {
            dstPtrs[i] = mv[i].ptr();
            if(dstStep != mv[i].step)
                return false;
        }

        return CV_INSTRUMENT_FUN_IPP(llwiCopySplit, src.ptr(), (int)src.step, dstPtrs, (int)dstStep, size, (int)src.elemSize1(), channels, 0) >= 0;
    }
    else
    {
        const Mat *arrays[5] = {NULL};
        uchar     *ptrs[5]   = {NULL};
        arrays[0] = &src;

        for(int i = 1; i < channels; i++)
        {
            arrays[i] = &mv[i-1];
        }

        NAryMatIterator it(arrays, ptrs);
        IppiSize size = { (int)it.size, 1 };

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            if(CV_INSTRUMENT_FUN_IPP(llwiCopySplit, ptrs[0], 0, (void**)&ptrs[1], 0, size, (int)src.elemSize1(), channels, 0) < 0)
                return false;
        }
        return true;
    }
#else
    CV_UNUSED(src); CV_UNUSED(mv); CV_UNUSED(channels);
    return false;
#endif
}
}
#endif

void cv::split(const Mat& src, Mat* mv)
{
    CV_INSTRUMENT_REGION()

    int k, depth = src.depth(), cn = src.channels();
    if( cn == 1 )
    {
        src.copyTo(mv[0]);
        return;
    }

    for( k = 0; k < cn; k++ )
    {
        mv[k].create(src.dims, src.size, depth);
    }

    CV_IPP_RUN_FAST(ipp_split(src, mv, cn));

    SplitFunc func = getSplitFunc(depth);
    CV_Assert( func != 0 );

    size_t esz = src.elemSize(), esz1 = src.elemSize1();
    size_t blocksize0 = (BLOCK_SIZE + esz-1)/esz;
    AutoBuffer<uchar> _buf((cn+1)*(sizeof(Mat*) + sizeof(uchar*)) + 16);
    const Mat** arrays = (const Mat**)(uchar*)_buf;
    uchar** ptrs = (uchar**)alignPtr(arrays + cn + 1, 16);

    arrays[0] = &src;
    for( k = 0; k < cn; k++ )
    {
        arrays[k+1] = &mv[k];
    }

    NAryMatIterator it(arrays, ptrs, cn+1);
    size_t total = it.size;
    size_t blocksize = std::min((size_t)CV_SPLIT_MERGE_MAX_BLOCK_SIZE(cn), cn <= 4 ? total : std::min(total, blocksize0));

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( size_t j = 0; j < total; j += blocksize )
        {
            size_t bsz = std::min(total - j, blocksize);
            func( ptrs[0], &ptrs[1], (int)bsz, cn );

            if( j + blocksize < total )
            {
                ptrs[0] += bsz*esz;
                for( k = 0; k < cn; k++ )
                    ptrs[k+1] += bsz*esz1;
            }
        }
    }
}

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_split( InputArray _m, OutputArrayOfArrays _mv )
{
    int type = _m.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
            rowsPerWI = ocl::Device::getDefault().isIntel() ? 4 : 1;

    String dstargs, processelem, indexdecl;
    for (int i = 0; i < cn; ++i)
    {
        dstargs += format("DECLARE_DST_PARAM(%d)", i);
        indexdecl += format("DECLARE_INDEX(%d)", i);
        processelem += format("PROCESS_ELEM(%d)", i);
    }

    ocl::Kernel k("split", ocl::core::split_merge_oclsrc,
                  format("-D T=%s -D OP_SPLIT -D cn=%d -D DECLARE_DST_PARAMS=%s"
                         " -D PROCESS_ELEMS_N=%s -D DECLARE_INDEX_N=%s",
                         ocl::memopTypeToStr(depth), cn, dstargs.c_str(),
                         processelem.c_str(), indexdecl.c_str()));
    if (k.empty())
        return false;

    Size size = _m.size();
    _mv.create(cn, 1, depth);
    for (int i = 0; i < cn; ++i)
        _mv.create(size, depth, i);

    std::vector<UMat> dst;
    _mv.getUMatVector(dst);

    int argidx = k.set(0, ocl::KernelArg::ReadOnly(_m.getUMat()));
    for (int i = 0; i < cn; ++i)
        argidx = k.set(argidx, ocl::KernelArg::WriteOnlyNoSize(dst[i]));
    k.set(argidx, rowsPerWI);

    size_t globalsize[2] = { (size_t)size.width, ((size_t)size.height + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

}

#endif

void cv::split(InputArray _m, OutputArrayOfArrays _mv)
{
    CV_INSTRUMENT_REGION()

    CV_OCL_RUN(_m.dims() <= 2 && _mv.isUMatVector(),
               ocl_split(_m, _mv))

    Mat m = _m.getMat();
    if( m.empty() )
    {
        _mv.release();
        return;
    }

    CV_Assert( !_mv.fixedType() || _mv.empty() || _mv.type() == m.depth() );

    int depth = m.depth(), cn = m.channels();
    _mv.create(cn, 1, depth);
    for (int i = 0; i < cn; ++i)
        _mv.create(m.dims, m.size.p, depth, i);

    std::vector<Mat> dst;
    _mv.getMatVector(dst);

    split(m, &dst[0]);
}

#ifdef HAVE_IPP

namespace cv {
static bool ipp_merge(const Mat* mv, Mat& dst, int channels)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP()

    if(channels != 3 && channels != 4)
        return false;

    if(mv[0].dims <= 2)
    {
        IppiSize    size       = ippiSize(mv[0].size());
        const void *srcPtrs[4] = {NULL};
        size_t      srcStep    = mv[0].step;
        for(int i = 0; i < channels; i++)
        {
            srcPtrs[i] = mv[i].ptr();
            if(srcStep != mv[i].step)
                return false;
        }

        return CV_INSTRUMENT_FUN_IPP(llwiCopyMerge, srcPtrs, (int)srcStep, dst.ptr(), (int)dst.step, size, (int)mv[0].elemSize1(), channels, 0) >= 0;
    }
    else
    {
        const Mat *arrays[5] = {NULL};
        uchar     *ptrs[5]   = {NULL};
        arrays[0] = &dst;

        for(int i = 1; i < channels; i++)
        {
            arrays[i] = &mv[i-1];
        }

        NAryMatIterator it(arrays, ptrs);
        IppiSize size = { (int)it.size, 1 };

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            if(CV_INSTRUMENT_FUN_IPP(llwiCopyMerge, (const void**)&ptrs[1], 0, ptrs[0], 0, size, (int)mv[0].elemSize1(), channels, 0) < 0)
                return false;
        }
        return true;
    }
#else
    CV_UNUSED(dst); CV_UNUSED(mv); CV_UNUSED(channels);
    return false;
#endif
}
}
#endif

void cv::merge(const Mat* mv, size_t n, OutputArray _dst)
{
    CV_INSTRUMENT_REGION()

    CV_Assert( mv && n > 0 );

    int depth = mv[0].depth();
    bool allch1 = true;
    int k, cn = 0;
    size_t i;

    for( i = 0; i < n; i++ )
    {
        CV_Assert(mv[i].size == mv[0].size && mv[i].depth() == depth);
        allch1 = allch1 && mv[i].channels() == 1;
        cn += mv[i].channels();
    }

    CV_Assert( 0 < cn && cn <= CV_CN_MAX );
    _dst.create(mv[0].dims, mv[0].size, CV_MAKETYPE(depth, cn));
    Mat dst = _dst.getMat();

    if( n == 1 )
    {
        mv[0].copyTo(dst);
        return;
    }

    CV_IPP_RUN_FAST(ipp_merge(mv, dst, (int)n));

    if( !allch1 )
    {
        AutoBuffer<int> pairs(cn*2);
        int j, ni=0;

        for( i = 0, j = 0; i < n; i++, j += ni )
        {
            ni = mv[i].channels();
            for( k = 0; k < ni; k++ )
            {
                pairs[(j+k)*2] = j + k;
                pairs[(j+k)*2+1] = j + k;
            }
        }
        mixChannels( mv, n, &dst, 1, &pairs[0], cn );
        return;
    }

    MergeFunc func = getMergeFunc(depth);
    CV_Assert( func != 0 );

    size_t esz = dst.elemSize(), esz1 = dst.elemSize1();
    size_t blocksize0 = (int)((BLOCK_SIZE + esz-1)/esz);
    AutoBuffer<uchar> _buf((cn+1)*(sizeof(Mat*) + sizeof(uchar*)) + 16);
    const Mat** arrays = (const Mat**)(uchar*)_buf;
    uchar** ptrs = (uchar**)alignPtr(arrays + cn + 1, 16);

    arrays[0] = &dst;
    for( k = 0; k < cn; k++ )
        arrays[k+1] = &mv[k];

    NAryMatIterator it(arrays, ptrs, cn+1);
    size_t total = (int)it.size;
    size_t blocksize = std::min((size_t)CV_SPLIT_MERGE_MAX_BLOCK_SIZE(cn), cn <= 4 ? total : std::min(total, blocksize0));

    for( i = 0; i < it.nplanes; i++, ++it )
    {
        for( size_t j = 0; j < total; j += blocksize )
        {
            size_t bsz = std::min(total - j, blocksize);
            func( (const uchar**)&ptrs[1], ptrs[0], (int)bsz, cn );

            if( j + blocksize < total )
            {
                ptrs[0] += bsz*esz;
                for( int t = 0; t < cn; t++ )
                    ptrs[t+1] += bsz*esz1;
            }
        }
    }
}

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_merge( InputArrayOfArrays _mv, OutputArray _dst )
{
    std::vector<UMat> src, ksrc;
    _mv.getUMatVector(src);
    CV_Assert(!src.empty());

    int type = src[0].type(), depth = CV_MAT_DEPTH(type),
            rowsPerWI = ocl::Device::getDefault().isIntel() ? 4 : 1;
    Size size = src[0].size();

    for (size_t i = 0, srcsize = src.size(); i < srcsize; ++i)
    {
        int itype = src[i].type(), icn = CV_MAT_CN(itype), idepth = CV_MAT_DEPTH(itype),
                esz1 = CV_ELEM_SIZE1(idepth);
        if (src[i].dims > 2)
            return false;

        CV_Assert(size == src[i].size() && depth == idepth);

        for (int cn = 0; cn < icn; ++cn)
        {
            UMat tsrc = src[i];
            tsrc.offset += cn * esz1;
            ksrc.push_back(tsrc);
        }
    }
    int dcn = (int)ksrc.size();

    String srcargs, processelem, cndecl, indexdecl;
    for (int i = 0; i < dcn; ++i)
    {
        srcargs += format("DECLARE_SRC_PARAM(%d)", i);
        processelem += format("PROCESS_ELEM(%d)", i);
        indexdecl += format("DECLARE_INDEX(%d)", i);
        cndecl += format(" -D scn%d=%d", i, ksrc[i].channels());
    }

    ocl::Kernel k("merge", ocl::core::split_merge_oclsrc,
                  format("-D OP_MERGE -D cn=%d -D T=%s -D DECLARE_SRC_PARAMS_N=%s"
                         " -D DECLARE_INDEX_N=%s -D PROCESS_ELEMS_N=%s%s",
                         dcn, ocl::memopTypeToStr(depth), srcargs.c_str(),
                         indexdecl.c_str(), processelem.c_str(), cndecl.c_str()));
    if (k.empty())
        return false;

    _dst.create(size, CV_MAKE_TYPE(depth, dcn));
    UMat dst = _dst.getUMat();

    int argidx = 0;
    for (int i = 0; i < dcn; ++i)
        argidx = k.set(argidx, ocl::KernelArg::ReadOnlyNoSize(ksrc[i]));
    argidx = k.set(argidx, ocl::KernelArg::WriteOnly(dst));
    k.set(argidx, rowsPerWI);

    size_t globalsize[2] = { (size_t)dst.cols, ((size_t)dst.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

}

#endif

void cv::merge(InputArrayOfArrays _mv, OutputArray _dst)
{
    CV_INSTRUMENT_REGION()

    CV_OCL_RUN(_mv.isUMatVector() && _dst.isUMat(),
               ocl_merge(_mv, _dst))

    std::vector<Mat> mv;
    _mv.getMatVector(mv);
    merge(!mv.empty() ? &mv[0] : 0, mv.size(), _dst);
}

/****************************************************************************************\
*                       Generalized split/merge: mixing channels                         *
\****************************************************************************************/

namespace cv
{

template<typename T> static void
mixChannels_( const T** src, const int* sdelta,
              T** dst, const int* ddelta,
              int len, int npairs )
{
    int i, k;
    for( k = 0; k < npairs; k++ )
    {
        const T* s = src[k];
        T* d = dst[k];
        int ds = sdelta[k], dd = ddelta[k];
        if( s )
        {
            for( i = 0; i <= len - 2; i += 2, s += ds*2, d += dd*2 )
            {
                T t0 = s[0], t1 = s[ds];
                d[0] = t0; d[dd] = t1;
            }
            if( i < len )
                d[0] = s[0];
        }
        else
        {
            for( i = 0; i <= len - 2; i += 2, d += dd*2 )
                d[0] = d[dd] = 0;
            if( i < len )
                d[0] = 0;
        }
    }
}


static void mixChannels8u( const uchar** src, const int* sdelta,
                           uchar** dst, const int* ddelta,
                           int len, int npairs )
{
    mixChannels_(src, sdelta, dst, ddelta, len, npairs);
}

static void mixChannels16u( const ushort** src, const int* sdelta,
                            ushort** dst, const int* ddelta,
                            int len, int npairs )
{
    mixChannels_(src, sdelta, dst, ddelta, len, npairs);
}

static void mixChannels32s( const int** src, const int* sdelta,
                            int** dst, const int* ddelta,
                            int len, int npairs )
{
    mixChannels_(src, sdelta, dst, ddelta, len, npairs);
}

static void mixChannels64s( const int64** src, const int* sdelta,
                            int64** dst, const int* ddelta,
                            int len, int npairs )
{
    mixChannels_(src, sdelta, dst, ddelta, len, npairs);
}

typedef void (*MixChannelsFunc)( const uchar** src, const int* sdelta,
        uchar** dst, const int* ddelta, int len, int npairs );

static MixChannelsFunc getMixchFunc(int depth)
{
    static MixChannelsFunc mixchTab[] =
    {
        (MixChannelsFunc)mixChannels8u, (MixChannelsFunc)mixChannels8u, (MixChannelsFunc)mixChannels16u,
        (MixChannelsFunc)mixChannels16u, (MixChannelsFunc)mixChannels32s, (MixChannelsFunc)mixChannels32s,
        (MixChannelsFunc)mixChannels64s, 0
    };

    return mixchTab[depth];
}

}

void cv::mixChannels( const Mat* src, size_t nsrcs, Mat* dst, size_t ndsts, const int* fromTo, size_t npairs )
{
    CV_INSTRUMENT_REGION()

    if( npairs == 0 )
        return;
    CV_Assert( src && nsrcs > 0 && dst && ndsts > 0 && fromTo && npairs > 0 );

    size_t i, j, k, esz1 = dst[0].elemSize1();
    int depth = dst[0].depth();

    AutoBuffer<uchar> buf((nsrcs + ndsts + 1)*(sizeof(Mat*) + sizeof(uchar*)) + npairs*(sizeof(uchar*)*2 + sizeof(int)*6));
    const Mat** arrays = (const Mat**)(uchar*)buf;
    uchar** ptrs = (uchar**)(arrays + nsrcs + ndsts);
    const uchar** srcs = (const uchar**)(ptrs + nsrcs + ndsts + 1);
    uchar** dsts = (uchar**)(srcs + npairs);
    int* tab = (int*)(dsts + npairs);
    int *sdelta = (int*)(tab + npairs*4), *ddelta = sdelta + npairs;

    for( i = 0; i < nsrcs; i++ )
        arrays[i] = &src[i];
    for( i = 0; i < ndsts; i++ )
        arrays[i + nsrcs] = &dst[i];
    ptrs[nsrcs + ndsts] = 0;

    for( i = 0; i < npairs; i++ )
    {
        int i0 = fromTo[i*2], i1 = fromTo[i*2+1];
        if( i0 >= 0 )
        {
            for( j = 0; j < nsrcs; i0 -= src[j].channels(), j++ )
                if( i0 < src[j].channels() )
                    break;
            CV_Assert(j < nsrcs && src[j].depth() == depth);
            tab[i*4] = (int)j; tab[i*4+1] = (int)(i0*esz1);
            sdelta[i] = src[j].channels();
        }
        else
        {
            tab[i*4] = (int)(nsrcs + ndsts); tab[i*4+1] = 0;
            sdelta[i] = 0;
        }

        for( j = 0; j < ndsts; i1 -= dst[j].channels(), j++ )
            if( i1 < dst[j].channels() )
                break;
        CV_Assert(i1 >= 0 && j < ndsts && dst[j].depth() == depth);
        tab[i*4+2] = (int)(j + nsrcs); tab[i*4+3] = (int)(i1*esz1);
        ddelta[i] = dst[j].channels();
    }

    NAryMatIterator it(arrays, ptrs, (int)(nsrcs + ndsts));
    int total = (int)it.size, blocksize = std::min(total, (int)((BLOCK_SIZE + esz1-1)/esz1));
    MixChannelsFunc func = getMixchFunc(depth);

    for( i = 0; i < it.nplanes; i++, ++it )
    {
        for( k = 0; k < npairs; k++ )
        {
            srcs[k] = ptrs[tab[k*4]] + tab[k*4+1];
            dsts[k] = ptrs[tab[k*4+2]] + tab[k*4+3];
        }

        for( int t = 0; t < total; t += blocksize )
        {
            int bsz = std::min(total - t, blocksize);
            func( srcs, sdelta, dsts, ddelta, bsz, (int)npairs );

            if( t + blocksize < total )
                for( k = 0; k < npairs; k++ )
                {
                    srcs[k] += blocksize*sdelta[k]*esz1;
                    dsts[k] += blocksize*ddelta[k]*esz1;
                }
        }
    }
}

#ifdef HAVE_OPENCL

namespace cv {

static void getUMatIndex(const std::vector<UMat> & um, int cn, int & idx, int & cnidx)
{
    int totalChannels = 0;
    for (size_t i = 0, size = um.size(); i < size; ++i)
    {
        int ccn = um[i].channels();
        totalChannels += ccn;

        if (totalChannels == cn)
        {
            idx = (int)(i + 1);
            cnidx = 0;
            return;
        }
        else if (totalChannels > cn)
        {
            idx = (int)i;
            cnidx = i == 0 ? cn : (cn - totalChannels + ccn);
            return;
        }
    }

    idx = cnidx = -1;
}

static bool ocl_mixChannels(InputArrayOfArrays _src, InputOutputArrayOfArrays _dst,
                            const int* fromTo, size_t npairs)
{
    std::vector<UMat> src, dst;
    _src.getUMatVector(src);
    _dst.getUMatVector(dst);

    size_t nsrc = src.size(), ndst = dst.size();
    CV_Assert(nsrc > 0 && ndst > 0);

    Size size = src[0].size();
    int depth = src[0].depth(), esz = CV_ELEM_SIZE(depth),
            rowsPerWI = ocl::Device::getDefault().isIntel() ? 4 : 1;

    for (size_t i = 1, ssize = src.size(); i < ssize; ++i)
        CV_Assert(src[i].size() == size && src[i].depth() == depth);
    for (size_t i = 0, dsize = dst.size(); i < dsize; ++i)
        CV_Assert(dst[i].size() == size && dst[i].depth() == depth);

    String declsrc, decldst, declproc, declcn, indexdecl;
    std::vector<UMat> srcargs(npairs), dstargs(npairs);

    for (size_t i = 0; i < npairs; ++i)
    {
        int scn = fromTo[i<<1], dcn = fromTo[(i<<1) + 1];
        int src_idx, src_cnidx, dst_idx, dst_cnidx;

        getUMatIndex(src, scn, src_idx, src_cnidx);
        getUMatIndex(dst, dcn, dst_idx, dst_cnidx);

        CV_Assert(dst_idx >= 0 && src_idx >= 0);

        srcargs[i] = src[src_idx];
        srcargs[i].offset += src_cnidx * esz;

        dstargs[i] = dst[dst_idx];
        dstargs[i].offset += dst_cnidx * esz;

        declsrc += format("DECLARE_INPUT_MAT(%d)", i);
        decldst += format("DECLARE_OUTPUT_MAT(%d)", i);
        indexdecl += format("DECLARE_INDEX(%d)", i);
        declproc += format("PROCESS_ELEM(%d)", i);
        declcn += format(" -D scn%d=%d -D dcn%d=%d", i, src[src_idx].channels(), i, dst[dst_idx].channels());
    }

    ocl::Kernel k("mixChannels", ocl::core::mixchannels_oclsrc,
                  format("-D T=%s -D DECLARE_INPUT_MAT_N=%s -D DECLARE_OUTPUT_MAT_N=%s"
                         " -D PROCESS_ELEM_N=%s -D DECLARE_INDEX_N=%s%s",
                         ocl::memopTypeToStr(depth), declsrc.c_str(), decldst.c_str(),
                         declproc.c_str(), indexdecl.c_str(), declcn.c_str()));
    if (k.empty())
        return false;

    int argindex = 0;
    for (size_t i = 0; i < npairs; ++i)
        argindex = k.set(argindex, ocl::KernelArg::ReadOnlyNoSize(srcargs[i]));
    for (size_t i = 0; i < npairs; ++i)
        argindex = k.set(argindex, ocl::KernelArg::WriteOnlyNoSize(dstargs[i]));
    argindex = k.set(argindex, size.height);
    argindex = k.set(argindex, size.width);
    k.set(argindex, rowsPerWI);

    size_t globalsize[2] = { (size_t)size.width, ((size_t)size.height + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

}

#endif

void cv::mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
                 const int* fromTo, size_t npairs)
{
    CV_INSTRUMENT_REGION()

    if (npairs == 0 || fromTo == NULL)
        return;

    CV_OCL_RUN(dst.isUMatVector(),
               ocl_mixChannels(src, dst, fromTo, npairs))

    bool src_is_mat = src.kind() != _InputArray::STD_VECTOR_MAT &&
            src.kind() != _InputArray::STD_ARRAY_MAT &&
            src.kind() != _InputArray::STD_VECTOR_VECTOR &&
            src.kind() != _InputArray::STD_VECTOR_UMAT;
    bool dst_is_mat = dst.kind() != _InputArray::STD_VECTOR_MAT &&
            dst.kind() != _InputArray::STD_ARRAY_MAT &&
            dst.kind() != _InputArray::STD_VECTOR_VECTOR &&
            dst.kind() != _InputArray::STD_VECTOR_UMAT;
    int i;
    int nsrc = src_is_mat ? 1 : (int)src.total();
    int ndst = dst_is_mat ? 1 : (int)dst.total();

    CV_Assert(nsrc > 0 && ndst > 0);
    cv::AutoBuffer<Mat> _buf(nsrc + ndst);
    Mat* buf = _buf;
    for( i = 0; i < nsrc; i++ )
        buf[i] = src.getMat(src_is_mat ? -1 : i);
    for( i = 0; i < ndst; i++ )
        buf[nsrc + i] = dst.getMat(dst_is_mat ? -1 : i);
    mixChannels(&buf[0], nsrc, &buf[nsrc], ndst, fromTo, npairs);
}

void cv::mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
                     const std::vector<int>& fromTo)
{
    CV_INSTRUMENT_REGION()

    if (fromTo.empty())
        return;

    CV_OCL_RUN(dst.isUMatVector(),
               ocl_mixChannels(src, dst, &fromTo[0], fromTo.size()>>1))

    bool src_is_mat = src.kind() != _InputArray::STD_VECTOR_MAT &&
            src.kind() != _InputArray::STD_ARRAY_MAT &&
            src.kind() != _InputArray::STD_VECTOR_VECTOR &&
            src.kind() != _InputArray::STD_VECTOR_UMAT;
    bool dst_is_mat = dst.kind() != _InputArray::STD_VECTOR_MAT &&
            dst.kind() != _InputArray::STD_ARRAY_MAT &&
            dst.kind() != _InputArray::STD_VECTOR_VECTOR &&
            dst.kind() != _InputArray::STD_VECTOR_UMAT;
    int i;
    int nsrc = src_is_mat ? 1 : (int)src.total();
    int ndst = dst_is_mat ? 1 : (int)dst.total();

    CV_Assert(fromTo.size()%2 == 0 && nsrc > 0 && ndst > 0);
    cv::AutoBuffer<Mat> _buf(nsrc + ndst);
    Mat* buf = _buf;
    for( i = 0; i < nsrc; i++ )
        buf[i] = src.getMat(src_is_mat ? -1 : i);
    for( i = 0; i < ndst; i++ )
        buf[nsrc + i] = dst.getMat(dst_is_mat ? -1 : i);
    mixChannels(&buf[0], nsrc, &buf[nsrc], ndst, &fromTo[0], fromTo.size()/2);
}

#ifdef HAVE_IPP

namespace cv
{
static bool ipp_extractChannel(const Mat &src, Mat &dst, int channel)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP()

    int srcChannels = src.channels();
    int dstChannels = dst.channels();

    if(src.dims != dst.dims)
        return false;

    if(src.dims <= 2)
    {
        IppiSize size = ippiSize(src.size());

        return CV_INSTRUMENT_FUN_IPP(llwiCopyChannel, src.ptr(), (int)src.step, srcChannels, channel, dst.ptr(), (int)dst.step, dstChannels, 0, size, (int)src.elemSize1()) >= 0;
    }
    else
    {
        const Mat      *arrays[] = {&dst, NULL};
        uchar          *ptrs[2]  = {NULL};
        NAryMatIterator it(arrays, ptrs);

        IppiSize size = {(int)it.size, 1};

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            if(CV_INSTRUMENT_FUN_IPP(llwiCopyChannel, ptrs[0], 0, srcChannels, channel, ptrs[1], 0, dstChannels, 0, size, (int)src.elemSize1()) < 0)
                return false;
        }
        return true;
    }
#else
    CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(channel);
    return false;
#endif
}

static bool ipp_insertChannel(const Mat &src, Mat &dst, int channel)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP()

    int srcChannels = src.channels();
    int dstChannels = dst.channels();

    if(src.dims != dst.dims)
        return false;

    if(src.dims <= 2)
    {
        IppiSize size = ippiSize(src.size());

        return CV_INSTRUMENT_FUN_IPP(llwiCopyChannel, src.ptr(), (int)src.step, srcChannels, 0, dst.ptr(), (int)dst.step, dstChannels, channel, size, (int)src.elemSize1()) >= 0;
    }
    else
    {
        const Mat      *arrays[] = {&dst, NULL};
        uchar          *ptrs[2]  = {NULL};
        NAryMatIterator it(arrays, ptrs);

        IppiSize size = {(int)it.size, 1};

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            if(CV_INSTRUMENT_FUN_IPP(llwiCopyChannel, ptrs[0], 0, srcChannels, 0, ptrs[1], 0, dstChannels, channel, size, (int)src.elemSize1()) < 0)
                return false;
        }
        return true;
    }
#else
    CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(channel);
    return false;
#endif
}
}
#endif

void cv::extractChannel(InputArray _src, OutputArray _dst, int coi)
{
    CV_INSTRUMENT_REGION()

    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert( 0 <= coi && coi < cn );
    int ch[] = { coi, 0 };

#ifdef HAVE_OPENCL
    if (ocl::isOpenCLActivated() && _src.dims() <= 2 && _dst.isUMat())
    {
        UMat src = _src.getUMat();
        _dst.create(src.dims, &src.size[0], depth);
        UMat dst = _dst.getUMat();
        mixChannels(std::vector<UMat>(1, src), std::vector<UMat>(1, dst), ch, 1);
        return;
    }
#endif

    Mat src = _src.getMat();
    _dst.create(src.dims, &src.size[0], depth);
    Mat dst = _dst.getMat();

    CV_IPP_RUN_FAST(ipp_extractChannel(src, dst, coi))

    mixChannels(&src, 1, &dst, 1, ch, 1);
}

void cv::insertChannel(InputArray _src, InputOutputArray _dst, int coi)
{
    CV_INSTRUMENT_REGION()

    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), scn = CV_MAT_CN(stype);
    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype), dcn = CV_MAT_CN(dtype);
    CV_Assert( _src.sameSize(_dst) && sdepth == ddepth );
    CV_Assert( 0 <= coi && coi < dcn && scn == 1 );

    int ch[] = { 0, coi };
#ifdef HAVE_OPENCL
    if (ocl::isOpenCLActivated() && _src.dims() <= 2 && _dst.isUMat())
    {
        UMat src = _src.getUMat(), dst = _dst.getUMat();
        mixChannels(std::vector<UMat>(1, src), std::vector<UMat>(1, dst), ch, 1);
        return;
    }
#endif

    Mat src = _src.getMat(), dst = _dst.getMat();

    CV_IPP_RUN_FAST(ipp_insertChannel(src, dst, coi))

    mixChannels(&src, 1, &dst, 1, ch, 1);
}

/****************************************************************************************\
*                                convertScale[Abs]                                       *
\****************************************************************************************/

namespace cv
{

template<typename T, typename DT, typename WT>
struct cvtScaleAbs_SIMD
{
    int operator () (const T *, DT *, int, WT, WT) const
    {
        return 0;
    }
};

#if CV_SIMD128

static inline void v_load_expand_from_u8_f32(const uchar* src, const v_float32x4 &v_scale, const v_float32x4 &v_shift, v_float32x4 &a, v_float32x4 &b)
{
    v_uint32x4 v_src0, v_src1;
    v_expand(v_load_expand(src), v_src0, v_src1);

    a = v_shift + v_scale * v_cvt_f32(v_reinterpret_as_s32(v_src0));
    b = v_shift + v_scale * v_cvt_f32(v_reinterpret_as_s32(v_src1));
}

static inline void v_load_expand_from_s8_f32(const schar* src, const v_float32x4 &v_scale, const v_float32x4 &v_shift, v_float32x4 &a, v_float32x4 &b)
{
    v_int32x4 v_src0, v_src1;
    v_expand(v_load_expand(src), v_src0, v_src1);

    a = v_shift + v_scale * v_cvt_f32(v_src0);
    b = v_shift + v_scale * v_cvt_f32(v_src1);
}

static inline void v_load_expand_from_u16_f32(const ushort* src, const v_float32x4 &v_scale, const v_float32x4 &v_shift, v_float32x4 &a, v_float32x4 &b)
{
    v_uint32x4 v_src0, v_src1;
    v_expand(v_load(src), v_src0, v_src1);

    a = v_shift + v_scale * v_cvt_f32(v_reinterpret_as_s32(v_src0));
    b = v_shift + v_scale * v_cvt_f32(v_reinterpret_as_s32(v_src1));
}

static inline void v_load_expand_from_s16_f32(const short* src, const v_float32x4 &v_scale, const v_float32x4 &v_shift, v_float32x4 &a, v_float32x4 &b)
{
    v_int32x4 v_src0, v_src1;
    v_expand(v_load(src), v_src0, v_src1);

    a = v_shift + v_scale * v_cvt_f32(v_src0);
    b = v_shift + v_scale * v_cvt_f32(v_src1);
}

static inline void v_load_expand_from_s32_f32(const int* src, const v_float32x4 &v_scale, const v_float32x4 &v_shift, v_float32x4 &a, v_float32x4 &b)
{
    a = v_shift + v_scale * v_cvt_f32(v_load(src));
    b = v_shift + v_scale * v_cvt_f32(v_load(src + v_int32x4::nlanes));
}

template <>
struct cvtScaleAbs_SIMD<uchar, uchar, float>
{
    int operator () (const uchar * src, uchar * dst, int width,
        float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift);
            v_float32x4 v_scale = v_setall_f32(scale);
            const int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_dst_0, v_dst_1, v_dst_2, v_dst_3;
                v_load_expand_from_u8_f32(src + x, v_scale, v_shift, v_dst_0, v_dst_1);
                v_load_expand_from_u8_f32(src + x + cWidth, v_scale, v_shift, v_dst_2, v_dst_3);
                v_dst_0 = v_abs(v_dst_0);
                v_dst_1 = v_abs(v_dst_1);
                v_dst_2 = v_abs(v_dst_2);
                v_dst_3 = v_abs(v_dst_3);

                v_int16x8 v_dsti_0 = v_pack(v_round(v_dst_0), v_round(v_dst_1));
                v_int16x8 v_dsti_1 = v_pack(v_round(v_dst_2), v_round(v_dst_3));
                v_store(dst + x, v_pack_u(v_dsti_0, v_dsti_1));
            }
        }
        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<schar, uchar, float>
{
    int operator () (const schar * src, uchar * dst, int width,
        float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift);
            v_float32x4 v_scale = v_setall_f32(scale);
            const int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth*2; x += cWidth*2)
            {
                v_float32x4 v_dst_0, v_dst_1, v_dst_2, v_dst_3;
                v_load_expand_from_s8_f32(src + x, v_scale, v_shift, v_dst_0, v_dst_1);
                v_load_expand_from_s8_f32(src + x + cWidth, v_scale, v_shift, v_dst_2, v_dst_3);
                v_dst_0 = v_abs(v_dst_0);
                v_dst_1 = v_abs(v_dst_1);
                v_dst_2 = v_abs(v_dst_2);
                v_dst_3 = v_abs(v_dst_3);

                v_uint16x8 v_dsti_0 = v_pack_u(v_round(v_dst_0), v_round(v_dst_1));
                v_uint16x8 v_dsti_1 = v_pack_u(v_round(v_dst_2), v_round(v_dst_3));
                v_store(dst + x, v_pack(v_dsti_0, v_dsti_1));
            }
        }
        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<ushort, uchar, float>
{
    int operator () (const ushort * src, uchar * dst, int width,
        float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift);
            v_float32x4 v_scale = v_setall_f32(scale);
            const int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_dst0, v_dst1;
                v_load_expand_from_u16_f32(src + x, v_scale, v_shift, v_dst0, v_dst1);
                v_dst0 = v_abs(v_dst0);
                v_dst1 = v_abs(v_dst1);

                v_int16x8 v_dst = v_pack(v_round(v_dst0), v_round(v_dst1));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<short, uchar, float>
{
    int operator () (const short * src, uchar * dst, int width,
        float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift);
            v_float32x4 v_scale = v_setall_f32(scale);
            const int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_dst0, v_dst1;
                v_load_expand_from_s16_f32(src + x, v_scale, v_shift, v_dst0, v_dst1);
                v_dst0 = v_abs(v_dst0);
                v_dst1 = v_abs(v_dst1);

                v_int16x8 v_dst = v_pack(v_round(v_dst0), v_round(v_dst1));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<int, uchar, float>
{
    int operator () (const int * src, uchar * dst, int width,
        float scale, float shift) const
    {
        int x = 0;
        v_float32x4 v_shift = v_setall_f32(shift);
        v_float32x4 v_scale = v_setall_f32(scale);
        const int cWidth = v_int32x4::nlanes;
        for (; x <= width - cWidth * 2; x += cWidth * 2)
        {
            v_float32x4 v_dst_0 = v_cvt_f32(v_load(src + x)) * v_scale;
            v_dst_0 = v_abs(v_dst_0 + v_shift);

            v_float32x4 v_dst_1 = v_cvt_f32(v_load(src + x + cWidth)) * v_scale;
            v_dst_1 = v_abs(v_dst_1 + v_shift);

            v_int16x8 v_dst = v_pack(v_round(v_dst_0), v_round(v_dst_1));
            v_pack_u_store(dst + x, v_dst);
        }

        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<float, uchar, float>
{
    int operator () (const float * src, uchar * dst, int width,
        float scale, float shift) const
    {
        int x = 0;
        v_float32x4 v_shift = v_setall_f32(shift);
        v_float32x4 v_scale = v_setall_f32(scale);
        int cWidth = v_float32x4::nlanes;
        for (; x <= width - cWidth * 2; x += cWidth * 2)
        {
            v_float32x4 v_dst_0 = v_load(src + x) * v_scale;
            v_dst_0 = v_abs(v_dst_0 + v_shift);

            v_float32x4 v_dst_1 = v_load(src + x + cWidth) * v_scale;
            v_dst_1 = v_abs(v_dst_1 + v_shift);

            v_int16x8 v_dst = v_pack(v_round(v_dst_0), v_round(v_dst_1));
            v_pack_u_store(dst + x, v_dst);
        }
        return x;
    }
};

#if CV_SIMD128_64F
template <>
struct cvtScaleAbs_SIMD<double, uchar, float>
{
    int operator () (const double * src, uchar * dst, int width,
        float scale, float shift) const
    {
        int x = 0;

        if (hasSIMD128())
        {
            v_float32x4 v_scale = v_setall_f32(scale);
            v_float32x4 v_shift = v_setall_f32(shift);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_float32x4 v_src1, v_src2, v_dummy;
                v_recombine(v_cvt_f32(v_load(src + x)), v_cvt_f32(v_load(src + x + cWidth)), v_src1, v_dummy);
                v_recombine(v_cvt_f32(v_load(src + x + cWidth * 2)), v_cvt_f32(v_load(src + x + cWidth * 3)), v_src2, v_dummy);

                v_float32x4 v_dst1 = v_abs((v_src1 * v_scale) + v_shift);
                v_float32x4 v_dst2 = v_abs((v_src2 * v_scale) + v_shift);

                v_int16x8 v_dst_i = v_pack(v_round(v_dst1), v_round(v_dst2));
                v_pack_u_store(dst + x, v_dst_i);
            }
        }

        return x;
    }
};
#endif // CV_SIMD128_64F

#endif

template<typename T, typename DT, typename WT> static void
cvtScaleAbs_( const T* src, size_t sstep,
              DT* dst, size_t dstep, Size size,
              WT scale, WT shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);
    cvtScaleAbs_SIMD<T, DT, WT> vop;

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = vop(src, dst, size.width, scale, shift);

        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = saturate_cast<DT>(std::abs(src[x]*scale + shift));
            t1 = saturate_cast<DT>(std::abs(src[x+1]*scale + shift));
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<DT>(std::abs(src[x+2]*scale + shift));
            t1 = saturate_cast<DT>(std::abs(src[x+3]*scale + shift));
            dst[x+2] = t0; dst[x+3] = t1;
        }
        #endif
        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<DT>(std::abs(src[x]*scale + shift));
    }
}

template <typename T, typename DT, typename WT>
struct cvtScale_SIMD
{
    int operator () (const T *, DT *, int, WT, WT) const
    {
        return 0;
    }
};

#if CV_SIMD128

// from uchar

template <>
struct cvtScale_SIMD<uchar, uchar, float>
{
    int operator () (const uchar * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, schar, float>
{
    int operator () (const uchar * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store_low(dst + x, v_pack(v_dst, v_dst));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, ushort, float>
{
    int operator () (const uchar * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
#if CV_TRY_SSE4_1
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::cvtScale_SIMD_u8u16f32_SSE41(src, dst, width, scale, shift);
#endif
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_uint16x8 v_dst = v_pack_u(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, short, float>
{
    int operator () (const uchar * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, int, float>
{
    int operator () (const uchar * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_store(dst + x, v_round(v_src1));
                v_store(dst + x + cWidth, v_round(v_src2));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, float, float>
{
    int operator () (const uchar * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_store(dst + x, v_src1);
                v_store(dst + x + cWidth, v_src2);
            }
        }
        return x;
    }
};

// from schar

template <>
struct cvtScale_SIMD<schar, uchar, float>
{
    int operator () (const schar * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, schar, float>
{
    int operator () (const schar * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store_low(dst + x, v_pack(v_dst, v_dst));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, ushort, float>
{
    int operator () (const schar * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
#if CV_TRY_SSE4_1
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::cvtScale_SIMD_s8u16f32_SSE41(src, dst, width, scale, shift);
#endif
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_uint16x8 v_dst = v_pack_u(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, short, float>
{
    int operator () (const schar * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, int, float>
{
    int operator () (const schar * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_store(dst + x, v_round(v_src1));
                v_store(dst + x + cWidth, v_round(v_src2));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, float, float>
{
    int operator () (const schar * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_store(dst + x, v_src1);
                v_store(dst + x + cWidth, v_src2);
            }
        }
        return x;
    }
};

// from ushort

template <>
struct cvtScale_SIMD<ushort, uchar, float>
{
    int operator () (const ushort * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, schar, float>
{
    int operator () (const ushort * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store_low(dst + x, v_pack(v_dst, v_dst));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, ushort, float>
{
    int operator () (const ushort * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
#if CV_TRY_SSE4_1
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::cvtScale_SIMD_u16u16f32_SSE41(src, dst, width, scale, shift);
#endif
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_uint16x8 v_dst = v_pack_u(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, short, float>
{
    int operator () (const ushort * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, int, float>
{
    int operator () (const ushort * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_store(dst + x, v_round(v_src1));
                v_store(dst + x + cWidth, v_round(v_src2));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, float, float>
{
    int operator () (const ushort * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_store(dst + x, v_src1);
                v_store(dst + x + cWidth, v_src2);
            }
        }
        return x;
    }
};

// from short

template <>
struct cvtScale_SIMD<short, uchar, float>
{
    int operator () (const short * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<short, schar, float>
{
    int operator () (const short * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store_low(dst + x, v_pack(v_dst, v_dst));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<short, ushort, float>
{
    int operator () (const short * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
#if CV_TRY_SSE4_1
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::cvtScale_SIMD_s16u16f32_SSE41(src, dst, width, scale, shift);
#endif
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_uint16x8 v_dst = v_pack_u(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<short, short, float>
{
    int operator () (const short * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<short, float, float>
{
    int operator () (const short * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_store(dst + x, v_src1);
                v_store(dst + x + cWidth, v_src2);
            }
        }
        return x;
    }
};

// from int

template <>
struct cvtScale_SIMD<int, uchar, float>
{
    int operator () (const int * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s32_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<int, schar, float>
{
    int operator () (const int * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s32_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store_low(dst + x, v_pack(v_dst, v_dst));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<int, ushort, float>
{
    int operator () (const int * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
#if CV_TRY_SSE4_1
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::cvtScale_SIMD_s32u16f32_SSE41(src, dst, width, scale, shift);
#endif
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s32_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_uint16x8 v_dst = v_pack_u(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<int, short, float>
{
    int operator () (const int * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s32_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

#if CV_SIMD128_64F
template <>
struct cvtScale_SIMD<int, int, double>
{
    int operator () (const int * src, int * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                double v_srcbuf[] = { (double)src[x], (double)src[x+1], (double)src[x+2], (double)src[x+3] };
                v_float64x2 v_src1 = v_shift + v_scale * v_load(v_srcbuf);
                v_float64x2 v_src2 = v_shift + v_scale * v_load(v_srcbuf + 2);
                v_store(dst + x, v_combine_low(v_round(v_src1), v_round(v_src2)));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<int, float, double>
{
    int operator () (const int * src, float * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                double v_srcbuf[] = { (double)src[x], (double)src[x+1], (double)src[x+2], (double)src[x+3] };
                v_float64x2 v_src1 = v_shift + v_scale * v_load(v_srcbuf);
                v_float64x2 v_src2 = v_shift + v_scale * v_load(v_srcbuf + 2);
                v_store(dst + x, v_combine_low(v_cvt_f32(v_src1), v_cvt_f32(v_src2)));
            }
        }
        return x;
    }
};
#endif //CV_SIMD128_64F

// from float

template <>
struct cvtScale_SIMD<float, uchar, float>
{
    int operator () (const float * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_dst1 = v_shift + v_scale * v_load(src + x);
                v_float32x4 v_dst2 = v_shift + v_scale * v_load(src + x + cWidth);

                v_int16x8 v_dst = v_pack(v_round(v_dst1), v_round(v_dst2));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<float, schar, float>
{
    int operator () (const float * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_dst1 = v_shift + v_scale * v_load(src + x);
                v_float32x4 v_dst2 = v_shift + v_scale * v_load(src + x + cWidth);

                v_int16x8 v_dst = v_pack(v_round(v_dst1), v_round(v_dst2));
                v_store_low(dst + x, v_pack(v_dst, v_dst));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<float, ushort, float>
{
    int operator () (const float * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
#if CV_TRY_SSE4_1
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::cvtScale_SIMD_f32u16f32_SSE41(src, dst, width, scale, shift);
#endif
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_dst1 = v_shift + v_scale * v_load(src + x);
                v_float32x4 v_dst2 = v_shift + v_scale * v_load(src + x + cWidth);

                v_uint16x8 v_dst = v_pack_u(v_round(v_dst1), v_round(v_dst2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<float, short, float>
{
    int operator () (const float * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_dst1 = v_shift + v_scale * v_load(src + x);
                v_float32x4 v_dst2 = v_shift + v_scale * v_load(src + x + cWidth);

                v_int16x8 v_dst = v_pack(v_round(v_dst1), v_round(v_dst2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<float, int, float>
{
    int operator () (const float * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth; x += cWidth)
                v_store(dst + x, v_round(v_load(src + x) * v_scale + v_shift));
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<float, float, float>
{
    int operator () (const float * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth; x += cWidth)
                v_store(dst + x, v_load(src + x) * v_scale + v_shift);
        }
        return x;
    }
};

#if CV_SIMD128_64F

static inline void v_load_scale_shift(const double* src, const v_float64x2& v_scale, const v_float64x2 &v_shift, v_float32x4& v_dst1, v_float32x4 &v_dst2)
{
    int cWidth = v_float64x2::nlanes;
    v_float64x2 v_src1 = v_shift + v_scale * v_load(src);
    v_float64x2 v_src2 = v_shift + v_scale * v_load(src + cWidth);
    v_float64x2 v_src3 = v_shift + v_scale * v_load(src + cWidth * 2);
    v_float64x2 v_src4 = v_shift + v_scale * v_load(src + cWidth * 3);
    v_dst1 = v_combine_low(v_cvt_f32(v_src1), v_cvt_f32(v_src2));
    v_dst2 = v_combine_low(v_cvt_f32(v_src3), v_cvt_f32(v_src4));
}

static inline void v_store_scale_shift_s32_to_f64(double *dst, const v_float64x2 &v_scale, const v_float64x2 &v_shift, const v_int32x4 &v1, const v_int32x4 &v2)
{
    v_float64x2 v_dst1 = v_shift + v_scale * v_cvt_f64(v1);
    v_float64x2 v_dst2 = v_shift + v_scale * v_cvt_f64_high(v1);
    v_float64x2 v_dst3 = v_shift + v_scale * v_cvt_f64(v2);
    v_float64x2 v_dst4 = v_shift + v_scale * v_cvt_f64_high(v2);

    v_store(dst, v_dst1);
    v_store(dst + v_float64x2::nlanes, v_dst2);
    v_store(dst + v_float64x2::nlanes * 2, v_dst3);
    v_store(dst + v_float64x2::nlanes * 3, v_dst4);
}

static inline void v_store_scale_shift_f32_to_f64(double *dst, const v_float64x2 &v_scale, const v_float64x2 &v_shift, const v_float32x4 &v1, const v_float32x4 &v2)
{
    v_float64x2 v_dst1 = v_shift + v_scale * v_cvt_f64(v1);
    v_float64x2 v_dst2 = v_shift + v_scale * v_cvt_f64_high(v1);
    v_float64x2 v_dst3 = v_shift + v_scale * v_cvt_f64(v2);
    v_float64x2 v_dst4 = v_shift + v_scale * v_cvt_f64_high(v2);

    v_store(dst, v_dst1);
    v_store(dst + v_float64x2::nlanes, v_dst2);
    v_store(dst + v_float64x2::nlanes * 2, v_dst3);
    v_store(dst + v_float64x2::nlanes * 3, v_dst4);
}

// from double

template <>
struct cvtScale_SIMD<double, uchar, float>
{
    int operator () (const double * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64((double)shift), v_scale = v_setall_f64((double)scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_float32x4 v_dst1, v_dst2;
                v_load_scale_shift(src + x, v_scale, v_shift, v_dst1, v_dst2);
                v_pack_u_store(dst + x, v_pack(v_round(v_dst1), v_round(v_dst2)));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<double, schar, float>
{
    int operator () (const double * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64((double)shift), v_scale = v_setall_f64((double)scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_float32x4 v_dst1, v_dst2;
                v_load_scale_shift(src + x, v_scale, v_shift, v_dst1, v_dst2);
                v_int16x8 v_dst = v_pack(v_round(v_dst1), v_round(v_dst2));
                v_pack_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<double, ushort, float>
{
    int operator () (const double * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
#if CV_TRY_SSE4_1
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::cvtScale_SIMD_f64u16f32_SSE41(src, dst, width, scale, shift);
#endif
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64((double)shift), v_scale = v_setall_f64((double)scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_dst1, v_dst2;
                v_load_scale_shift(src + x, v_scale, v_shift, v_dst1, v_dst2);
                v_uint16x8 v_dst = v_pack_u(v_round(v_dst1), v_round(v_dst2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<double, short, float>
{
    int operator () (const double * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64((double)shift), v_scale = v_setall_f64((double)scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_dst1, v_dst2;
                v_load_scale_shift(src + x, v_scale, v_shift, v_dst1, v_dst2);
                v_int16x8 v_dst = v_pack(v_round(v_dst1), v_round(v_dst2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<double, int, double>
{
    int operator () (const double * src, int * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float64x2 v_src1 = v_shift + v_scale * v_load(src + x);
                v_float64x2 v_src2 = v_shift + v_scale * v_load(src + x + cWidth);

                v_store(dst + x, v_combine_low(v_round(v_src1), v_round(v_src2)));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<double, float, double>
{
    int operator () (const double * src, float * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float64x2 v_src1 = v_shift + v_scale * v_load(src + x);
                v_float64x2 v_src2 = v_shift + v_scale * v_load(src + x + cWidth);
                v_float32x4 v_dst1 = v_cvt_f32(v_src1);
                v_float32x4 v_dst2 = v_cvt_f32(v_src2);

                v_store(dst + x, v_combine_low(v_dst1, v_dst2));
            }
        }
        return x;
    }
};

// to double

template <>
struct cvtScale_SIMD<uchar, double, double>
{
    int operator () (const uchar * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_uint32x4 v_src1, v_src2;
                v_expand(v_load_expand(src + x), v_src1, v_src2);
                v_store_scale_shift_s32_to_f64(dst + x, v_scale, v_shift
                    , v_reinterpret_as_s32(v_src1), v_reinterpret_as_s32(v_src2));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, double, double>
{
    int operator () (const schar * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_int32x4 v_src1, v_src2;
                v_expand(v_load_expand(src + x), v_src1, v_src2);
                v_store_scale_shift_s32_to_f64(dst + x, v_scale, v_shift, v_src1, v_src2);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, double, double>
{
    int operator () (const ushort * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_uint32x4 v_src1, v_src2;
                v_expand(v_load(src + x), v_src1, v_src2);
                v_store_scale_shift_s32_to_f64(dst + x, v_scale, v_shift
                    , v_reinterpret_as_s32(v_src1), v_reinterpret_as_s32(v_src2));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<short, double, double>
{
    int operator () (const short * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_int32x4 v_src1, v_src2;
                v_expand(v_load(src + x), v_src1, v_src2);
                v_store_scale_shift_s32_to_f64(dst + x, v_scale, v_shift, v_src1, v_src2);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<int, double, double>
{
    int operator () (const int * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int32x4 v_src1 = v_load(src + x);
                v_int32x4 v_src2 = v_load(src + x + cWidth);
                v_store_scale_shift_s32_to_f64(dst + x, v_scale, v_shift, v_src1, v_src2);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<float, double, double>
{
    int operator () (const float * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src1 = v_load(src + x);
                v_float32x4 v_src2 = v_load(src + x + cWidth);
                v_store_scale_shift_f32_to_f64(dst + x, v_scale, v_shift, v_src1, v_src2);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<double, double, double>
{
    int operator () (const double * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float64x2 v_src1 = v_shift + v_scale * v_load(src + x);
                v_float64x2 v_src2 = v_shift + v_scale * v_load(src + x + cWidth);
                v_store(dst + x, v_src1);
                v_store(dst + x + cWidth, v_src2);
            }
        }
        return x;
    }
};
#endif
#endif

template<typename T, typename DT, typename WT> static void
cvtScale_( const T* src, size_t sstep,
           DT* dst, size_t dstep, Size size,
           WT scale, WT shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    cvtScale_SIMD<T, DT, WT> vop;

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = vop(src, dst, size.width, scale, shift);

        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = saturate_cast<DT>(src[x]*scale + shift);
            t1 = saturate_cast<DT>(src[x+1]*scale + shift);
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<DT>(src[x+2]*scale + shift);
            t1 = saturate_cast<DT>(src[x+3]*scale + shift);
            dst[x+2] = t0; dst[x+3] = t1;
        }
        #endif

        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<DT>(src[x]*scale + shift);
    }
}

template<> void
cvtScale_<short, int, float>( const short* src, size_t sstep,
           int* dst, size_t dstep, Size size,
           float scale, float shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;
        #if CV_TRY_AVX2
        if (CV_CPU_HAS_SUPPORT_AVX2)
        {
            opt_AVX2::cvtScale_s16s32f32Line_AVX2(src, dst, scale, shift, size.width);
            continue;
        }
        #endif
        #if CV_SIMD128
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift);
            v_float32x4 v_scale = v_setall_f32(scale);
            int cWidth = v_int32x4::nlanes;
            for (; x <= size.width - cWidth * 2; x += cWidth * 2)
            {
                v_int16x8 v_src = v_load(src + x);
                v_int32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_float32x4 v_tmp1 = v_cvt_f32(v_src1);
                v_float32x4 v_tmp2 = v_cvt_f32(v_src2);

                v_tmp1 = v_tmp1 * v_scale + v_shift;
                v_tmp2 = v_tmp2 * v_scale + v_shift;

                v_store(dst + x, v_round(v_tmp1));
                v_store(dst + x + cWidth, v_round(v_tmp2));
            }
        }
        #endif

        for(; x < size.width; x++ )
            dst[x] = saturate_cast<int>(src[x]*scale + shift);
    }
}

template <typename T, typename DT>
struct Cvt_SIMD
{
    int operator() (const T *, DT *, int) const
    {
        return 0;
    }
};

#if CV_SIMD128
// from uchar

template <>
struct Cvt_SIMD<uchar, schar>
{
    int operator() (const uchar * src, schar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_int16x8 v_src = v_reinterpret_as_s16(v_load_expand(src + x));
                v_store_low(dst + x, v_pack(v_src, v_src));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<uchar, ushort>
{
    int operator() (const uchar * src, ushort * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
                v_store(dst + x, v_load_expand(src + x));
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<uchar, short>
{
    int operator() (const uchar * src, short * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_int16x8 v_src = v_reinterpret_as_s16(v_load_expand(src + x));
                v_store(dst + x, v_src);
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<uchar, int>
{
    int operator() (const uchar * src, int * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_uint16x8 v_src = v_load_expand(src + x);
                v_uint32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_reinterpret_as_s32(v_src1));
                v_store(dst + x + cWidth, v_reinterpret_as_s32(v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<uchar, float>
{
    int operator() (const uchar * src, float * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_uint16x8 v_src = v_load_expand(src + x);
                v_uint32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_cvt_f32(v_reinterpret_as_s32(v_src1)));
                v_store(dst + x + cWidth, v_cvt_f32(v_reinterpret_as_s32(v_src2)));
            }
        }
        return x;
    }
};

// from schar

template <>
struct Cvt_SIMD<schar, uchar>
{
    int operator() (const schar * src, uchar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
                v_pack_u_store(dst + x, v_load_expand(src + x));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<schar, short>
{
    int operator() (const schar * src, short * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
                v_store(dst + x, v_load_expand(src + x));
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<schar, ushort>
{
    int operator() (const schar * src, ushort * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_int16x8 v_src = v_load_expand(src + x);
                v_int32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_pack_u(v_src1, v_src2));
            }
        }
        return x;
    }
};


template <>
struct Cvt_SIMD<schar, int>
{
    int operator() (const schar * src, int * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int16x8 v_src = v_load_expand(src + x);
                v_int32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_src1);
                v_store(dst + x + cWidth, v_src2);
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<schar, float>
{
    int operator() (const schar * src, float * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int16x8 v_src = v_load_expand(src + x);
                v_int32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_cvt_f32(v_src1));
                v_store(dst + x + cWidth, v_cvt_f32(v_src2));
            }
        }
        return x;
    }
};

// from ushort

template <>
struct Cvt_SIMD<ushort, uchar>
{
    int operator() (const ushort * src, uchar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_uint16x8 v_src1 = v_load(src + x), v_src2 = v_load(src + x + cWidth);
                v_store(dst + x, v_pack(v_src1, v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<ushort, schar>
{
    int operator() (const ushort * src, schar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_uint16x8 v_src1 = v_load(src + x), v_src2 = v_load(src + x + cWidth);
                v_uint32x4 v_dst10, v_dst11, v_dst20, v_dst21;
                v_expand(v_src1, v_dst10, v_dst11);
                v_expand(v_src2, v_dst20, v_dst21);

                v_store(dst + x, v_pack(
                    v_pack(v_reinterpret_as_s32(v_dst10), v_reinterpret_as_s32(v_dst11)),
                    v_pack(v_reinterpret_as_s32(v_dst20), v_reinterpret_as_s32(v_dst21))));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<ushort, short>
{
    int operator() (const ushort * src, short * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_uint16x8 v_src = v_load(src + x);
                v_uint32x4 v_dst0, v_dst1;
                v_expand(v_src, v_dst0, v_dst1);
                v_store(dst + x, v_pack(v_reinterpret_as_s32(v_dst0), v_reinterpret_as_s32(v_dst1)));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<ushort, int>
{
    int operator() (const ushort * src, int * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_uint16x8 v_src = v_load(src + x);
                v_uint32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_reinterpret_as_s32(v_src1));
                v_store(dst + x + cWidth, v_reinterpret_as_s32(v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<ushort, float>
{
    int operator() (const ushort * src, float * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_uint16x8 v_src = v_load(src + x);
                v_uint32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_cvt_f32(v_reinterpret_as_s32(v_src1)));
                v_store(dst + x + cWidth, v_cvt_f32(v_reinterpret_as_s32(v_src2)));
            }
        }
        return x;
    }
};


// from short

template <>
struct Cvt_SIMD<short, uchar>
{
    int operator() (const short * src, uchar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int16x8 v_src1 = v_load(src + x), v_src2 = v_load(src + x + cWidth);
                v_store(dst + x, v_pack_u(v_src1, v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<short, schar>
{
    int operator() (const short * src, schar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int16x8 v_src1 = v_load(src + x), v_src2 = v_load(src + x + cWidth);
                v_store(dst + x, v_pack(v_src1, v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<short, ushort>
{
    int operator() (const short * src, ushort * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_int16x8 v_src = v_load(src + x);
                v_int32x4 v_dst1, v_dst2;
                v_expand(v_src, v_dst1, v_dst2);
                v_store(dst + x, v_pack_u(v_dst1, v_dst2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<short, int>
{
    int operator() (const short * src, int * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int16x8 v_src = v_load(src + x);
                v_int32x4 v_dst1, v_dst2;
                v_expand(v_src, v_dst1, v_dst2);
                v_store(dst + x, v_dst1);
                v_store(dst + x + cWidth, v_dst2);
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<short, float>
{
    int operator() (const short * src, float * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int16x8 v_src = v_load(src + x);
                v_int32x4 v_dst1, v_dst2;
                v_expand(v_src, v_dst1, v_dst2);
                v_store(dst + x, v_cvt_f32(v_dst1));
                v_store(dst + x + cWidth, v_cvt_f32(v_dst2));
            }
        }
        return x;
    }
};

// from int

template <>
struct Cvt_SIMD<int, uchar>
{
    int operator() (const int * src, uchar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_int32x4 v_src1 = v_load(src + x), v_src2 = v_load(src + x + cWidth);
                v_int32x4 v_src3 = v_load(src + x + cWidth * 2), v_src4 = v_load(src + x + cWidth * 3);
                v_uint16x8 v_dst1 = v_pack_u(v_src1, v_src2);
                v_uint16x8 v_dst2 = v_pack_u(v_src3, v_src4);
                v_store(dst + x, v_pack(v_dst1, v_dst2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<int, schar>
{
    int operator() (const int * src, schar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_int32x4 v_src1 = v_load(src + x), v_src2 = v_load(src + x + cWidth);
                v_int32x4 v_src3 = v_load(src + x + cWidth * 2), v_src4 = v_load(src + x + cWidth * 3);
                v_int16x8 v_dst1 = v_pack(v_src1, v_src2);
                v_int16x8 v_dst2 = v_pack(v_src3, v_src4);
                v_store(dst + x, v_pack(v_dst1, v_dst2));
            }
        }
        return x;
    }
};


template <>
struct Cvt_SIMD<int, ushort>
{
    int operator() (const int * src, ushort * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int32x4 v_src1 = v_load(src + x), v_src2 = v_load(src + x + cWidth);
                v_store(dst + x, v_pack_u(v_src1, v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<int, short>
{
    int operator() (const int * src, short * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int32x4 v_src1 = v_load(src + x), v_src2 = v_load(src + x + cWidth);
                v_store(dst + x, v_pack(v_src1, v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<int, float>
{
    int operator() (const int * src, float * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth; x += cWidth)
                v_store(dst + x, v_cvt_f32(v_load(src + x)));
        }
        return x;
    }
};

// from float

template <>
struct Cvt_SIMD<float, uchar>
{
    int operator() (const float * src, uchar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_int32x4 v_src1 = v_round(v_load(src + x));
                v_int32x4 v_src2 = v_round(v_load(src + x + cWidth));
                v_int32x4 v_src3 = v_round(v_load(src + x + cWidth * 2));
                v_int32x4 v_src4 = v_round(v_load(src + x + cWidth * 3));
                v_uint16x8 v_dst1 = v_pack_u(v_src1, v_src2);
                v_uint16x8 v_dst2 = v_pack_u(v_src3, v_src4);
                v_store(dst + x, v_pack(v_dst1, v_dst2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<float, schar>
{
    int operator() (const float * src, schar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_int32x4 v_src1 = v_round(v_load(src + x));
                v_int32x4 v_src2 = v_round(v_load(src + x + cWidth));
                v_int32x4 v_src3 = v_round(v_load(src + x + cWidth * 2));
                v_int32x4 v_src4 = v_round(v_load(src + x + cWidth * 3));
                v_int16x8 v_dst1 = v_pack(v_src1, v_src2);
                v_int16x8 v_dst2 = v_pack(v_src3, v_src4);
                v_store(dst + x, v_pack(v_dst1, v_dst2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<float, ushort>
{
    int operator() (const float * src, ushort * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int32x4 v_src1 = v_round(v_load(src + x));
                v_int32x4 v_src2 = v_round(v_load(src + x + cWidth));
                v_store(dst + x, v_pack_u(v_src1, v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<float, short>
{
    int operator() (const float * src, short * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int32x4 v_src1 = v_round(v_load(src + x));
                v_int32x4 v_src2 = v_round(v_load(src + x + cWidth));
                v_store(dst + x, v_pack(v_src1, v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<float, int>
{
    int operator() (const float * src, int * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth; x += cWidth)
                v_store(dst + x, v_round(v_load(src + x)));
        }
        return x;
    }
};
#if CV_SIMD128_64F
// from double

template <>
struct Cvt_SIMD<double, uchar>
{
    int operator() (const double * src, uchar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_float32x4 v_src0 = v_cvt_f32(v_load(src + x));
                v_float32x4 v_src1 = v_cvt_f32(v_load(src + x + cWidth));
                v_float32x4 v_src2 = v_cvt_f32(v_load(src + x + cWidth * 2));
                v_float32x4 v_src3 = v_cvt_f32(v_load(src + x + cWidth * 3));

                v_src0 = v_combine_low(v_src0, v_src1);
                v_src1 = v_combine_low(v_src2, v_src3);

                v_int16x8 v_dst = v_pack(v_round(v_src0), v_round(v_src1));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<double, schar>
{
    int operator() (const double * src, schar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_float32x4 v_src0 = v_cvt_f32(v_load(src + x));
                v_float32x4 v_src1 = v_cvt_f32(v_load(src + x + cWidth));
                v_float32x4 v_src2 = v_cvt_f32(v_load(src + x + cWidth * 2));
                v_float32x4 v_src3 = v_cvt_f32(v_load(src + x + cWidth * 3));

                v_src0 = v_combine_low(v_src0, v_src1);
                v_src1 = v_combine_low(v_src2, v_src3);

                v_int16x8 v_dst = v_pack(v_round(v_src0), v_round(v_src1));
                v_store_low(dst + x, v_pack(v_dst, v_dst));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<double, ushort>
{
    int operator() (const double * src, ushort * dst, int width) const
    {
        int x = 0;
#if CV_TRY_SSE4_1
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::Cvt_SIMD_f64u16_SSE41(src, dst, width);
#endif
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_float32x4 v_src0 = v_cvt_f32(v_load(src + x));
                v_float32x4 v_src1 = v_cvt_f32(v_load(src + x + cWidth));
                v_float32x4 v_src2 = v_cvt_f32(v_load(src + x + cWidth * 2));
                v_float32x4 v_src3 = v_cvt_f32(v_load(src + x + cWidth * 3));

                v_src0 = v_combine_low(v_src0, v_src1);
                v_src1 = v_combine_low(v_src2, v_src3);

                v_uint16x8 v_dst = v_pack_u(v_round(v_src0), v_round(v_src1));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<double, short>
{
    int operator() (const double * src, short * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_float32x4 v_src0 = v_cvt_f32(v_load(src + x));
                v_float32x4 v_src1 = v_cvt_f32(v_load(src + x + cWidth));
                v_float32x4 v_src2 = v_cvt_f32(v_load(src + x + cWidth * 2));
                v_float32x4 v_src3 = v_cvt_f32(v_load(src + x + cWidth * 3));

                v_src0 = v_combine_low(v_src0, v_src1);
                v_src1 = v_combine_low(v_src2, v_src3);

                v_int16x8 v_dst = v_pack(v_round(v_src0), v_round(v_src1));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<double, int>
{
    int operator() (const double * src, int * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src0 = v_cvt_f32(v_load(src + x));
                v_float32x4 v_src1 = v_cvt_f32(v_load(src + x + cWidth));

                v_store(dst + x, v_round(v_combine_low(v_src0, v_src1)));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<double, float>
{
    int operator() (const double * src, float * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src0 = v_cvt_f32(v_load(src + x));
                v_float32x4 v_src1 = v_cvt_f32(v_load(src + x + cWidth));

                v_store(dst + x, v_combine_low(v_src0, v_src1));
            }
        }
        return x;
    }
};

// to double

template <>
struct Cvt_SIMD<uchar, double>
{
    int operator() (const uchar* src, double* dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_uint16x8 v_src = v_load_expand(src + x);
                v_uint32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_cvt_f64(v_reinterpret_as_s32(v_src1)));
                v_store(dst + x + cWidth, v_cvt_f64_high(v_reinterpret_as_s32(v_src1)));
                v_store(dst + x + cWidth * 2, v_cvt_f64(v_reinterpret_as_s32(v_src2)));
                v_store(dst + x + cWidth * 3, v_cvt_f64_high(v_reinterpret_as_s32(v_src2)));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<schar, double>
{
    int operator() (const schar* src, double* dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_int16x8 v_src = v_load_expand(src + x);
                v_int32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_cvt_f64(v_src1));
                v_store(dst + x + cWidth, v_cvt_f64_high(v_src1));
                v_store(dst + x + cWidth * 2, v_cvt_f64(v_src2));
                v_store(dst + x + cWidth * 3, v_cvt_f64_high(v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<ushort, double>
{
    int operator() (const ushort* src, double* dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_uint32x4 v_src = v_load_expand(src + x);

                v_store(dst + x, v_cvt_f64(v_reinterpret_as_s32(v_src)));
                v_store(dst + x + cWidth, v_cvt_f64_high(v_reinterpret_as_s32(v_src)));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<short, double>
{
    int operator() (const short* src, double* dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int32x4 v_src = v_load_expand(src + x);

                v_store(dst + x, v_cvt_f64(v_src));
                v_store(dst + x + cWidth, v_cvt_f64_high(v_src));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<int, double>
{
    int operator() (const int* src, double* dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int32x4 v_src = v_load(src + x);

                v_store(dst + x, v_cvt_f64(v_src));
                v_store(dst + x + cWidth, v_cvt_f64_high(v_src));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<float, double>
{
    int operator() (const float* src, double* dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src = v_load(src + x);

                v_store(dst + x, v_cvt_f64(v_src));
                v_store(dst + x + cWidth, v_cvt_f64_high(v_src));
            }
        }
        return x;
    }
};
#endif // CV_SIMD128_64F
#endif // CV_SIMD128

// template for FP16 HW conversion function
template<typename T, typename DT> static void
cvtScaleHalf_( const T* src, size_t sstep, DT* dst, size_t dstep, Size size);

template<> void
cvtScaleHalf_<float, short>( const float* src, size_t sstep, short* dst, size_t dstep, Size size )
{
    CV_CPU_CALL_FP16(cvtScaleHalf_SIMD32f16f, (src, sstep, dst, dstep, size));

#if !defined(CV_CPU_COMPILE_FP16)
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        for ( int x = 0; x < size.width; x++ )
        {
            dst[x] = convertFp16SW(src[x]);
        }
    }
#endif
}

template<> void
cvtScaleHalf_<short, float>( const short* src, size_t sstep, float* dst, size_t dstep, Size size )
{
    CV_CPU_CALL_FP16(cvtScaleHalf_SIMD16f32f, (src, sstep, dst, dstep, size));

#if !defined(CV_CPU_COMPILE_FP16)
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        for ( int x = 0; x < size.width; x++ )
        {
            dst[x] = convertFp16SW(src[x]);
        }
    }
#endif
}

#ifdef HAVE_OPENVX

template<typename T, typename DT>
static bool _openvx_cvt(const T* src, size_t sstep,
                        DT* dst, size_t dstep, Size continuousSize)
{
    using namespace ivx;

    if(!(continuousSize.width > 0 && continuousSize.height > 0))
    {
        return true;
    }

    //.height is for number of continuous pieces
    //.width  is for length of one piece
    Size imgSize = continuousSize;
    if(continuousSize.height == 1)
    {
        if(sstep / sizeof(T) == dstep / sizeof(DT) && sstep / sizeof(T) > 0 &&
           continuousSize.width % (sstep / sizeof(T)) == 0)
        {
            //continuous n-lines image
            imgSize.width  = sstep / sizeof(T);
            imgSize.height = continuousSize.width / (sstep / sizeof(T));
        }
        else
        {
            //1-row image with possibly incorrect step
            sstep = continuousSize.width * sizeof(T);
            dstep = continuousSize.width * sizeof(DT);
        }
    }

    int srcType = DataType<T>::type, dstType = DataType<DT>::type;

    if (ovx::skipSmallImages<VX_KERNEL_CONVERTDEPTH>(imgSize.width, imgSize.height))
        return false;

    try
    {
        Context context = ovx::getOpenVXContext();

        // Other conversions are marked as "experimental"
        if(context.vendorID() == VX_ID_KHRONOS &&
           !(srcType == CV_8U  && dstType == CV_16S) &&
           !(srcType == CV_16S && dstType == CV_8U))
        {
            return false;
        }

        Image srcImage = Image::createFromHandle(context, Image::matTypeToFormat(srcType),
                                                 Image::createAddressing(imgSize.width, imgSize.height,
                                                                         (vx_uint32)sizeof(T), (vx_uint32)sstep),
                                                 (void*)src);
        Image dstImage = Image::createFromHandle(context, Image::matTypeToFormat(dstType),
                                                 Image::createAddressing(imgSize.width, imgSize.height,
                                                                         (vx_uint32)sizeof(DT), (vx_uint32)dstep),
                                                 (void*)dst);

        IVX_CHECK_STATUS(vxuConvertDepth(context, srcImage, dstImage, VX_CONVERT_POLICY_SATURATE, 0));

#ifdef VX_VERSION_1_1
        //we should take user memory back before release
        //(it's not done automatically according to standard)
        srcImage.swapHandle(); dstImage.swapHandle();
#endif
    }
    catch (RuntimeError & e)
    {
        VX_DbgThrow(e.what());
    }
    catch (WrapperError & e)
    {
        VX_DbgThrow(e.what());
    }

    return true;
}

template<typename T, typename DT>
static bool openvx_cvt(const T* src, size_t sstep,
                       DT* dst, size_t dstep, Size size)
{
    (void)src; (void)sstep; (void)dst; (void)dstep; (void)size;
    return false;
}

#define DEFINE_OVX_CVT_SPECIALIZATION(T, DT) \
template<>                                                                    \
bool openvx_cvt(const T *src, size_t sstep, DT *dst, size_t dstep, Size size) \
{                                                                             \
    return _openvx_cvt<T, DT>(src, sstep, dst, dstep, size);                  \
}

DEFINE_OVX_CVT_SPECIALIZATION(uchar, ushort)
DEFINE_OVX_CVT_SPECIALIZATION(uchar, short)
DEFINE_OVX_CVT_SPECIALIZATION(uchar, int)
DEFINE_OVX_CVT_SPECIALIZATION(ushort, uchar)
DEFINE_OVX_CVT_SPECIALIZATION(ushort, int)
DEFINE_OVX_CVT_SPECIALIZATION(short, uchar)
DEFINE_OVX_CVT_SPECIALIZATION(short, int)
DEFINE_OVX_CVT_SPECIALIZATION(int, uchar)
DEFINE_OVX_CVT_SPECIALIZATION(int, ushort)
DEFINE_OVX_CVT_SPECIALIZATION(int, short)

#endif

template<typename T, typename DT> static void
cvt_( const T* src, size_t sstep,
      DT* dst, size_t dstep, Size size )
{
    CV_OVX_RUN(
        true,
        openvx_cvt(src, sstep, dst, dstep, size)
    )

    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);
    Cvt_SIMD<T, DT> vop;

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = vop(src, dst, size.width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = saturate_cast<DT>(src[x]);
            t1 = saturate_cast<DT>(src[x+1]);
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<DT>(src[x+2]);
            t1 = saturate_cast<DT>(src[x+3]);
            dst[x+2] = t0; dst[x+3] = t1;
        }
        #endif
        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<DT>(src[x]);
    }
}

template<typename T> static void
cpy_( const T* src, size_t sstep, T* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
        memcpy(dst, src, size.width*sizeof(src[0]));
}

#define DEF_CVT_SCALE_ABS_FUNC(suffix, tfunc, stype, dtype, wtype) \
static void cvtScaleAbs##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double* scale) \
{ \
    tfunc(src, sstep, dst, dstep, size, (wtype)scale[0], (wtype)scale[1]); \
}

#define DEF_CVT_SCALE_FP16_FUNC(suffix, stype, dtype) \
static void cvtScaleHalf##suffix( const stype* src, size_t sstep, \
dtype* dst, size_t dstep, Size size) \
{ \
    cvtScaleHalf_<stype,dtype>(src, sstep, dst, dstep, size); \
}


#define DEF_CVT_SCALE_FUNC(suffix, stype, dtype, wtype) \
static void cvtScale##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
dtype* dst, size_t dstep, Size size, double* scale) \
{ \
    cvtScale_(src, sstep, dst, dstep, size, (wtype)scale[0], (wtype)scale[1]); \
}

#if defined(HAVE_IPP)
#define DEF_CVT_FUNC_F(suffix, stype, dtype, ippFavor) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    CV_IPP_RUN(src && dst, CV_INSTRUMENT_FUN_IPP(ippiConvert_##ippFavor, src, (int)sstep, dst, (int)dstep, ippiSize(size.width, size.height)) >= 0) \
    cvt_(src, sstep, dst, dstep, size); \
}

#define DEF_CVT_FUNC_F2(suffix, stype, dtype, ippFavor) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    CV_IPP_RUN(src && dst, CV_INSTRUMENT_FUN_IPP(ippiConvert_##ippFavor, src, (int)sstep, dst, (int)dstep, ippiSize(size.width, size.height), ippRndFinancial, 0) >= 0) \
    cvt_(src, sstep, dst, dstep, size); \
}
#else
#define DEF_CVT_FUNC_F(suffix, stype, dtype, ippFavor) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    cvt_(src, sstep, dst, dstep, size); \
}
#define DEF_CVT_FUNC_F2 DEF_CVT_FUNC_F
#endif

#define DEF_CVT_FUNC(suffix, stype, dtype) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    cvt_(src, sstep, dst, dstep, size); \
}

#define DEF_CPY_FUNC(suffix, stype) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         stype* dst, size_t dstep, Size size, double*) \
{ \
    cpy_(src, sstep, dst, dstep, size); \
}


DEF_CVT_SCALE_ABS_FUNC(8u, cvtScaleAbs_, uchar, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(8s8u, cvtScaleAbs_, schar, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16u8u, cvtScaleAbs_, ushort, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16s8u, cvtScaleAbs_, short, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32s8u, cvtScaleAbs_, int, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32f8u, cvtScaleAbs_, float, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(64f8u, cvtScaleAbs_, double, uchar, float)

DEF_CVT_SCALE_FP16_FUNC(32f16f, float, short)
DEF_CVT_SCALE_FP16_FUNC(16f32f, short, float)

DEF_CVT_SCALE_FUNC(8u,     uchar, uchar, float)
DEF_CVT_SCALE_FUNC(8s8u,   schar, uchar, float)
DEF_CVT_SCALE_FUNC(16u8u,  ushort, uchar, float)
DEF_CVT_SCALE_FUNC(16s8u,  short, uchar, float)
DEF_CVT_SCALE_FUNC(32s8u,  int, uchar, float)
DEF_CVT_SCALE_FUNC(32f8u,  float, uchar, float)
DEF_CVT_SCALE_FUNC(64f8u,  double, uchar, float)

DEF_CVT_SCALE_FUNC(8u8s,   uchar, schar, float)
DEF_CVT_SCALE_FUNC(8s,     schar, schar, float)
DEF_CVT_SCALE_FUNC(16u8s,  ushort, schar, float)
DEF_CVT_SCALE_FUNC(16s8s,  short, schar, float)
DEF_CVT_SCALE_FUNC(32s8s,  int, schar, float)
DEF_CVT_SCALE_FUNC(32f8s,  float, schar, float)
DEF_CVT_SCALE_FUNC(64f8s,  double, schar, float)

DEF_CVT_SCALE_FUNC(8u16u,  uchar, ushort, float)
DEF_CVT_SCALE_FUNC(8s16u,  schar, ushort, float)
DEF_CVT_SCALE_FUNC(16u,    ushort, ushort, float)
DEF_CVT_SCALE_FUNC(16s16u, short, ushort, float)
DEF_CVT_SCALE_FUNC(32s16u, int, ushort, float)
DEF_CVT_SCALE_FUNC(32f16u, float, ushort, float)
DEF_CVT_SCALE_FUNC(64f16u, double, ushort, float)

DEF_CVT_SCALE_FUNC(8u16s,  uchar, short, float)
DEF_CVT_SCALE_FUNC(8s16s,  schar, short, float)
DEF_CVT_SCALE_FUNC(16u16s, ushort, short, float)
DEF_CVT_SCALE_FUNC(16s,    short, short, float)
DEF_CVT_SCALE_FUNC(32s16s, int, short, float)
DEF_CVT_SCALE_FUNC(32f16s, float, short, float)
DEF_CVT_SCALE_FUNC(64f16s, double, short, float)

DEF_CVT_SCALE_FUNC(8u32s,  uchar, int, float)
DEF_CVT_SCALE_FUNC(8s32s,  schar, int, float)
DEF_CVT_SCALE_FUNC(16u32s, ushort, int, float)
DEF_CVT_SCALE_FUNC(16s32s, short, int, float)
DEF_CVT_SCALE_FUNC(32s,    int, int, double)
DEF_CVT_SCALE_FUNC(32f32s, float, int, float)
DEF_CVT_SCALE_FUNC(64f32s, double, int, double)

DEF_CVT_SCALE_FUNC(8u32f,  uchar, float, float)
DEF_CVT_SCALE_FUNC(8s32f,  schar, float, float)
DEF_CVT_SCALE_FUNC(16u32f, ushort, float, float)
DEF_CVT_SCALE_FUNC(16s32f, short, float, float)
DEF_CVT_SCALE_FUNC(32s32f, int, float, double)
DEF_CVT_SCALE_FUNC(32f,    float, float, float)
DEF_CVT_SCALE_FUNC(64f32f, double, float, double)

DEF_CVT_SCALE_FUNC(8u64f,  uchar, double, double)
DEF_CVT_SCALE_FUNC(8s64f,  schar, double, double)
DEF_CVT_SCALE_FUNC(16u64f, ushort, double, double)
DEF_CVT_SCALE_FUNC(16s64f, short, double, double)
DEF_CVT_SCALE_FUNC(32s64f, int, double, double)
DEF_CVT_SCALE_FUNC(32f64f, float, double, double)
DEF_CVT_SCALE_FUNC(64f,    double, double, double)

DEF_CPY_FUNC(8u,     uchar)
DEF_CVT_FUNC_F(8s8u,   schar, uchar, 8s8u_C1Rs)
DEF_CVT_FUNC_F(16u8u,  ushort, uchar, 16u8u_C1R)
DEF_CVT_FUNC_F(16s8u,  short, uchar, 16s8u_C1R)
DEF_CVT_FUNC_F(32s8u,  int, uchar, 32s8u_C1R)
DEF_CVT_FUNC_F2(32f8u,  float, uchar, 32f8u_C1RSfs)
DEF_CVT_FUNC(64f8u,  double, uchar)

DEF_CVT_FUNC_F2(8u8s,   uchar, schar, 8u8s_C1RSfs)
DEF_CVT_FUNC_F2(16u8s,  ushort, schar, 16u8s_C1RSfs)
DEF_CVT_FUNC_F2(16s8s,  short, schar, 16s8s_C1RSfs)
DEF_CVT_FUNC_F(32s8s,  int, schar, 32s8s_C1R)
DEF_CVT_FUNC_F2(32f8s,  float, schar, 32f8s_C1RSfs)
DEF_CVT_FUNC(64f8s,  double, schar)

DEF_CVT_FUNC_F(8u16u,  uchar, ushort, 8u16u_C1R)
DEF_CVT_FUNC_F(8s16u,  schar, ushort, 8s16u_C1Rs)
DEF_CPY_FUNC(16u,    ushort)
DEF_CVT_FUNC_F(16s16u, short, ushort, 16s16u_C1Rs)
DEF_CVT_FUNC_F2(32s16u, int, ushort, 32s16u_C1RSfs)
DEF_CVT_FUNC_F2(32f16u, float, ushort, 32f16u_C1RSfs)
DEF_CVT_FUNC(64f16u, double, ushort)

DEF_CVT_FUNC_F(8u16s,  uchar, short, 8u16s_C1R)
DEF_CVT_FUNC_F(8s16s,  schar, short, 8s16s_C1R)
DEF_CVT_FUNC_F2(16u16s, ushort, short, 16u16s_C1RSfs)
DEF_CVT_FUNC_F2(32s16s, int, short, 32s16s_C1RSfs)
DEF_CVT_FUNC(32f16s, float, short)
DEF_CVT_FUNC(64f16s, double, short)

DEF_CVT_FUNC_F(8u32s,  uchar, int, 8u32s_C1R)
DEF_CVT_FUNC_F(8s32s,  schar, int, 8s32s_C1R)
DEF_CVT_FUNC_F(16u32s, ushort, int, 16u32s_C1R)
DEF_CVT_FUNC_F(16s32s, short, int, 16s32s_C1R)
DEF_CPY_FUNC(32s,    int)
DEF_CVT_FUNC_F2(32f32s, float, int, 32f32s_C1RSfs)
DEF_CVT_FUNC(64f32s, double, int)

DEF_CVT_FUNC_F(8u32f,  uchar, float, 8u32f_C1R)
DEF_CVT_FUNC_F(8s32f,  schar, float, 8s32f_C1R)
DEF_CVT_FUNC_F(16u32f, ushort, float, 16u32f_C1R)
DEF_CVT_FUNC_F(16s32f, short, float, 16s32f_C1R)
DEF_CVT_FUNC_F(32s32f, int, float, 32s32f_C1R)
DEF_CVT_FUNC(64f32f, double, float)

DEF_CVT_FUNC(8u64f,  uchar, double)
DEF_CVT_FUNC(8s64f,  schar, double)
DEF_CVT_FUNC(16u64f, ushort, double)
DEF_CVT_FUNC(16s64f, short, double)
DEF_CVT_FUNC(32s64f, int, double)
DEF_CVT_FUNC(32f64f, float, double)
DEF_CPY_FUNC(64s,    int64)

static BinaryFunc getCvtScaleAbsFunc(int depth)
{
    static BinaryFunc cvtScaleAbsTab[] =
    {
        (BinaryFunc)cvtScaleAbs8u, (BinaryFunc)cvtScaleAbs8s8u, (BinaryFunc)cvtScaleAbs16u8u,
        (BinaryFunc)cvtScaleAbs16s8u, (BinaryFunc)cvtScaleAbs32s8u, (BinaryFunc)cvtScaleAbs32f8u,
        (BinaryFunc)cvtScaleAbs64f8u, 0
    };

    return cvtScaleAbsTab[depth];
}

typedef void (*UnaryFunc)(const uchar* src1, size_t step1,
                       uchar* dst, size_t step, Size sz,
                       void*);

static UnaryFunc getConvertFuncFp16(int ddepth)
{
    static UnaryFunc cvtTab[] =
    {
        0, 0, 0,
        (UnaryFunc)(cvtScaleHalf32f16f), 0, (UnaryFunc)(cvtScaleHalf16f32f),
        0, 0,
    };
    return cvtTab[CV_MAT_DEPTH(ddepth)];
}

BinaryFunc getConvertFunc(int sdepth, int ddepth)
{
    static BinaryFunc cvtTab[][8] =
    {
        {
            (BinaryFunc)(cvt8u), (BinaryFunc)GET_OPTIMIZED(cvt8s8u), (BinaryFunc)GET_OPTIMIZED(cvt16u8u),
            (BinaryFunc)GET_OPTIMIZED(cvt16s8u), (BinaryFunc)GET_OPTIMIZED(cvt32s8u), (BinaryFunc)GET_OPTIMIZED(cvt32f8u),
            (BinaryFunc)GET_OPTIMIZED(cvt64f8u), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u8s), (BinaryFunc)cvt8u, (BinaryFunc)GET_OPTIMIZED(cvt16u8s),
            (BinaryFunc)GET_OPTIMIZED(cvt16s8s), (BinaryFunc)GET_OPTIMIZED(cvt32s8s), (BinaryFunc)GET_OPTIMIZED(cvt32f8s),
            (BinaryFunc)GET_OPTIMIZED(cvt64f8s), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u16u), (BinaryFunc)GET_OPTIMIZED(cvt8s16u), (BinaryFunc)cvt16u,
            (BinaryFunc)GET_OPTIMIZED(cvt16s16u), (BinaryFunc)GET_OPTIMIZED(cvt32s16u), (BinaryFunc)GET_OPTIMIZED(cvt32f16u),
            (BinaryFunc)GET_OPTIMIZED(cvt64f16u), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u16s), (BinaryFunc)GET_OPTIMIZED(cvt8s16s), (BinaryFunc)GET_OPTIMIZED(cvt16u16s),
            (BinaryFunc)cvt16u, (BinaryFunc)GET_OPTIMIZED(cvt32s16s), (BinaryFunc)GET_OPTIMIZED(cvt32f16s),
            (BinaryFunc)GET_OPTIMIZED(cvt64f16s), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u32s), (BinaryFunc)GET_OPTIMIZED(cvt8s32s), (BinaryFunc)GET_OPTIMIZED(cvt16u32s),
            (BinaryFunc)GET_OPTIMIZED(cvt16s32s), (BinaryFunc)cvt32s, (BinaryFunc)GET_OPTIMIZED(cvt32f32s),
            (BinaryFunc)GET_OPTIMIZED(cvt64f32s), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u32f), (BinaryFunc)GET_OPTIMIZED(cvt8s32f), (BinaryFunc)GET_OPTIMIZED(cvt16u32f),
            (BinaryFunc)GET_OPTIMIZED(cvt16s32f), (BinaryFunc)GET_OPTIMIZED(cvt32s32f), (BinaryFunc)cvt32s,
            (BinaryFunc)GET_OPTIMIZED(cvt64f32f), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u64f), (BinaryFunc)GET_OPTIMIZED(cvt8s64f), (BinaryFunc)GET_OPTIMIZED(cvt16u64f),
            (BinaryFunc)GET_OPTIMIZED(cvt16s64f), (BinaryFunc)GET_OPTIMIZED(cvt32s64f), (BinaryFunc)GET_OPTIMIZED(cvt32f64f),
            (BinaryFunc)(cvt64s), 0
        },
        {
            0, 0, 0, 0, 0, 0, 0, 0
        }
    };

    return cvtTab[CV_MAT_DEPTH(ddepth)][CV_MAT_DEPTH(sdepth)];
}

static BinaryFunc getConvertScaleFunc(int sdepth, int ddepth)
{
    static BinaryFunc cvtScaleTab[][8] =
    {
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u), (BinaryFunc)GET_OPTIMIZED(cvtScale8s8u), (BinaryFunc)GET_OPTIMIZED(cvtScale16u8u),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s8u), (BinaryFunc)GET_OPTIMIZED(cvtScale32s8u), (BinaryFunc)GET_OPTIMIZED(cvtScale32f8u),
            (BinaryFunc)cvtScale64f8u, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u8s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u8s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s8s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s8s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f8s),
            (BinaryFunc)cvtScale64f8s, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u16u), (BinaryFunc)GET_OPTIMIZED(cvtScale8s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale16u),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale32s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale32f16u),
            (BinaryFunc)cvtScale64f16u, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u16s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s16s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u16s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s16s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f16s),
            (BinaryFunc)cvtScale64f16s, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u32s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s32s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u32s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s32s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f32s),
            (BinaryFunc)cvtScale64f32s, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u32f), (BinaryFunc)GET_OPTIMIZED(cvtScale8s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale16u32f),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale32s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale32f),
            (BinaryFunc)cvtScale64f32f, 0
        },
        {
            (BinaryFunc)cvtScale8u64f, (BinaryFunc)cvtScale8s64f, (BinaryFunc)cvtScale16u64f,
            (BinaryFunc)cvtScale16s64f, (BinaryFunc)cvtScale32s64f, (BinaryFunc)cvtScale32f64f,
            (BinaryFunc)cvtScale64f, 0
        },
        {
            0, 0, 0, 0, 0, 0, 0, 0
        }
    };

    return cvtScaleTab[CV_MAT_DEPTH(ddepth)][CV_MAT_DEPTH(sdepth)];
}

#ifdef HAVE_OPENCL

static bool ocl_convertScaleAbs( InputArray _src, OutputArray _dst, double alpha, double beta )
{
    const ocl::Device & d = ocl::Device::getDefault();

    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    bool doubleSupport = d.doubleFPConfig() > 0;
    if (!doubleSupport && depth == CV_64F)
        return false;

    _dst.create(_src.size(), CV_8UC(cn));
    int kercn = 1;
    if (d.isIntel())
    {
        static const int vectorWidths[] = {4, 4, 4, 4, 4, 4, 4, -1};
        kercn = ocl::checkOptimalVectorWidth( vectorWidths, _src, _dst,
                                              noArray(), noArray(), noArray(),
                                              noArray(), noArray(), noArray(),
                                              noArray(), ocl::OCL_VECTOR_MAX);
    }
    else
        kercn = ocl::predictOptimalVectorWidthMax(_src, _dst);

    int rowsPerWI = d.isIntel() ? 4 : 1;
    char cvt[2][50];
    int wdepth = std::max(depth, CV_32F);
    String build_opt = format("-D OP_CONVERT_SCALE_ABS -D UNARY_OP -D dstT=%s -D srcT1=%s"
                         " -D workT=%s -D wdepth=%d -D convertToWT1=%s -D convertToDT=%s"
                         " -D workT1=%s -D rowsPerWI=%d%s",
                         ocl::typeToStr(CV_8UC(kercn)),
                         ocl::typeToStr(CV_MAKE_TYPE(depth, kercn)),
                         ocl::typeToStr(CV_MAKE_TYPE(wdepth, kercn)), wdepth,
                         ocl::convertTypeStr(depth, wdepth, kercn, cvt[0]),
                         ocl::convertTypeStr(wdepth, CV_8U, kercn, cvt[1]),
                         ocl::typeToStr(wdepth), rowsPerWI,
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "");
    ocl::Kernel k("KF", ocl::core::arithm_oclsrc, build_opt);
    if (k.empty())
        return false;

    UMat src = _src.getUMat();
    UMat dst = _dst.getUMat();

    ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
            dstarg = ocl::KernelArg::WriteOnly(dst, cn, kercn);

    if (wdepth == CV_32F)
        k.args(srcarg, dstarg, (float)alpha, (float)beta);
    else if (wdepth == CV_64F)
        k.args(srcarg, dstarg, alpha, beta);

    size_t globalsize[2] = { (size_t)src.cols * cn / kercn, ((size_t)src.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

static bool ocl_convertFp16( InputArray _src, OutputArray _dst, int ddepth )
{
    int type = _src.type(), cn = CV_MAT_CN(type);

    _dst.createSameSize( _src, CV_MAKETYPE(ddepth, cn) );
    int kercn = 1;
    int rowsPerWI = 1;
    String build_opt = format("-D HALF_SUPPORT -D dstT=%s -D srcT=%s -D rowsPerWI=%d%s",
                           ddepth == CV_16S ? "half" : "float",
                           ddepth == CV_16S ? "float" : "half",
                           rowsPerWI,
                           ddepth == CV_16S ? " -D FLOAT_TO_HALF " : "");
    ocl::Kernel k("convertFp16", ocl::core::halfconvert_oclsrc, build_opt);
    if (k.empty())
        return false;

    UMat src = _src.getUMat();
    UMat dst = _dst.getUMat();

    ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
            dstarg = ocl::KernelArg::WriteOnly(dst, cn, kercn);

    k.args(srcarg, dstarg);

    size_t globalsize[2] = { (size_t)src.cols * cn / kercn, ((size_t)src.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

}

void cv::convertScaleAbs( InputArray _src, OutputArray _dst, double alpha, double beta )
{
    CV_INSTRUMENT_REGION()

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_convertScaleAbs(_src, _dst, alpha, beta))

    Mat src = _src.getMat();
    int cn = src.channels();
    double scale[] = {alpha, beta};
    _dst.create( src.dims, src.size, CV_8UC(cn) );
    Mat dst = _dst.getMat();
    BinaryFunc func = getCvtScaleAbsFunc(src.depth());
    CV_Assert( func != 0 );

    if( src.dims <= 2 )
    {
        Size sz = getContinuousSize(src, dst, cn);
        func( src.ptr(), src.step, 0, 0, dst.ptr(), dst.step, sz, scale );
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)it.size*cn, 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func( ptrs[0], 0, 0, 0, ptrs[1], 0, sz, scale );
    }
}

void cv::convertFp16( InputArray _src, OutputArray _dst)
{
    CV_INSTRUMENT_REGION()

    int ddepth = 0;
    switch( _src.depth() )
    {
    case CV_32F:
        ddepth = CV_16S;
        break;
    case CV_16S:
        ddepth = CV_32F;
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "Unsupported input depth");
        return;
    }

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_convertFp16(_src, _dst, ddepth))

    Mat src = _src.getMat();

    int type = CV_MAKETYPE(ddepth, src.channels());
    _dst.create( src.dims, src.size, type );
    Mat dst = _dst.getMat();
    UnaryFunc func = getConvertFuncFp16(ddepth);
    int cn = src.channels();
    CV_Assert( func != 0 );

    if( src.dims <= 2 )
    {
        Size sz = getContinuousSize(src, dst, cn);
        func( src.data, src.step, dst.data, dst.step, sz, 0);
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)(it.size*cn), 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func(ptrs[0], 1, ptrs[1], 1, sz, 0);
    }
}

#ifdef HAVE_IPP
namespace cv
{
static bool ipp_convertTo(Mat &src, Mat &dst, double alpha, double beta)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP()

    IppDataType srcDepth = ippiGetDataType(src.depth());
    IppDataType dstDepth = ippiGetDataType(dst.depth());
    int         channels = src.channels();

    if(src.dims == 0)
        return false;

    ::ipp::IwiImage iwSrc;
    ::ipp::IwiImage iwDst;

    try
    {
        IppHintAlgorithm mode = ippAlgHintFast;
        if(dstDepth == ipp64f ||
            (dstDepth == ipp32f && (srcDepth == ipp32s || srcDepth == ipp64f)) ||
            (dstDepth == ipp32s && (srcDepth == ipp32s || srcDepth == ipp64f)))
            mode = ippAlgHintAccurate;

        if(src.dims <= 2)
        {
            Size sz = getContinuousSize(src, dst, channels);

            iwSrc.Init(ippiSize(sz), srcDepth, 1, NULL, (void*)src.ptr(), src.step);
            iwDst.Init(ippiSize(sz), dstDepth, 1, NULL, (void*)dst.ptr(), dst.step);

            CV_INSTRUMENT_FUN_IPP(::ipp::iwiScale, iwSrc, iwDst, alpha, beta, ::ipp::IwiScaleParams(mode));
        }
        else
        {
            const Mat *arrays[] = {&src, &dst, NULL};
            uchar     *ptrs[2]  = {NULL};
            NAryMatIterator it(arrays, ptrs);

            iwSrc.Init(ippiSize(it.size, 1), srcDepth, channels);
            iwDst.Init(ippiSize(it.size, 1), dstDepth, channels);

            for(size_t i = 0; i < it.nplanes; i++, ++it)
            {
                iwSrc.m_ptr  = ptrs[0];
                iwDst.m_ptr  = ptrs[1];

                CV_INSTRUMENT_FUN_IPP(::ipp::iwiScale, iwSrc, iwDst, alpha, beta, ::ipp::IwiScaleParams(mode));
            }
        }
    }
    catch (::ipp::IwException)
    {
        return false;
    }
    return true;
#else
    CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(alpha); CV_UNUSED(beta);
    return false;
#endif
}
}
#endif

void cv::Mat::convertTo(OutputArray _dst, int _type, double alpha, double beta) const
{
    CV_INSTRUMENT_REGION()

    bool noScale = fabs(alpha-1) < DBL_EPSILON && fabs(beta) < DBL_EPSILON;

    if( _type < 0 )
        _type = _dst.fixedType() ? _dst.type() : type();
    else
        _type = CV_MAKETYPE(CV_MAT_DEPTH(_type), channels());

    int sdepth = depth(), ddepth = CV_MAT_DEPTH(_type);
    if( sdepth == ddepth && noScale )
    {
        copyTo(_dst);
        return;
    }

    Mat src = *this;
    if( dims <= 2 )
        _dst.create( size(), _type );
    else
        _dst.create( dims, size, _type );
    Mat dst = _dst.getMat();

    CV_IPP_RUN_FAST(ipp_convertTo(src, dst, alpha, beta ));

    BinaryFunc func = noScale ? getConvertFunc(sdepth, ddepth) : getConvertScaleFunc(sdepth, ddepth);
    double scale[] = {alpha, beta};
    int cn = channels();
    CV_Assert( func != 0 );

    if( dims <= 2 )
    {
        Size sz = getContinuousSize(src, dst, cn);

        func( src.data, src.step, 0, 0, dst.data, dst.step, sz, scale );
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)(it.size*cn), 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func(ptrs[0], 1, 0, 0, ptrs[1], 1, sz, scale);
    }
}

/****************************************************************************************\
*                                    LUT Transform                                       *
\****************************************************************************************/

namespace cv
{

template<typename T> static void
LUT8u_( const uchar* src, const T* lut, T* dst, int len, int cn, int lutcn )
{
    if( lutcn == 1 )
    {
        for( int i = 0; i < len*cn; i++ )
            dst[i] = lut[src[i]];
    }
    else
    {
        for( int i = 0; i < len*cn; i += cn )
            for( int k = 0; k < cn; k++ )
                dst[i+k] = lut[src[i+k]*cn+k];
    }
}

static void LUT8u_8u( const uchar* src, const uchar* lut, uchar* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_8s( const uchar* src, const schar* lut, schar* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_16u( const uchar* src, const ushort* lut, ushort* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_16s( const uchar* src, const short* lut, short* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_32s( const uchar* src, const int* lut, int* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_32f( const uchar* src, const float* lut, float* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_64f( const uchar* src, const double* lut, double* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

typedef void (*LUTFunc)( const uchar* src, const uchar* lut, uchar* dst, int len, int cn, int lutcn );

static LUTFunc lutTab[] =
{
    (LUTFunc)LUT8u_8u, (LUTFunc)LUT8u_8s, (LUTFunc)LUT8u_16u, (LUTFunc)LUT8u_16s,
    (LUTFunc)LUT8u_32s, (LUTFunc)LUT8u_32f, (LUTFunc)LUT8u_64f, 0
};

#ifdef HAVE_OPENCL

static bool ocl_LUT(InputArray _src, InputArray _lut, OutputArray _dst)
{
    int lcn = _lut.channels(), dcn = _src.channels(), ddepth = _lut.depth();

    UMat src = _src.getUMat(), lut = _lut.getUMat();
    _dst.create(src.size(), CV_MAKETYPE(ddepth, dcn));
    UMat dst = _dst.getUMat();
    int kercn = lcn == 1 ? std::min(4, ocl::predictOptimalVectorWidth(_src, _dst)) : dcn;

    ocl::Kernel k("LUT", ocl::core::lut_oclsrc,
                  format("-D dcn=%d -D lcn=%d -D srcT=%s -D dstT=%s", kercn, lcn,
                         ocl::typeToStr(src.depth()), ocl::memopTypeToStr(ddepth)));
    if (k.empty())
        return false;

    k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::ReadOnlyNoSize(lut),
        ocl::KernelArg::WriteOnly(dst, dcn, kercn));

    size_t globalSize[2] = { (size_t)dst.cols * dcn / kercn, ((size_t)dst.rows + 3) / 4 };
    return k.run(2, globalSize, NULL, false);
}

#endif

#ifdef HAVE_OPENVX
static bool openvx_LUT(Mat src, Mat dst, Mat _lut)
{
    if (src.type() != CV_8UC1 || dst.type() != src.type() || _lut.type() != src.type() || !_lut.isContinuous())
        return false;

    try
    {
        ivx::Context ctx = ovx::getOpenVXContext();

        ivx::Image
            ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(src.cols, src.rows, 1, (vx_int32)(src.step)), src.data),
            ib = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(dst.cols, dst.rows, 1, (vx_int32)(dst.step)), dst.data);

        ivx::LUT lut = ivx::LUT::create(ctx);
        lut.copyFrom(_lut);
        ivx::IVX_CHECK_STATUS(vxuTableLookup(ctx, ia, lut, ib));
    }
    catch (ivx::RuntimeError & e)
    {
        VX_DbgThrow(e.what());
    }
    catch (ivx::WrapperError & e)
    {
        VX_DbgThrow(e.what());
    }

    return true;
}
#endif

#if defined(HAVE_IPP)
#if !IPP_DISABLE_PERF_LUT // there are no performance benefits (PR #2653)
namespace ipp {

class IppLUTParallelBody_LUTC1 : public ParallelLoopBody
{
public:
    bool* ok;
    const Mat& src_;
    const Mat& lut_;
    Mat& dst_;

    int width;
    size_t elemSize1;

    IppLUTParallelBody_LUTC1(const Mat& src, const Mat& lut, Mat& dst, bool* _ok)
        : ok(_ok), src_(src), lut_(lut), dst_(dst)
    {
        width = dst.cols * dst.channels();
        elemSize1 = CV_ELEM_SIZE1(dst.depth());

        CV_DbgAssert(elemSize1 == 1 || elemSize1 == 4);
        *ok = true;
    }

    void operator()( const cv::Range& range ) const
    {
        if (!*ok)
            return;

        const int row0 = range.start;
        const int row1 = range.end;

        Mat src = src_.rowRange(row0, row1);
        Mat dst = dst_.rowRange(row0, row1);

        IppiSize sz = { width, dst.rows };

        if (elemSize1 == 1)
        {
            if (CV_INSTRUMENT_FUN_IPP(ippiLUTPalette_8u_C1R, (const Ipp8u*)src.data, (int)src.step[0], dst.data, (int)dst.step[0], sz, lut_.data, 8) >= 0)
                return;
        }
        else if (elemSize1 == 4)
        {
            if (CV_INSTRUMENT_FUN_IPP(ippiLUTPalette_8u32u_C1R, (const Ipp8u*)src.data, (int)src.step[0], (Ipp32u*)dst.data, (int)dst.step[0], sz, (Ipp32u*)lut_.data, 8) >= 0)
                return;
        }
        *ok = false;
    }
private:
    IppLUTParallelBody_LUTC1(const IppLUTParallelBody_LUTC1&);
    IppLUTParallelBody_LUTC1& operator=(const IppLUTParallelBody_LUTC1&);
};

class IppLUTParallelBody_LUTCN : public ParallelLoopBody
{
public:
    bool *ok;
    const Mat& src_;
    const Mat& lut_;
    Mat& dst_;

    int lutcn;

    uchar* lutBuffer;
    uchar* lutTable[4];

    IppLUTParallelBody_LUTCN(const Mat& src, const Mat& lut, Mat& dst, bool* _ok)
        : ok(_ok), src_(src), lut_(lut), dst_(dst), lutBuffer(NULL)
    {
        lutcn = lut.channels();
        IppiSize sz256 = {256, 1};

        size_t elemSize1 = dst.elemSize1();
        CV_DbgAssert(elemSize1 == 1);
        lutBuffer = (uchar*)CV_IPP_MALLOC(256 * (int)elemSize1 * 4);
        lutTable[0] = lutBuffer + 0;
        lutTable[1] = lutBuffer + 1 * 256 * elemSize1;
        lutTable[2] = lutBuffer + 2 * 256 * elemSize1;
        lutTable[3] = lutBuffer + 3 * 256 * elemSize1;

        CV_DbgAssert(lutcn == 3 || lutcn == 4);
        if (lutcn == 3)
        {
            IppStatus status = CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C3P3R, lut.ptr(), (int)lut.step[0], lutTable, (int)lut.step[0], sz256);
            if (status < 0)
                return;
        }
        else if (lutcn == 4)
        {
            IppStatus status = CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C4P4R, lut.ptr(), (int)lut.step[0], lutTable, (int)lut.step[0], sz256);
            if (status < 0)
                return;
        }

        *ok = true;
    }

    ~IppLUTParallelBody_LUTCN()
    {
        if (lutBuffer != NULL)
            ippFree(lutBuffer);
        lutBuffer = NULL;
        lutTable[0] = NULL;
    }

    void operator()( const cv::Range& range ) const
    {
        if (!*ok)
            return;

        const int row0 = range.start;
        const int row1 = range.end;

        Mat src = src_.rowRange(row0, row1);
        Mat dst = dst_.rowRange(row0, row1);

        if (lutcn == 3)
        {
            if (CV_INSTRUMENT_FUN_IPP(ippiLUTPalette_8u_C3R, src.ptr(), (int)src.step[0], dst.ptr(), (int)dst.step[0], ippiSize(dst.size()), lutTable, 8) >= 0)
                return;
        }
        else if (lutcn == 4)
        {
            if (CV_INSTRUMENT_FUN_IPP(ippiLUTPalette_8u_C4R, src.ptr(), (int)src.step[0], dst.ptr(), (int)dst.step[0], ippiSize(dst.size()), lutTable, 8) >= 0)
                return;
        }
        *ok = false;
    }
private:
    IppLUTParallelBody_LUTCN(const IppLUTParallelBody_LUTCN&);
    IppLUTParallelBody_LUTCN& operator=(const IppLUTParallelBody_LUTCN&);
};
} // namespace ipp

static bool ipp_lut(Mat &src, Mat &lut, Mat &dst)
{
    CV_INSTRUMENT_REGION_IPP()

    int lutcn = lut.channels();

    if(src.dims > 2)
        return false;

    bool ok = false;
    Ptr<ParallelLoopBody> body;

    size_t elemSize1 = CV_ELEM_SIZE1(dst.depth());

    if (lutcn == 1)
    {
        ParallelLoopBody* p = new ipp::IppLUTParallelBody_LUTC1(src, lut, dst, &ok);
        body.reset(p);
    }
    else if ((lutcn == 3 || lutcn == 4) && elemSize1 == 1)
    {
        ParallelLoopBody* p = new ipp::IppLUTParallelBody_LUTCN(src, lut, dst, &ok);
        body.reset(p);
    }

    if (body != NULL && ok)
    {
        Range all(0, dst.rows);
        if (dst.total()>>18)
            parallel_for_(all, *body, (double)std::max((size_t)1, dst.total()>>16));
        else
            (*body)(all);
        if (ok)
            return true;
    }

    return false;
}

#endif
#endif // IPP

class LUTParallelBody : public ParallelLoopBody
{
public:
    bool* ok;
    const Mat& src_;
    const Mat& lut_;
    Mat& dst_;

    LUTFunc func;

    LUTParallelBody(const Mat& src, const Mat& lut, Mat& dst, bool* _ok)
        : ok(_ok), src_(src), lut_(lut), dst_(dst)
    {
        func = lutTab[lut.depth()];
        *ok = (func != NULL);
    }

    void operator()( const cv::Range& range ) const
    {
        CV_DbgAssert(*ok);

        const int row0 = range.start;
        const int row1 = range.end;

        Mat src = src_.rowRange(row0, row1);
        Mat dst = dst_.rowRange(row0, row1);

        int cn = src.channels();
        int lutcn = lut_.channels();

        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        int len = (int)it.size;

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func(ptrs[0], lut_.ptr(), ptrs[1], len, cn, lutcn);
    }
private:
    LUTParallelBody(const LUTParallelBody&);
    LUTParallelBody& operator=(const LUTParallelBody&);
};

}

void cv::LUT( InputArray _src, InputArray _lut, OutputArray _dst )
{
    CV_INSTRUMENT_REGION()

    int cn = _src.channels(), depth = _src.depth();
    int lutcn = _lut.channels();

    CV_Assert( (lutcn == cn || lutcn == 1) &&
        _lut.total() == 256 && _lut.isContinuous() &&
        (depth == CV_8U || depth == CV_8S) );

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2,
               ocl_LUT(_src, _lut, _dst))

    Mat src = _src.getMat(), lut = _lut.getMat();
    _dst.create(src.dims, src.size, CV_MAKETYPE(_lut.depth(), cn));
    Mat dst = _dst.getMat();

    CV_OVX_RUN(!ovx::skipSmallImages<VX_KERNEL_TABLE_LOOKUP>(src.cols, src.rows),
               openvx_LUT(src, dst, lut))

#if !IPP_DISABLE_PERF_LUT
    CV_IPP_RUN(_src.dims() <= 2, ipp_lut(src, lut, dst));
#endif

    if (_src.dims() <= 2)
    {
        bool ok = false;
        Ptr<ParallelLoopBody> body;

        if (body == NULL || ok == false)
        {
            ok = false;
            ParallelLoopBody* p = new LUTParallelBody(src, lut, dst, &ok);
            body.reset(p);
        }
        if (body != NULL && ok)
        {
            Range all(0, dst.rows);
            if (dst.total()>>18)
                parallel_for_(all, *body, (double)std::max((size_t)1, dst.total()>>16));
            else
                (*body)(all);
            if (ok)
                return;
        }
    }

    LUTFunc func = lutTab[lut.depth()];
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &dst, 0};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], lut.ptr(), ptrs[1], len, cn, lutcn);
}

namespace cv {

#ifdef HAVE_OPENCL

static bool ocl_normalize( InputArray _src, InputOutputArray _dst, InputArray _mask, int dtype,
                           double scale, double delta )
{
    UMat src = _src.getUMat();

    if( _mask.empty() )
        src.convertTo( _dst, dtype, scale, delta );
    else if (src.channels() <= 4)
    {
        const ocl::Device & dev = ocl::Device::getDefault();

        int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype),
                ddepth = CV_MAT_DEPTH(dtype), wdepth = std::max(CV_32F, std::max(sdepth, ddepth)),
                rowsPerWI = dev.isIntel() ? 4 : 1;

        float fscale = static_cast<float>(scale), fdelta = static_cast<float>(delta);
        bool haveScale = std::fabs(scale - 1) > DBL_EPSILON,
                haveZeroScale = !(std::fabs(scale) > DBL_EPSILON),
                haveDelta = std::fabs(delta) > DBL_EPSILON,
                doubleSupport = dev.doubleFPConfig() > 0;

        if (!haveScale && !haveDelta && stype == dtype)
        {
            _src.copyTo(_dst, _mask);
            return true;
        }
        if (haveZeroScale)
        {
            _dst.setTo(Scalar(delta), _mask);
            return true;
        }

        if ((sdepth == CV_64F || ddepth == CV_64F) && !doubleSupport)
            return false;

        char cvt[2][40];
        String opts = format("-D srcT=%s -D dstT=%s -D convertToWT=%s -D cn=%d -D rowsPerWI=%d"
                             " -D convertToDT=%s -D workT=%s%s%s%s -D srcT1=%s -D dstT1=%s",
                             ocl::typeToStr(stype), ocl::typeToStr(dtype),
                             ocl::convertTypeStr(sdepth, wdepth, cn, cvt[0]), cn,
                             rowsPerWI, ocl::convertTypeStr(wdepth, ddepth, cn, cvt[1]),
                             ocl::typeToStr(CV_MAKE_TYPE(wdepth, cn)),
                             doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                             haveScale ? " -D HAVE_SCALE" : "",
                             haveDelta ? " -D HAVE_DELTA" : "",
                             ocl::typeToStr(sdepth), ocl::typeToStr(ddepth));

        ocl::Kernel k("normalizek", ocl::core::normalize_oclsrc, opts);
        if (k.empty())
            return false;

        UMat mask = _mask.getUMat(), dst = _dst.getUMat();

        ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
                maskarg = ocl::KernelArg::ReadOnlyNoSize(mask),
                dstarg = ocl::KernelArg::ReadWrite(dst);

        if (haveScale)
        {
            if (haveDelta)
                k.args(srcarg, maskarg, dstarg, fscale, fdelta);
            else
                k.args(srcarg, maskarg, dstarg, fscale);
        }
        else
        {
            if (haveDelta)
                k.args(srcarg, maskarg, dstarg, fdelta);
            else
                k.args(srcarg, maskarg, dstarg);
        }

        size_t globalsize[2] = { (size_t)src.cols, ((size_t)src.rows + rowsPerWI - 1) / rowsPerWI };
        return k.run(2, globalsize, NULL, false);
    }
    else
    {
        UMat temp;
        src.convertTo( temp, dtype, scale, delta );
        temp.copyTo( _dst, _mask );
    }

    return true;
}

#endif

}

void cv::normalize( InputArray _src, InputOutputArray _dst, double a, double b,
                    int norm_type, int rtype, InputArray _mask )
{
    CV_INSTRUMENT_REGION()

    double scale = 1, shift = 0;
    if( norm_type == CV_MINMAX )
    {
        double smin = 0, smax = 0;
        double dmin = MIN( a, b ), dmax = MAX( a, b );
        minMaxIdx( _src, &smin, &smax, 0, 0, _mask );
        scale = (dmax - dmin)*(smax - smin > DBL_EPSILON ? 1./(smax - smin) : 0);
        shift = dmin - smin*scale;
    }
    else if( norm_type == CV_L2 || norm_type == CV_L1 || norm_type == CV_C )
    {
        scale = norm( _src, norm_type, _mask );
        scale = scale > DBL_EPSILON ? a/scale : 0.;
        shift = 0;
    }
    else
        CV_Error( CV_StsBadArg, "Unknown/unsupported norm type" );

    int type = _src.type(), depth = CV_MAT_DEPTH(type);
    if( rtype < 0 )
        rtype = _dst.fixedType() ? _dst.depth() : depth;

    CV_OCL_RUN(_dst.isUMat(),
               ocl_normalize(_src, _dst, _mask, rtype, scale, shift))

    Mat src = _src.getMat();
    if( _mask.empty() )
        src.convertTo( _dst, rtype, scale, shift );
    else
    {
        Mat temp;
        src.convertTo( temp, rtype, scale, shift );
        temp.copyTo( _dst, _mask );
    }
}

CV_IMPL void
cvSplit( const void* srcarr, void* dstarr0, void* dstarr1, void* dstarr2, void* dstarr3 )
{
    void* dptrs[] = { dstarr0, dstarr1, dstarr2, dstarr3 };
    cv::Mat src = cv::cvarrToMat(srcarr);
    int i, j, nz = 0;
    for( i = 0; i < 4; i++ )
        nz += dptrs[i] != 0;
    CV_Assert( nz > 0 );
    std::vector<cv::Mat> dvec(nz);
    std::vector<int> pairs(nz*2);

    for( i = j = 0; i < 4; i++ )
    {
        if( dptrs[i] != 0 )
        {
            dvec[j] = cv::cvarrToMat(dptrs[i]);
            CV_Assert( dvec[j].size() == src.size() );
            CV_Assert( dvec[j].depth() == src.depth() );
            CV_Assert( dvec[j].channels() == 1 );
            CV_Assert( i < src.channels() );
            pairs[j*2] = i;
            pairs[j*2+1] = j;
            j++;
        }
    }
    if( nz == src.channels() )
        cv::split( src, dvec );
    else
    {
        cv::mixChannels( &src, 1, &dvec[0], nz, &pairs[0], nz );
    }
}


CV_IMPL void
cvMerge( const void* srcarr0, const void* srcarr1, const void* srcarr2,
         const void* srcarr3, void* dstarr )
{
    const void* sptrs[] = { srcarr0, srcarr1, srcarr2, srcarr3 };
    cv::Mat dst = cv::cvarrToMat(dstarr);
    int i, j, nz = 0;
    for( i = 0; i < 4; i++ )
        nz += sptrs[i] != 0;
    CV_Assert( nz > 0 );
    std::vector<cv::Mat> svec(nz);
    std::vector<int> pairs(nz*2);

    for( i = j = 0; i < 4; i++ )
    {
        if( sptrs[i] != 0 )
        {
            svec[j] = cv::cvarrToMat(sptrs[i]);
            CV_Assert( svec[j].size == dst.size &&
                svec[j].depth() == dst.depth() &&
                svec[j].channels() == 1 && i < dst.channels() );
            pairs[j*2] = j;
            pairs[j*2+1] = i;
            j++;
        }
    }

    if( nz == dst.channels() )
        cv::merge( svec, dst );
    else
    {
        cv::mixChannels( &svec[0], nz, &dst, 1, &pairs[0], nz );
    }
}


CV_IMPL void
cvMixChannels( const CvArr** src, int src_count,
               CvArr** dst, int dst_count,
               const int* from_to, int pair_count )
{
    cv::AutoBuffer<cv::Mat> buf(src_count + dst_count);

    int i;
    for( i = 0; i < src_count; i++ )
        buf[i] = cv::cvarrToMat(src[i]);
    for( i = 0; i < dst_count; i++ )
        buf[i+src_count] = cv::cvarrToMat(dst[i]);
    cv::mixChannels(&buf[0], src_count, &buf[src_count], dst_count, from_to, pair_count);
}

CV_IMPL void
cvConvertScaleAbs( const void* srcarr, void* dstarr,
                   double scale, double shift )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.size == dst.size && dst.type() == CV_8UC(src.channels()));
    cv::convertScaleAbs( src, dst, scale, shift );
}

CV_IMPL void
cvConvertScale( const void* srcarr, void* dstarr,
                double scale, double shift )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size == dst.size && src.channels() == dst.channels() );
    src.convertTo(dst, dst.type(), scale, shift);
}

CV_IMPL void cvLUT( const void* srcarr, void* dstarr, const void* lutarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), lut = cv::cvarrToMat(lutarr);

    CV_Assert( dst.size() == src.size() && dst.type() == CV_MAKETYPE(lut.depth(), src.channels()) );
    cv::LUT( src, lut, dst );
}

CV_IMPL void cvNormalize( const CvArr* srcarr, CvArr* dstarr,
                          double a, double b, int norm_type, const CvArr* maskarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    CV_Assert( dst.size() == src.size() && src.channels() == dst.channels() );
    cv::normalize( src, dst, a, b, norm_type, dst.type(), mask );
}

/* End of file. */
