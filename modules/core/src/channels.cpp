// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "opencl_kernels_core.hpp"
#include "convert.hpp"
#include "channel_permutation.simd.hpp"
#include "channel_permutation.simd_declarations.hpp"

#include <bitset>
#include <functional>

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


static void mixChannels8u( const void** src, const int* sdelta,
                           void** dst, const int* ddelta,
                           int len, int npairs )
{
    mixChannels_((const uchar**)src, sdelta, (uchar**)dst, ddelta, len, npairs);
}

static void mixChannels16u( const void** src, const int* sdelta,
                            void** dst, const int* ddelta,
                            int len, int npairs )
{
    mixChannels_((const ushort**)src, sdelta, (ushort**)dst, ddelta, len, npairs);
}

static void mixChannels32s( const void** src, const int* sdelta,
                            void** dst, const int* ddelta,
                            int len, int npairs )
{
    mixChannels_((const int**)src, sdelta, (int**)dst, ddelta, len, npairs);
}

static void mixChannels64s( const void** src, const int* sdelta,
                            void** dst, const int* ddelta,
                            int len, int npairs )
{
    mixChannels_((const int64**)src, sdelta, (int64**)dst, ddelta, len, npairs);
}

typedef void (*MixChannelsFunc)( const void** src, const int* sdelta,
        void** dst, const int* ddelta, int len, int npairs );

static MixChannelsFunc getMixchFunc(int depth)
{
    static MixChannelsFunc mixchTab[CV_DEPTH_MAX] =
    {
        mixChannels8u, mixChannels8u, mixChannels16u,
        mixChannels16u, mixChannels32s, mixChannels32s,
        mixChannels64s, mixChannels16u, mixChannels16u,
        mixChannels8u, mixChannels64s, mixChannels64s,
        mixChannels32s, mixChannels8u, mixChannels8u
    };

    return mixchTab[depth];
}

template<typename T>
static void copyPermutedPixels(const T* src, T* dst, int pixels, const int* order, int channels)
{
    for (int pixel = 0; pixel < pixels; ++pixel, src += channels, dst += channels)
        for (int channel = 0; channel < channels; ++channel)
            dst[channel] = src[order[channel]];
}

#if CV_SIMD || CV_SIMD_SCALABLE
template<typename T, typename VecT, int channels, bool swapRB = false>
static void permuteChannels(NAryMatIterator& it, const int* order)
{
    const int lanes = VTraits<VecT>::vlanes();
    const int pixels = static_cast<int>(it.size);
    VecT a, b, c, d;
    const VecT* vectors[] = {&a, &b, &c, &d};
    const VecT& first = swapRB ? c : *vectors[order[0]];
    const VecT& second = swapRB ? b : *vectors[order[1]];
    const VecT& third = swapRB ? a : *vectors[channels == 2 ? 0 : order[2]];
    const VecT& fourth = swapRB ? d : *vectors[channels == 4 ? order[3] : 0];

    for (size_t plane = 0; plane < it.nplanes; ++plane, ++it)
    {
        const T* src = reinterpret_cast<const T*>(it.ptrs[0]);
        T* dst = reinterpret_cast<T*>(it.ptrs[1]);
        int pixel = 0;
        for (; pixel <= pixels - lanes; pixel += lanes, src += lanes * channels, dst += lanes * channels)
        {
            if constexpr (channels == 2)
            {
                v_load_deinterleave(src, a, b);
                v_store_interleave(dst, b, a);
            }
            else if constexpr (channels == 3)
            {
                v_load_deinterleave(src, a, b, c);
                v_store_interleave(dst, first, second, third);
            }
            else
            {
                v_load_deinterleave(src, a, b, c, d);
                v_store_interleave(dst, first, second, third, fourth);
            }
        }
        copyPermutedPixels(src, dst, pixels - pixel, order, channels);
    }
    vx_cleanup();
}

template<typename T, typename VecT>
static void permuteChannels(NAryMatIterator& it, const int* order, int channels)
{
    const bool swapRB = order[0] == 2 && order[1] == 1;
    if (channels == 2)
        permuteChannels<T, VecT, 2>(it, order);
    else if (channels == 3)
    {
        if (swapRB)
            permuteChannels<T, VecT, 3, true>(it, order);
        else
            permuteChannels<T, VecT, 3>(it, order);
    }
    else if (swapRB && order[3] == 3)
        permuteChannels<T, VecT, 4, true>(it, order);
    else
        permuteChannels<T, VecT, 4>(it, order);
}
#endif

static bool tryPermuteChannels(const Mat& src, const Mat& dst, const int* fromTo, size_t npairs,
                               NAryMatIterator& it)
{
    const int channels = src.channels();
    if (npairs != static_cast<size_t>(channels) || src.type() != dst.type() || src.empty() || dst.empty())
        return false;

    const size_t elemSize = src.elemSize1();
    const size_t pixels = it.size * it.nplanes;
    const bool bytes = channels >= 5 && channels <= 16 && elemSize == 1 && it.size >= 16 && pixels >= 1024;
    const bool wideFloat = channels >= 32 && src.depth() == CV_32F && pixels >= 65536;
#if CV_SIMD || CV_SIMD_SCALABLE
    const bool small = channels >= 2 && channels <= 4 && elemSize <= 4;
#else
    const bool small = false;
#endif
    if (!(small || bytes || wideFloat))
        return false;

    // Preserve the existing write order for overlapping matrices.
    const size_t planeBytes = it.size * src.elemSize();
    const size_t totalBytes = it.nplanes * planeBytes;
    const uchar* srcStart = src.isContinuous() ? src.data : src.datastart;
    const uchar* srcEnd = src.isContinuous() ? src.data + totalBytes : src.dataend;
    const uchar* dstStart = dst.isContinuous() ? dst.data : dst.datastart;
    const uchar* dstEnd = dst.isContinuous() ? dst.data + totalBytes : dst.dataend;
    std::less<const uchar*> less;
    if (less(srcStart, dstEnd) && less(dstStart, srcEnd))
        return false;

    int order[CV_CN_MAX];
    std::fill_n(order, channels, -1);
    std::bitset<CV_CN_MAX> used;
    bool identity = true;
    for (size_t pair = 0; pair < npairs; ++pair)
    {
        const int from = fromTo[2 * pair];
        const int to = fromTo[2 * pair + 1];
        if (from < 0 || order[to] >= 0 || used[from])
            return false;
        order[to] = from;
        used[from] = true;
        identity &= from == to;
    }

    if (identity)
    {
        for (size_t plane = 0; plane < it.nplanes; ++plane, ++it)
            std::memcpy(it.ptrs[1], it.ptrs[0], planeBytes);
        return true;
    }
    if (bytes)
    {
        CV_CPU_DISPATCH(permuteByteChannels, (it, order, channels), CV_CPU_DISPATCH_MODES_ALL);
    }
    if (wideFloat)
    {
        for (size_t plane = 0; plane < it.nplanes; ++plane, ++it)
            copyPermutedPixels(reinterpret_cast<const unsigned*>(it.ptrs[0]), reinterpret_cast<unsigned*>(it.ptrs[1]),
                               static_cast<int>(it.size), order, channels);
        return true;
    }

#if CV_SIMD || CV_SIMD_SCALABLE
    switch (elemSize)
    {
    case 1:
        permuteChannels<uchar, v_uint8>(it, order, channels);
        return true;
    case 2:
        permuteChannels<ushort, v_uint16>(it, order, channels);
        return true;
    case 4:
        permuteChannels<unsigned, v_uint32>(it, order, channels);
        return true;
    }
#endif
    return false;
}

} // cv::


void cv::mixChannels( const Mat* src, size_t nsrcs, Mat* dst, size_t ndsts, const int* fromTo, size_t npairs )
{
    CV_INSTRUMENT_REGION();

    if( npairs == 0 )
        return;
    CV_Assert( src && nsrcs > 0 && dst && ndsts > 0 && fromTo && npairs > 0 );

    size_t i, j, k, esz1 = dst[0].elemSize1();
    int depth = dst[0].depth();

    AutoBuffer<uchar> buf((nsrcs + ndsts + 1)*(sizeof(Mat*) + sizeof(uchar*)) + npairs*(sizeof(uchar*)*2 + sizeof(int)*6));
    const Mat** arrays = (const Mat**)(uchar*)buf.data();
    uchar** ptrs = (uchar**)(arrays + nsrcs + ndsts);
    const uchar** srcs = (const uchar**)(ptrs + nsrcs + ndsts + 1);
    uchar** dsts = (uchar**)(srcs + npairs);
    int* tab = (int*)(dsts + npairs);
    int *sdelta = (int*)(tab + npairs*4), *ddelta = sdelta + npairs;

    for( i = 0; i < nsrcs; i++ )
    {
        CV_Assert( !src[i].empty() && src[i].size == dst[0].size && src[i].depth() == depth );
        arrays[i] = &src[i];
    }
    for( i = 0; i < ndsts; i++ )
    {
        CV_Assert( !dst[i].empty() && dst[i].size == dst[0].size && dst[i].depth() == depth );
        arrays[i + nsrcs] = &dst[i];
    }
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
    CV_Assert(func);

    if (nsrcs == 1 && ndsts == 1 && tryPermuteChannels(src[0], dst[0], fromTo, npairs, it))
        return;

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
            func( (const void**)srcs, sdelta, (void **)dsts, ddelta, bsz, (int)npairs );

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

        declsrc += format("DECLARE_INPUT_MAT(%zu)", i);
        decldst += format("DECLARE_OUTPUT_MAT(%zu)", i);
        indexdecl += format("DECLARE_INDEX(%zu)", i);
        declproc += format("PROCESS_ELEM(%zu)", i);
        declcn += format(" -D scn%zu=%d -D dcn%zu=%d", i, src[src_idx].channels(), i, dst[dst_idx].channels());
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
    CV_INSTRUMENT_REGION();

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
    Mat* buf = _buf.data();
    for( i = 0; i < nsrc; i++ )
        buf[i] = src.getMat(src_is_mat ? -1 : i);
    for( i = 0; i < ndst; i++ )
        buf[nsrc + i] = dst.getMat(dst_is_mat ? -1 : i);
    mixChannels(&buf[0], nsrc, &buf[nsrc], ndst, fromTo, npairs);
}

void cv::mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
                     const std::vector<int>& fromTo)
{
    CV_INSTRUMENT_REGION();

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
    Mat* buf = _buf.data();
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
#ifdef HAVE_IPP_IW_LL
    CV_INSTRUMENT_REGION_IPP();

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
#ifdef HAVE_IPP_IW_LL
    CV_INSTRUMENT_REGION_IPP();

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
    CV_INSTRUMENT_REGION();

    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert( 0 <= coi && coi < cn );
    int ch[] = { coi, 0 };

#ifdef HAVE_OPENCL
    if (ocl::isOpenCLActivated() && _src.dims() <= 2 && _dst.isUMat())
    {
        UMat src = _src.getUMat();
        // NB: fetch the source before (re)allocating the destination so that
        // in-place operation (_src and _dst referencing the same array) keeps
        // the multi-channel source alive while dst is reshaped to 1 channel.
        _dst.createSameSize(_src, depth);
        UMat dst = _dst.getUMat();
        mixChannels(std::vector<UMat>(1, src), std::vector<UMat>(1, dst), ch, 1);
        return;
    }
#endif

    Mat src = _src.getMat();
    _dst.createSameSize(_src, depth);
    Mat dst = _dst.getMat();

    CV_IPP_RUN_FAST(ipp_extractChannel(src, dst, coi))

    mixChannels(&src, 1, &dst, 1, ch, 1);
}

void cv::insertChannel(InputArray _src, InputOutputArray _dst, int coi)
{
    CV_INSTRUMENT_REGION();

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
