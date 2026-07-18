// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __PRECOMP_IPP_HPP__
#define __PRECOMP_IPP_HPP__

#include <opencv2/imgproc.hpp>

#ifdef HAVE_IPP_IW
#include "iw++/iw.hpp"
#endif

static inline IppiSize ippiSize(size_t width, size_t height)
{
    IppiSize size = { (int)width, (int)height };
    return size;
}

static inline IppiSize ippiSize(const cv::Size & _size)
{
    IppiSize size = { _size.width, _size.height };
    return size;
}

#if IPP_VERSION_X100 >= 201700
static inline IppiSizeL ippiSizeL(size_t width, size_t height)
{
    IppiSizeL size = { (IppSizeL)width, (IppSizeL)height };
    return size;
}

static inline IppiSizeL ippiSizeL(const cv::Size & _size)
{
    IppiSizeL size = { _size.width, _size.height };
    return size;
}
#endif

static inline IppDataType ippiGetDataType(int depth)
{
    depth = CV_MAT_DEPTH(depth);
    return depth == CV_8U ? ipp8u :
    depth == CV_8S ? ipp8s :
    depth == CV_16U ? ipp16u :
    depth == CV_16S ? ipp16s :
    depth == CV_32S ? ipp32s :
    depth == CV_32F ? ipp32f :
    depth == CV_64F ? ipp64f :
    (IppDataType)-1;
}

static inline IppiInterpolationType ippiGetInterpolation(int inter)
{
    inter &= cv::InterpolationFlags::INTER_MAX;
    return inter == cv::InterpolationFlags::INTER_NEAREST ? ippNearest :
        inter == cv::InterpolationFlags::INTER_LINEAR ? ippLinear :
        inter == cv::InterpolationFlags::INTER_CUBIC ? ippCubic :
        inter == cv::InterpolationFlags::INTER_LANCZOS4 ? ippLanczos :
        inter == cv::InterpolationFlags::INTER_AREA ? ippSuper :
        (IppiInterpolationType)-1;
}

static inline IppiBorderType ippiGetBorderType(int borderTypeNI)
{
    return borderTypeNI == cv::BorderTypes::BORDER_CONSTANT    ? ippBorderConst   :
           borderTypeNI == cv::BorderTypes::BORDER_TRANSPARENT ? ippBorderTransp  :
           borderTypeNI == cv::BorderTypes::BORDER_REPLICATE   ? ippBorderRepl    :
           (IppiBorderType)-1;
}

static inline int ippiSuggestThreadsNum(size_t width, size_t height, size_t elemSize, double multiplier)
{
    int threads = cv::getNumThreads();
    if(threads > 1 && height >= 64)
    {
        size_t opMemory = (int)(width*height*elemSize*multiplier);
        int l2cache = 0;
#if IPP_VERSION_X100 >= 201700
        ippGetL2CacheSize(&l2cache);
#endif
        if(!l2cache)
            l2cache = 1 << 18;

        return IPP_MAX(1, (IPP_MIN((int)(opMemory/l2cache), threads)));
    }
    return 1;
}

static inline int ippiSuggestRowThreadsNum(size_t width, size_t height, size_t elemSize, size_t payloadSize)
{
    int num_threads = cv::getNumThreads();
    if(num_threads > 1)
    {
        long rowThreads = static_cast<long>(height);

        // row-based range shall not allow to split rows
        num_threads = (rowThreads < num_threads) ? rowThreads : num_threads;
        long rows_per_thread = (rowThreads + num_threads - 1) / num_threads;
        size_t item_size = width * elemSize; // row size in bytes

        if(static_cast<size_t>(item_size * rows_per_thread) < payloadSize)
        {
            long items_per_thread = IPP_MAX(1L, static_cast<long>(payloadSize / item_size ));
            num_threads = static_cast<int>((height + items_per_thread - 1L) / items_per_thread);
        }
    }
    return num_threads;
}

#ifdef HAVE_IPP_IW
static inline int ippiSuggestThreadsNum(const ::ipp::IwiImage &image, double multiplier)
{
    return ippiSuggestThreadsNum(image.m_size.width, image.m_size.height, image.m_typeSize*image.m_channels, multiplier);
}

static inline int ippiSuggestRowThreadsNum(const ::ipp::IwiImage &image, size_t payloadSize)
{
    return ippiSuggestRowThreadsNum(image.m_size.width, image.m_size.height, image.m_typeSize*image.m_channels, payloadSize);
}
#endif

#if IPP_VERSION_X100 >= 201700
#define CV_IPP_MALLOC(SIZE) ippMalloc_L(SIZE)
#else
#define CV_IPP_MALLOC(SIZE) ippMalloc((int)SIZE)
#endif

template<typename T>
class IppAutoBuffer
{
public:
    IppAutoBuffer() { m_size = 0; m_pBuffer = NULL; }
    explicit IppAutoBuffer(size_t size) { m_size = 0; m_pBuffer = NULL; allocate(size); }
    ~IppAutoBuffer() { deallocate(); }
    T* allocate(size_t size)   { if(m_size < size) { deallocate(); m_pBuffer = (T*)CV_IPP_MALLOC(size); m_size = size; } return m_pBuffer; }
    void deallocate() { if(m_pBuffer) { ippFree(m_pBuffer); m_pBuffer = NULL; } m_size = 0; }
    inline T* get() { return (T*)m_pBuffer;}
    inline operator T* () { return (T*)m_pBuffer;}
    inline operator const T* () const { return (const T*)m_pBuffer;}
private:
    IppAutoBuffer(IppAutoBuffer &) {}
    IppAutoBuffer& operator =(const IppAutoBuffer &) {return *this;}

    size_t m_size;
    T*     m_pBuffer;
};

#endif //__PRECOMP_IPP_HPP__
