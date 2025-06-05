#ifndef __PRECOMP_IPP_HPP__
#define __PRECOMP_IPP_HPP__

#include <opencv2/imgproc.hpp>

#ifdef HAVE_IPP_IW
#include "iw++/iw.hpp"
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

#ifdef HAVE_IPP_IW
static inline int ippiSuggestThreadsNum(const ::ipp::IwiImage &image, double multiplier)
{
    return ippiSuggestThreadsNum(image.m_size.width, image.m_size.height, image.m_typeSize*image.m_channels, multiplier);
}
#endif

#endif //__PRECOMP_IPP_HPP__
