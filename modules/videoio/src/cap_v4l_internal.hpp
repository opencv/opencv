// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_VIDEOIO_CAP_V4L_INTERNAL_HPP
#define OPENCV_VIDEOIO_CAP_V4L_INTERNAL_HPP

#include <cstddef>
#include <cstring>
#include <limits>

namespace cv { namespace v4l2_internal {

inline std::size_t minimumPayloadSize(unsigned int pixelFormat, std::size_t width, std::size_t height)
{
    if (width == 0 || height == 0 || width > std::numeric_limits<std::size_t>::max() / height)
        return 0;

    const std::size_t pixels = width * height;
    std::size_t bytesPerPixel = 0;

    switch (pixelFormat)
    {
    case V4L2_PIX_FMT_YVU420:
    case V4L2_PIX_FMT_YUV420:
    case V4L2_PIX_FMT_NV12:
    case V4L2_PIX_FMT_NV21:
    {
        const std::size_t chromaWidth = width / 2 + width % 2;
        const std::size_t chromaHeight = height / 2 + height % 2;
        if (chromaWidth > std::numeric_limits<std::size_t>::max() / chromaHeight)
            return 0;
        const std::size_t chromaPlaneSize = chromaWidth * chromaHeight;
        if (chromaPlaneSize > (std::numeric_limits<std::size_t>::max() - pixels) / 2)
            return 0;
        return pixels + 2 * chromaPlaneSize;
    }
    case V4L2_PIX_FMT_YUV411P:
    {
        const std::size_t chromaWidth = width / 4 + (width % 4 != 0);
        if (chromaWidth > std::numeric_limits<std::size_t>::max() / height)
            return 0;
        const std::size_t chromaPlaneSize = chromaWidth * height;
        if (chromaPlaneSize > (std::numeric_limits<std::size_t>::max() - pixels) / 2)
            return 0;
        return pixels + 2 * chromaPlaneSize;
    }
    case V4L2_PIX_FMT_YUYV:
    case V4L2_PIX_FMT_UYVY:
    case V4L2_PIX_FMT_Y16:
    case V4L2_PIX_FMT_Y16_BE:
    case V4L2_PIX_FMT_Y10:
    case V4L2_PIX_FMT_Y12:
        bytesPerPixel = 2;
        break;
    case V4L2_PIX_FMT_RGB24:
    case V4L2_PIX_FMT_BGR24:
        bytesPerPixel = 3;
        break;
    case V4L2_PIX_FMT_XBGR32:
    case V4L2_PIX_FMT_ABGR32:
        bytesPerPixel = 4;
        break;
    case V4L2_PIX_FMT_SRGGB8:
    case V4L2_PIX_FMT_SBGGR8:
    case V4L2_PIX_FMT_SGBRG8:
    case V4L2_PIX_FMT_SGRBG8:
    case V4L2_PIX_FMT_GREY:
        bytesPerPixel = 1;
        break;
    // These formats have variable payload lengths.
    case V4L2_PIX_FMT_MJPEG:
    case V4L2_PIX_FMT_JPEG:
    case V4L2_PIX_FMT_SN9C10X:
    default:
        return 0;
    }

    if (pixels > std::numeric_limits<std::size_t>::max() / bytesPerPixel)
        return 0;
    return pixels * bytesPerPixel;
}

inline bool getPlanePayloadSize(std::size_t bytesUsed, std::size_t dataOffset,
                                std::size_t planeLength, std::size_t mappedLength,
                                std::size_t& payloadSize, std::size_t fallbackPayloadSize = 0)
{
    if (planeLength != mappedLength || dataOffset > planeLength)
        return false;
    // Some capture drivers do not report bytesused for fixed-size formats.
    // Infer only a known image size, never the unused capacity of the mapping.
    if (bytesUsed == 0 && fallbackPayloadSize > 0)
    {
        if (fallbackPayloadSize > planeLength - dataOffset)
            return false;
        payloadSize = fallbackPayloadSize;
        return true;
    }
    if (bytesUsed < dataOffset || bytesUsed > planeLength)
        return false;
    payloadSize = bytesUsed - dataOffset;
    return true;
}

inline bool copyPlanePayload(void* destination, std::size_t destinationLength,
                             std::size_t& destinationOffset, const void* source,
                             std::size_t planeLength, std::size_t mappedLength,
                             std::size_t bytesUsed, std::size_t dataOffset)
{
    std::size_t planePayloadSize = 0;
    if (!destination || !source ||
        !getPlanePayloadSize(bytesUsed, dataOffset, planeLength, mappedLength, planePayloadSize) ||
        destinationOffset > destinationLength ||
        planePayloadSize > destinationLength - destinationOffset)
        return false;

    std::memcpy(static_cast<unsigned char*>(destination) + destinationOffset,
                static_cast<const unsigned char*>(source) + dataOffset, planePayloadSize);
    destinationOffset += planePayloadSize;
    return true;
}

inline bool isFrameValid(bool errorFlagSet, std::size_t payloadSize, std::size_t minimumPayloadSize)
{
    return !errorFlagSet && payloadSize > 0 && payloadSize >= minimumPayloadSize;
}

}} // namespace cv::v4l2_internal

#endif // OPENCV_VIDEOIO_CAP_V4L_INTERNAL_HPP
