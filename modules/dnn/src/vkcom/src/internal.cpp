// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "internal.hpp"

namespace cv { namespace dnn { namespace vkcom {
#ifdef HAVE_VULKAN

bool checkFormat(Format fmt)
{
    return (fmt > -1 && fmt < kFormatNum) ? true : false;
}

size_t elementSize(Format fmt)
{
    if (fmt == kFormatFp32 || fmt == kFormatInt32)
    {
        return 4;
    }
    else if (fmt >= 0 && fmt < kFormatNum)
    {
        CV_LOG_WARNING(NULL, format("Unsupported format %d", fmt));
    }
    else
    {
        CV_Error(Error::StsError, format("Invalid format %d", fmt));
    }
    return 0;
}

int shapeCount(const Shape& shape, int start, int end)
{
    if (start == -1) start = 0;
    if (end == -1) end = (int)shape.size();

    if (shape.empty())
        return 0;

    int elems = 1;
    assert(start <= (int)shape.size() &&
           end <= (int)shape.size() &&
           start <= end);
    for(int i = start; i < end; i++)
    {
        elems *= shape[i];
    }
    return elems;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom