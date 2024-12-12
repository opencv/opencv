// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_VIDEOIO_UTILS_PRIVATE_HPP
#define OPENCV_VIDEOIO_UTILS_PRIVATE_HPP

#include "opencv2/core/cvdef.h"
#include <string>

namespace cv {
CV_EXPORTS std::string icvExtractPattern(const std::string& filename, unsigned *offset);

class CvStream : public std::streambuf
{
public:
    CvStream(void* _opaque = nullptr,
             long long(*_read)(void* opaque, char* buffer, long long size) = nullptr,
             long long(*_seek)(void* opaque, long long offset, int way) = nullptr)
    {
        opaque = _opaque;
        read = _read;
        seek = _seek;
    }

    std::streamsize xsgetn(char* s, std::streamsize n) override
    {
        return read(opaque, s, (int)n);
    }

    std::streampos seekoff(std::streamoff off, std::ios_base::seekdir way, std::ios_base::openmode = std::ios_base::in | std::ios_base::out) override
    {
        return seek(opaque, off, way == std::ios_base::beg ? SEEK_SET : (way == std::ios_base::end ? SEEK_END : SEEK_CUR));
    }

    // Required for sgetc (check for end-of-stream)
    int underflow() override
    {
        char s;
        if (xsgetn(&s, 1) == 1)
        {
            seekoff(-1, std::ios_base::cur);
            return static_cast<int>(s);
        }
        else
            return EOF;
    }
private:
    void* opaque;
    long long(*read)(void* opaque, char* buffer, long long size);
    long long(*seek)(void* opaque, long long offset, int way);
};
}

#endif // OPENCV_VIDEOIO_UTILS_PRIVATE_HPP
