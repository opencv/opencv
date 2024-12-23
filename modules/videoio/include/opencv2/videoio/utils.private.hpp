// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_VIDEOIO_UTILS_PRIVATE_HPP
#define OPENCV_VIDEOIO_UTILS_PRIVATE_HPP

#include "opencv2/core/cvdef.h"
#include <string>

namespace cv {
CV_EXPORTS std::string icvExtractPattern(const std::string& filename, unsigned *offset);

class IReadStream
{
public:
    virtual ~IReadStream() {};
    virtual long long read(char* buffer, long long size) = 0;
    virtual long long seek(long long offset, int way) = 0;
};

class StreambufReadStream : public IReadStream
{
public:
    StreambufReadStream(Ptr<std::streambuf> _stream) : stream(_stream) {}

    virtual ~StreambufReadStream() {}

    long long read(char* buffer, long long size) override
    {
        return stream->sgetn(buffer, size);
    }

    long long seek(long long offset, int way) override
    {
        return stream->pubseekoff(offset, way == SEEK_SET ? std::ios_base::beg : (way == SEEK_END ? std::ios_base::end : std::ios_base::cur));
    }

    static Ptr<IReadStream> create(Ptr<std::streambuf> stream)
    {
        return Ptr<IReadStream>(new StreambufReadStream(stream));
    }

private:
    Ptr<std::streambuf> stream;
};


class ReadStreamPluginProvider : public IReadStream
{
public:
    ReadStreamPluginProvider(void* _opaque,
                             long long (*_read)(void* opaque, char* buffer, long long size),
                             long long (*_seek)(void* opaque, long long offset, int way))
    {
        opaque = _opaque;
        readCallback = _read;
        seekCallback = _seek;
    }

    virtual ~ReadStreamPluginProvider() {}

    long long read(char* buffer, long long size) override
    {
        return readCallback(opaque, buffer, size);
    }

    long long seek(long long offset, int way) override
    {
        return seekCallback(opaque, offset, way);
    }

private:
    void* opaque;
    long long (*readCallback)(void* opaque, char* buffer, long long size);
    long long (*seekCallback)(void* opaque, long long offset, int way);
};

}

#endif // OPENCV_VIDEOIO_UTILS_PRIVATE_HPP
