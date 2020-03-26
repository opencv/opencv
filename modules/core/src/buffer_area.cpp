// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/core/utils/buffer_area.private.hpp"
#include "opencv2/core/utils/configuration.private.hpp"

#ifndef OPENCV_ENABLE_MEMORY_SANITIZER
static bool CV_BUFFER_AREA_OVERRIDE_SAFE_MODE =
    cv::utils::getConfigurationParameterBool("OPENCV_BUFFER_AREA_ALWAYS_SAFE", false);
#endif

namespace cv { namespace utils {

//==================================================================================================

class BufferArea::Block
{
private:
    inline size_t reserve_count() const
    {
        return alignment / type_size - 1;
    }
public:
    Block(void **ptr_, ushort type_size_, size_t count_, ushort alignment_)
        : ptr(ptr_), raw_mem(0), count(count_), type_size(type_size_), alignment(alignment_)
    {
        CV_Assert(ptr && *ptr == NULL);
    }
    void cleanup() const
    {
        CV_Assert(ptr && *ptr);
        *ptr = 0;
        if (raw_mem)
            fastFree(raw_mem);
    }
    size_t getByteCount() const
    {
        return type_size * (count + reserve_count());
    }
    void real_allocate()
    {
        CV_Assert(ptr && *ptr == NULL);
        const size_t allocated_count = count + reserve_count();
        raw_mem = fastMalloc(type_size * allocated_count);
        if (alignment != type_size)
        {
            *ptr = alignPtr(raw_mem, alignment);
            CV_Assert(reinterpret_cast<size_t>(*ptr) % alignment == 0);
            CV_Assert(static_cast<uchar*>(*ptr) + type_size * count <= static_cast<uchar*>(raw_mem) + type_size * allocated_count);
        }
        else
        {
            *ptr = raw_mem;
        }
    }
#ifndef OPENCV_ENABLE_MEMORY_SANITIZER
    void * fast_allocate(void * buf) const
    {
        CV_Assert(ptr && *ptr == NULL);
        buf = alignPtr(buf, alignment);
        CV_Assert(reinterpret_cast<size_t>(buf) % alignment == 0);
        *ptr = buf;
        return static_cast<void*>(static_cast<uchar*>(*ptr) + type_size * count);
    }
#endif
    bool operator==(void **other) const
    {
        CV_Assert(ptr && other);
        return *ptr == *other;
    }
    void zeroFill() const
    {
        CV_Assert(ptr && *ptr);
        memset(static_cast<uchar*>(*ptr), 0, count * type_size);
    }
private:
    void **ptr;
    void * raw_mem;
    size_t count;
    ushort type_size;
    ushort alignment;
};

//==================================================================================================

#ifndef OPENCV_ENABLE_MEMORY_SANITIZER
BufferArea::BufferArea(bool safe_) :
    oneBuf(0),
    totalSize(0),
    safe(safe_ || CV_BUFFER_AREA_OVERRIDE_SAFE_MODE)
{
    // nothing
}
#else
BufferArea::BufferArea(bool safe_)
{
    CV_UNUSED(safe_);
}
#endif

BufferArea::~BufferArea()
{
    release();
}

void BufferArea::allocate_(void **ptr, ushort type_size, size_t count, ushort alignment)
{
    blocks.push_back(Block(ptr, type_size, count, alignment));
#ifndef OPENCV_ENABLE_MEMORY_SANITIZER
    if (!safe)
    {
        totalSize += blocks.back().getByteCount();
    }
    else
#endif
    {
        blocks.back().real_allocate();
    }
}

void BufferArea::zeroFill_(void **ptr)
{
    for(std::vector<Block>::const_iterator i = blocks.begin(); i != blocks.end(); ++i)
    {
        if (*i == ptr)
        {
            i->zeroFill();
            break;
        }
    }
}

void BufferArea::zeroFill()
{
    for(std::vector<Block>::const_iterator i = blocks.begin(); i != blocks.end(); ++i)
    {
        i->zeroFill();
    }
}

void BufferArea::commit()
{
#ifndef OPENCV_ENABLE_MEMORY_SANITIZER
    if (!safe)
    {
        CV_Assert(totalSize > 0);
        CV_Assert(oneBuf == NULL);
        CV_Assert(!blocks.empty());
        oneBuf = fastMalloc(totalSize);
        void * ptr = oneBuf;
        for(std::vector<Block>::const_iterator i = blocks.begin(); i != blocks.end(); ++i)
        {
            ptr = i->fast_allocate(ptr);
        }
    }
#endif
}

void BufferArea::release()
{
    for(std::vector<Block>::const_iterator i = blocks.begin(); i != blocks.end(); ++i)
    {
        i->cleanup();
    }
    blocks.clear();
#ifndef OPENCV_ENABLE_MEMORY_SANITIZER
    if (oneBuf)
    {
        fastFree(oneBuf);
        oneBuf = 0;
    }
#endif
}

//==================================================================================================

}} // cv::utils::
