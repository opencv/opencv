// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/core/utils/buffer_area.private.hpp"
#include "opencv2/core/utils/configuration.private.hpp"

#ifdef OPENCV_ENABLE_MEMORY_SANITIZER
#define BUFFER_AREA_DEFAULT_MODE true
#else
#define BUFFER_AREA_DEFAULT_MODE false
#endif

static bool CV_BUFFER_AREA_OVERRIDE_SAFE_MODE =
    cv::utils::getConfigurationParameterBool("OPENCV_BUFFER_AREA_ALWAYS_SAFE", BUFFER_AREA_DEFAULT_MODE);

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
    void * fast_allocate(void * buf) const
    {
        CV_Assert(ptr && *ptr == NULL);
        buf = alignPtr(buf, alignment);
        CV_Assert(reinterpret_cast<size_t>(buf) % alignment == 0);
        *ptr = buf;
        return static_cast<void*>(static_cast<uchar*>(*ptr) + type_size * count);
    }
private:
    void **ptr;
    void * raw_mem;
    size_t count;
    ushort type_size;
    ushort alignment;
};

//==================================================================================================

BufferArea::BufferArea(bool safe_) :
    oneBuf(0),
    totalSize(0),
    safe(safe_ || CV_BUFFER_AREA_OVERRIDE_SAFE_MODE)
{
}

BufferArea::~BufferArea()
{
    for(std::vector<Block>::const_iterator i = blocks.begin(); i != blocks.end(); ++i)
        i->cleanup();
    if (oneBuf)
        fastFree(oneBuf);
}

void BufferArea::allocate_(void **ptr, ushort type_size, size_t count, ushort alignment)
{
    blocks.push_back(Block(ptr, type_size, count, alignment));
    if (safe)
        blocks.back().real_allocate();
    else
        totalSize += blocks.back().getByteCount();
}

void BufferArea::commit()
{
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
}

//==================================================================================================

}} // cv::utils::
