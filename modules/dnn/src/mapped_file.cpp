// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "mapped_file.hpp"

#include <atomic>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

// Holds the mapping in UMatData::userdata so it is released with the last Mat over it, the
// way NumpyAllocator (modules/python/src2) keeps a PyObject alive behind a wrapped array.
class MappedFileAllocator CV_FINAL : public MatAllocator
{
public:
    MappedFileAllocator() : stdAllocator(Mat::getStdAllocator()) {}

    UMatData* allocate(int dims, const int* sizes, int type, void* data, size_t* step,
                       AccessFlag flags, UMatUsageFlags usageFlags) const CV_OVERRIDE
    {
        return stdAllocator->allocate(dims, sizes, type, data, step, flags, usageFlags);
    }

    bool allocate(UMatData* u, AccessFlag accessFlags, UMatUsageFlags usageFlags) const CV_OVERRIDE
    {
        return stdAllocator->allocate(u, accessFlags, usageFlags);
    }

    void deallocate(UMatData* u) const CV_OVERRIDE
    {
        if (!u)
            return;
        if (u->refcount == 0)
        {
            delete static_cast<Ptr<MappedFile>*>(u->userdata);
            delete u;
        }
    }

    const MatAllocator* stdAllocator;
};

MappedFileAllocator& mappedFileAllocator()
{
    static MappedFileAllocator allocator;
    return allocator;
}

std::atomic<uint64_t>& viewCounter()
{
    static std::atomic<uint64_t> counter(0);
    return counter;
}

// The allocator identity is the marker: wrap() is the only thing that installs it.
const Ptr<MappedFile>* mappedViewOf(const Mat& m)
{
    if (!m.u || m.u->currAllocator != &mappedFileAllocator())
        return nullptr;
    return static_cast<const Ptr<MappedFile>*>(m.u->userdata);
}

}

uint64_t mappedViewCount()
{
    return viewCounter().load();
}

bool isMappedTensor(const Mat& m)
{
    const Ptr<MappedFile>* view = mappedViewOf(m);
    return view != nullptr && !view->empty();
}

bool releaseMappedPages(const Mat& m, size_t offset, size_t length)
{
    const Ptr<MappedFile>* view = mappedViewOf(m);
    if (!view || view->empty())
        return false;
    return (*view)->discardPayload(offset, length);
}

bool MappedFile::discardPayload(size_t offset, size_t length) const
{
    const size_t granularity = mappingGranularity();
    if (!viewBase || length == 0 || granularity == 0)
        return false;

    const size_t payloadLength = viewLength - (size_t)(payload - viewBase);
    if (offset > payloadLength || length > payloadLength - offset)
        return false;

    // Whole pages only: a partial page at either edge also holds bytes outside the range.
    uchar* first = alignPtr(payload + offset, (int)granularity);
    uchar* last = (uchar*)((size_t)(payload + offset + length) & ~(granularity - 1));
    return first < last && discardRange(first, (size_t)(last - first));
}

Mat MappedFile::wrap(const Ptr<MappedFile>& file, int dims, const int* sizes, int type)
{
    Mat m(dims, sizes, type, file->payload);
    UMatData* u = new UMatData(&mappedFileAllocator());
    u->data = u->origdata = file->payload;
    u->size = m.total() * m.elemSize();
    u->flags |= UMatData::USER_ALLOCATED;
    u->userdata = new Ptr<MappedFile>(file);
    m.u = u;
    m.addref();
    m.allocator = &mappedFileAllocator();
    return m;
}

bool MappedFile::prepareRange(const Ptr<MappedSource>& src, size_t offset, size_t length,
                              size_t& viewOffset, size_t& viewLength, size_t& delta)
{
    // mmap() does not check the range against the file; reading past the end raises SIGBUS.
    if (src.empty() || length == 0 || offset > src->fileSize() ||
        length > src->fileSize() - offset)
        return false;

    const size_t granularity = mappingGranularity();
    if (granularity == 0)
        return false;

    delta = offset % granularity;
    viewOffset = offset - delta;
    viewLength = delta + length;
    return true;
}

Ptr<MappedFile> MappedFile::finishView(void* base, size_t viewLength, size_t delta)
{
    if (!base)
        return Ptr<MappedFile>();

    Ptr<MappedFile> view = makePtr<MappedFile>();
    view->viewBase = (uchar*)base;
    view->viewLength = viewLength;
    view->payload = (uchar*)base + delta;
    viewCounter()++;
    return view;
}

CV__DNN_INLINE_NS_END
}}
