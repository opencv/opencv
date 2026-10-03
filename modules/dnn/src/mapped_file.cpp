// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "mapped_file.hpp"

#include <atomic>

#ifdef _WIN32
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#define DNN_MMAP_AVAILABLE 1
#endif

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

static MappedFileAllocator& mappedFileAllocator()
{
    static MappedFileAllocator allocator;
    return allocator;
}

// Windows wants the allocation granularity, not its page size. 0 means the caller must not map.
static size_t mappingGranularity()
{
#ifdef _WIN32
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return (size_t)info.dwAllocationGranularity;
#elif defined(DNN_MMAP_AVAILABLE)
    static const long pageSize = sysconf(_SC_PAGESIZE);
    return pageSize > 0 ? (size_t)pageSize : 0;
#else
    return 0;
#endif
}

#ifdef _WIN32
// CreateFileW rather than CreateFileA, so a model under a non-ASCII path still maps.
static bool widenPath(const std::string& path, std::wstring& wide)
{
    const int len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                        path.c_str(), (int)path.size(), NULL, 0);
    if (len <= 0)
        return false;
    wide.assign((size_t)len, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                        path.c_str(), (int)path.size(), &wide[0], len);
    return true;
}
#endif

static std::atomic<uint64_t>& viewCounter()
{
    static std::atomic<uint64_t> counter(0);
    return counter;
}

}

uint64_t mappedViewCount()
{
    return viewCounter().load();
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

Ptr<MappedSource> MappedSource::open(const std::string& path)
{
    size_t fileSize = 0;
#ifdef _WIN32
    std::wstring widePath;
    if (!widenPath(path, widePath))
        return Ptr<MappedSource>();
    HANDLE f = CreateFileW(widePath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE)
        return Ptr<MappedSource>();
    LARGE_INTEGER sz;
    const bool ok = GetFileSizeEx(f, &sz) != 0 && sz.QuadPart > 0;
    CloseHandle(f);
    if (!ok)
        return Ptr<MappedSource>();
    fileSize = (size_t)sz.QuadPart;
#elif defined(DNN_MMAP_AVAILABLE)
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        return Ptr<MappedSource>();
    struct stat st;
    const bool ok = fstat(fd, &st) == 0 && st.st_size > 0;
    ::close(fd);
    if (!ok)
        return Ptr<MappedSource>();
    fileSize = (size_t)st.st_size;
#else
    CV_UNUSED(path);
    CV_UNUSED(fileSize);
    return Ptr<MappedSource>();
#endif
    Ptr<MappedSource> src(new MappedSource());
    src->path_ = path;
    src->fileSize_ = fileSize;
    return src;
}

MappedFile::~MappedFile()
{
    if (!viewBase)
        return;
#ifdef _WIN32
    UnmapViewOfFile(viewBase);
#elif defined(DNN_MMAP_AVAILABLE)
    munmap(viewBase, viewLength);
#endif
}

Ptr<MappedFile> MappedFile::open(const Ptr<MappedSource>& src, size_t offset, size_t length)
{
    // mmap() does not check the range against the file; reading past the end raises SIGBUS.
    if (src.empty() || length == 0 || offset > src->fileSize() ||
        length > src->fileSize() - offset)
        return Ptr<MappedFile>();

    const size_t granularity = mappingGranularity();
    if (granularity == 0)
        return Ptr<MappedFile>();
    const size_t delta = offset % granularity;
    const size_t viewOffset = offset - delta;
    const size_t viewLength = delta + length;

    void* p = NULL;
#ifdef _WIN32
    std::wstring widePath;
    if (!widenPath(src->path_, widePath))
        return Ptr<MappedFile>();
    HANDLE f = CreateFileW(widePath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE)
        return Ptr<MappedFile>();
    HANDLE m = CreateFileMappingW(f, NULL, PAGE_WRITECOPY, 0, 0, NULL);
    if (m)
    {
        const uint64_t off = (uint64_t)viewOffset;
        p = MapViewOfFile(m, FILE_MAP_COPY,
                          (DWORD)(off >> 32), (DWORD)(off & 0xFFFFFFFFu), viewLength);
        CloseHandle(m);
    }
    // The view holds the section and the section the file, so a model split across one file per
    // tensor never accumulates handles.
    CloseHandle(f);
#elif defined(DNN_MMAP_AVAILABLE)
    int fd = ::open(src->path_.c_str(), O_RDONLY);
    if (fd < 0)
        return Ptr<MappedFile>();
    p = mmap(NULL, viewLength, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, (off_t)viewOffset);
    // The mapping carries its own reference, so a model split across one file per tensor never
    // accumulates descriptors.
    ::close(fd);
    if (p == MAP_FAILED)
        p = NULL;
#else
    CV_UNUSED(viewOffset);
#endif
    if (!p)
        return Ptr<MappedFile>();

    Ptr<MappedFile> view(new MappedFile());
    view->viewBase = (uchar*)p;
    view->viewLength = viewLength;
    view->payload = (uchar*)p + delta;
    viewCounter()++;
    return view;
}

CV__DNN_INLINE_NS_END
}}
