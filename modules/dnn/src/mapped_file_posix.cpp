// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "mapped_file.hpp"

#if !defined(_WIN32) && (defined(__unix__) || defined(__APPLE__))

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

size_t mappingGranularity()
{
    static const long pageSize = sysconf(_SC_PAGESIZE);
    return pageSize > 0 ? (size_t)pageSize : 0;
}

Ptr<MappedSource> MappedSource::open(const std::string& path)
{
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        return Ptr<MappedSource>();

    struct stat st;
    const bool ok = fstat(fd, &st) == 0 && st.st_size > 0;
    ::close(fd);
    if (!ok)
        return Ptr<MappedSource>();

    Ptr<MappedSource> src = makePtr<MappedSource>();
    src->path_ = path;
    src->fileSize_ = (size_t)st.st_size;
    return src;
}

Ptr<MappedFile> MappedFile::open(const Ptr<MappedSource>& src, size_t offset, size_t length)
{
    size_t viewOffset = 0, viewLength = 0, delta = 0;
    if (!prepareRange(src, offset, length, viewOffset, viewLength, delta))
        return Ptr<MappedFile>();

    int fd = ::open(src->path_.c_str(), O_RDONLY);
    if (fd < 0)
        return Ptr<MappedFile>();

    // MAP_PRIVATE with PROT_WRITE, so writing through the Mat detaches the page instead of
    // faulting, and never reaches the file.
    void* p = mmap(NULL, viewLength, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, (off_t)viewOffset);
    // The mapping carries its own reference, so a model split across one file per tensor never
    // accumulates descriptors.
    ::close(fd);
    if (p == MAP_FAILED)
        p = NULL;

    return finishView(p, viewLength, delta);
}

MappedFile::~MappedFile()
{
    if (viewBase)
        munmap(viewBase, viewLength);
}

CV__DNN_INLINE_NS_END
}}

#endif  // !_WIN32 && (__unix__ || __APPLE__)
