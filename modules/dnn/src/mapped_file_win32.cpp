// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "mapped_file.hpp"

#ifdef _WIN32

#include <windows.h>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

// CreateFileW rather than CreateFileA, so a model under a non-ASCII path still maps.
bool widenPath(const std::string& path, std::wstring& wide)
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

}

// Windows wants the allocation granularity, not its page size: a view must start on a
// multiple of it or MapViewOfFile fails with ERROR_MAPPED_ALIGNMENT.
size_t mappingGranularity()
{
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return (size_t)info.dwAllocationGranularity;
}

Ptr<MappedSource> MappedSource::open(const std::string& path)
{
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

    Ptr<MappedSource> src = makePtr<MappedSource>();
    src->path_ = path;
    src->fileSize_ = (size_t)sz.QuadPart;
    return src;
}

Ptr<MappedFile> MappedFile::open(const Ptr<MappedSource>& src, size_t offset, size_t length)
{
    size_t viewOffset = 0, viewLength = 0, delta = 0;
    if (!prepareRange(src, offset, length, viewOffset, viewLength, delta))
        return Ptr<MappedFile>();

    std::wstring widePath;
    if (!widenPath(src->path_, widePath))
        return Ptr<MappedFile>();

    HANDLE f = CreateFileW(widePath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE)
        return Ptr<MappedFile>();

    void* p = NULL;
    // PAGE_WRITECOPY, so writing through the Mat detaches the page instead of faulting.
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

    return finishView(p, viewLength, delta);
}

MappedFile::~MappedFile()
{
    if (viewBase)
        UnmapViewOfFile(viewBase);
}

CV__DNN_INLINE_NS_END
}}

#endif  // _WIN32
