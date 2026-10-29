// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "mapped_file.hpp"

#if !defined(_WIN32) && !defined(__unix__) && !defined(__APPLE__)

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// No mapping on this platform. Every entry point reports that, and the ONNX importer reads
// external data through its ordinary copy path instead.

size_t mappingGranularity()
{
    return 0;
}

Ptr<MappedSource> MappedSource::open(const std::string& path)
{
    CV_UNUSED(path);
    return Ptr<MappedSource>();
}

Ptr<MappedFile> MappedFile::open(const Ptr<MappedSource>& src, size_t offset, size_t length)
{
    CV_UNUSED(src); CV_UNUSED(offset); CV_UNUSED(length);
    return Ptr<MappedFile>();
}

MappedFile::~MappedFile()
{
}

CV__DNN_INLINE_NS_END
}}

#endif  // !_WIN32 && !__unix__ && !__APPLE__
