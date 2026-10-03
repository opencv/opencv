// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_SRC_MAPPED_FILE_HPP
#define OPENCV_DNN_SRC_MAPPED_FILE_HPP

#include <string>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// Where a weight file is and how far it extends, so each constant can be mapped on its own.
class MappedSource
{
public:
    // Empty Ptr when the platform or the file cannot supply a mapping; callers copy instead.
    static Ptr<MappedSource> open(const std::string& path);

    size_t fileSize() const { return fileSize_; }

private:
    MappedSource() {}
    MappedSource(const MappedSource&);
    MappedSource& operator=(const MappedSource&);
    friend class MappedFile;

    std::string path_;
    size_t fileSize_ = 0;
};

// One constant's bytes as a Mat header over the file. Copy-on-write: a Mat is always writable.
class MappedFile
{
public:
    ~MappedFile();

    // Maps [offset, offset + length) alone. Empty Ptr leaves the caller on the copy path.
    static Ptr<MappedFile> open(const Ptr<MappedSource>& src, size_t offset, size_t length);

    // Mat holding a reference to this mapping, so it outlives the Net like any other Mat.
    static Mat wrap(const Ptr<MappedFile>& file, int dims, const int* sizes, int type);

    uchar* data() const { return payload; }

private:
    MappedFile() {}
    MappedFile(const MappedFile&);
    MappedFile& operator=(const MappedFile&);

    // What the platform unmaps, never the payload: the base is rounded down to the granularity.
    uchar* viewBase = nullptr;
    size_t viewLength = 0;
    uchar* payload = nullptr;
};

// Views created so far; a refused mapping copies instead and yields identical values.
CV_EXPORTS uint64_t mappedViewCount();

CV__DNN_INLINE_NS_END
}}

#endif
