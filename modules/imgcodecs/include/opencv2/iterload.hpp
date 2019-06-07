//
// Created by jan on 04.06.19.
//

#ifndef OPENCV_IMGCODECS_ITERLOAD_HPP
#define OPENCV_IMGCODECS_ITERLOAD_HPP

#include "opencv2/imgcodecs.hpp"

#include <map>

namespace cv
{
    class BaseImageDecoder;

    class CV_EXPORTS IterLoad
    {
    public:
        IterLoad();

        ~IterLoad();

        /// flags: only IMREAD_UNCHANGED, IMREAD_LOAD_GDAL are recognized here
        bool read(const String &filename, int flags = IMREAD_UNCHANGED);

        /// flags: only IMREAD_UNCHANGED, IMREAD_LOAD_GDAL are recognized here
        bool decode(InputArray buf, int flags = IMREAD_UNCHANGED);

        bool empty() const;

        std::size_t size() const;

        // sequential access
        bool hasNext() const;

        // sequential access
        Mat next(int flags = IMREAD_COLOR, Mat *dst = 0);

        // random access
        Mat at(int idx, int flags = IMREAD_COLOR, Mat *dst = 0);

    private:
        String m_file;
        Mat m_buf;
        String m_tempfile;
        Ptr <BaseImageDecoder> m_decoder;
        bool m_has_next;

        IterLoad(const IterLoad &); ///< deny copy

        IterLoad &operator=(const IterLoad &); ///< deny copy

        /// filename XOR buf
        /// flags: only IMREAD_UNCHANGED, IMREAD_LOAD_GDAL are recognized here
        bool load(const String *filename, const _InputArray *buf, int flags);

        void clear();
    };
}

#endif /*OPENCV_IMGCODECS_ITERLOAD_HPP*/
