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
        explicit IterLoad(bool no_throw = true) : m_no_throw(no_throw) {}

        ~IterLoad();

        void read(const String &filename, int flags = 0);

        void decode(InputArray buf, int flags = 0);

        bool empty() const { return size() == 0; }

        std::size_t size() const;

        Mat next(int flags = IMREAD_COLOR, std::map<String, String> *properties = nullptr,
                 Mat *dst = nullptr) const;

        Mat at(int idx = 0, int flags = IMREAD_COLOR, std::map<String, String> *properties = nullptr,
               Mat *dst = nullptr) const;

    private:
        bool m_no_throw;
        String m_tempfile;
        Ptr <BaseImageDecoder> m_decoder;

        IterLoad(const IterLoad &);

        IterLoad &operator=(const IterLoad &);

        void load(const String *filename, const _InputArray *buf, int flags);
    };
}

#endif /*OPENCV_IMGCODECS_ITERLOAD_HPP*/
