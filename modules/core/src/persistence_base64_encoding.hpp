// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_CORE_BASE64_ENCODING_HPP
#define OPENCV_CORE_BASE64_ENCODING_HPP

namespace cv
{

namespace base64
{
/* A decorator for CvFileStorage
* - no copyable
* - not safe for now
* - move constructor may be needed if C++11
*/
uint8_t const base64_mapping[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

uint8_t const base64_padding = '=';

std::string make_base64_header(const char * dt);

size_t base64_encode(uint8_t const * src, uint8_t * dst, size_t off, size_t cnt);


int icvCalcStructSize( const char* dt, int initial_size );

class Base64ContextEmitter;
class Impl;

class Base64Writer
{
public:
    Base64Writer(cv::FileStorage::Impl& fs, bool can_indent);
    ~Base64Writer();
    void write(const void* _data, size_t len, const char* dt);
    template<typename _to_binary_convertor_t> void write(_to_binary_convertor_t & convertor, const char* dt);

private:
    void check_dt(const char* dt);

private:
    // disable copy and assignment
    Base64Writer(const Base64Writer &);
    Base64Writer & operator=(const Base64Writer &);

private:

    Base64ContextEmitter * emitter;
    std::string data_type_string;
};

size_t base64_encode_buffer_size(size_t cnt, bool is_end_with_zero = true);

template<typename _uint_t> inline size_t
to_binary(_uint_t val, uchar * cur)
{
    size_t delta = CHAR_BIT;
    size_t cnt = sizeof(_uint_t);
    while (cnt --> static_cast<size_t>(0U)) {
        *cur++ = static_cast<uchar>(val);
        val >>= delta;
    }
    return sizeof(_uint_t);
}

template<> inline size_t to_binary(bool val, uchar * cur)
{
    return to_binary(static_cast<uchar>(val), cur);
}

template<> inline size_t to_binary(double val, uchar * cur)
{
    Cv64suf bit64;
    bit64.f = val;
    return to_binary(bit64.u, cur);
}

template<> inline size_t to_binary(float val, uchar * cur)
{
    Cv32suf bit32;
    bit32.f = val;
    return to_binary(bit32.u, cur);
}

template<typename _primitive_t> inline size_t
to_binary(uchar const * val, uchar * cur)
{
    return to_binary<_primitive_t>(*reinterpret_cast<_primitive_t const *>(val), cur);
}



class RawDataToBinaryConvertor
{
public:
    // NOTE: len is already multiplied by element size here
    RawDataToBinaryConvertor(const void* src, int len, const std::string & dt);

    inline RawDataToBinaryConvertor & operator >>(uchar * & dst);
    inline operator bool() const;

private:
    typedef size_t(*to_binary_t)(const uchar *, uchar *);
    struct elem_to_binary_t
    {
        size_t      offset;
        size_t      offset_packed;
        to_binary_t func;
    };

private:
    size_t make_to_binary_funcs(const std::string &dt);

private:
    const uchar * beg;
    const uchar * cur;
    const uchar * end;

    size_t step;
    size_t step_packed;
    std::vector<elem_to_binary_t> to_binary_funcs;
};

}

}
#endif
