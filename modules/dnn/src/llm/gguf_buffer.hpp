#include "../precomp.hpp"

#ifndef __OPENCV_GGUFBUFFER_HPP__
#define __OPENCV_GGUFBUFFER_HPP__


namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

struct GGUFBuffer
{
    GGUFBuffer(const std::string & fileName);
    std::vector<uint8_t> buf; 
};


struct GGUFBufferReader
{
    GGUFBufferReader(Ptr<const GGUFBuffer> buffer) : buffer(buffer), current_offset(0) {}
    template<typename T>T readSingleValue();
    std::string readString(size_t len);
    Mat read2DMat(ggml_type type, size_t rows, size_t cols, size_t offset);
    Mat read1DMat(ggml_type type, size_t rows, size_t offset);
    Ptr<const GGUFBuffer> buffer;
    size_t current_offset;
};

template<typename T>
T GGUFBufferReader::readSingleValue() {
    T value = *reinterpret_cast<const T*>(buffer->buf.data() + current_offset);
    current_offset += sizeof(T);
    return value;
}

CV__DNN_INLINE_NS_END
}}

#endif // __OPENCV_GGUFBUFFER_HPP__