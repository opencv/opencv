#include "../precomp.hpp"

#ifndef __OPENCV_GGUFBUFFER_HPP__
#define __OPENCV_GGUFBUFFER_HPP__


namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN


// https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
enum ggml_type : uint32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_COUNT,
};

// https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
enum gguf_metadata_value_type : uint32_t {
    GGUF_METADATA_VALUE_TYPE_UINT8    = 0,
    GGUF_METADATA_VALUE_TYPE_INT8     = 1,
    GGUF_METADATA_VALUE_TYPE_UINT16   = 2,
    GGUF_METADATA_VALUE_TYPE_INT16    = 3,
    GGUF_METADATA_VALUE_TYPE_UINT32   = 4,
    GGUF_METADATA_VALUE_TYPE_INT32    = 5,
    GGUF_METADATA_VALUE_TYPE_FLOAT32  = 6,
    GGUF_METADATA_VALUE_TYPE_BOOL     = 7,
    GGUF_METADATA_VALUE_TYPE_STRING   = 8,
    GGUF_METADATA_VALUE_TYPE_ARRAY    = 9,
    GGUF_METADATA_VALUE_TYPE_UINT64   = 10,
    GGUF_METADATA_VALUE_TYPE_INT64    = 11,
    GGUF_METADATA_VALUE_TYPE_FLOAT64  = 12,
};

struct GGUFBuffer
{
    GGUFBuffer(const std::string & fileName);
    std::vector<uint8_t> buf; 
};

struct GGUFBufferReader
{
    GGUFBufferReader(Ptr<const GGUFBuffer> buffer) : buffer(buffer), current_offset(0) {}
    // read single value
    template<typename T,typename R> R readSingleValue();
    template<typename T>int readSingleValueInt();
    int64 readSingleValueInt(gguf_metadata_value_type type);
    template<typename T>double readSingleValueReal();
    double readSingleValueReal(gguf_metadata_value_type type);
    std::string readString();
    DictValue readSingleValue(uint32_t type);

    // read array
    DictValue readIntArray(int n, gguf_metadata_value_type type);
    DictValue readRealArray(int n, gguf_metadata_value_type type);
    DictValue readStringArray(int n);

    // Mat 
    Mat read2DMat(ggml_type type, size_t rows, size_t cols, size_t offset);
    Mat read1DMat(ggml_type type, size_t rows, size_t offset);
    Ptr<const GGUFBuffer> buffer;
    size_t current_offset;
};


template<typename T,typename R>
R GGUFBufferReader::readSingleValue() {
    T value = *reinterpret_cast<const T*>(buffer->buf.data() + current_offset);
    current_offset += sizeof(T);
    return static_cast<R>(value);
}

template <typename T, typename R>
R checkRange(T value) {
    using Common = typename std::common_type<T, R>::type;
    if (static_cast<Common>(value) < static_cast<Common>(std::numeric_limits<R>::min()) ||
    static_cast<Common>(value) > static_cast<Common>(std::numeric_limits<R>::max())) {
        throw std::out_of_range("Value out of range");
    }
    return value;
}

// Parse single int value
template<typename T>
int GGUFBufferReader::readSingleValueInt() {
    T value = *reinterpret_cast<const T*>(buffer->buf.data() + current_offset);
    current_offset += sizeof(T);
    return value;
}

// Parse single float value
template<typename T>
double GGUFBufferReader::readSingleValueReal() {
    T value = *reinterpret_cast<const T*>(buffer->buf.data() + current_offset);
    current_offset += sizeof(T);
    return checkRange<T,float>(value) ;
}

CV__DNN_INLINE_NS_END
}}

#endif // __OPENCV_GGUFBUFFER_HPP__
