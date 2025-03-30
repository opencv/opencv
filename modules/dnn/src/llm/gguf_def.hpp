#ifndef __OPENCV_GGUFDEF_HPP__
#define __OPENCV_GGUFDEF_HPP__

#include <cstdint>
#include <string>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN
// =====================
// GGML Type Enumeration
// =====================
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

// ==============================
// GGUF Metadata Value Type Enum
// ==============================
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

struct gguf_string_t {
    uint64_t len;
    std::string data;
};

union gguf_metadata_value_t {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    int64_t int64;
    double float64;
    bool bool_;
    gguf_string_t string;
    int16_t int16;
    struct {
        // Any value type is valid, including arrays.
        gguf_metadata_value_type type;
        // Number of elements, not bytes
        uint64_t len;
        // The array of values
        gguf_metadata_value_t*array;
    } array;
};



CV__DNN_INLINE_NS_END
}}


#endif