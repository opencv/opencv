#ifndef __OPENCV_DNN_GGUF_DEF_HPP__
#define __OPENCV_DNN_GGUF_DEF_HPP__

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <type_traits>
#include <variant>
#include <vector>
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

// ===============
// GGUF Structures
// ===============
struct gguf_string_t {
    uint64_t len;
    std::string data;
};





union gguf_metadata_value_t {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    int64_t int64;
    double float64;
    bool bool_;
    gguf_string_t string;
    struct {
        // Any value type is valid, including arrays.
        gguf_metadata_value_type type;
        // Number of elements, not bytes
        uint64_t len;
        // The array of values
        gguf_metadata_value_t*array;
    } array;
};

struct gguf_metadata_kv_t {
    gguf_string_t key;
    gguf_metadata_value_type value_type;
    gguf_metadata_value_t value;
};

struct gguf_header_t {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    std::vector<gguf_metadata_kv_t> metadata_kv;
};

struct gguf_tensor_info_t {
    gguf_string_t name;
    uint32_t n_dimensions;
    std::vector<uint64_t> dimensions;
    ggml_type type;
    uint64_t offset;
};

struct gguf_file_t {
    gguf_header_t header;
    std::vector<gguf_tensor_info_t> tensor_infos;
    uint8_t _padding;
    uint8_t tensor_data[]; // flexible array member (pointer interpreted)
};





// inline gguf_string_t parseGGUFString(const uint8_t* buffer, size_t offset)
// {
//     gguf_string_t result;
//     result.len = *reinterpret_cast<const uint64_t*>(buffer + offset);
//     result.data.assign(reinterpret_cast<const char*>(buffer + offset + sizeof(uint64_t)),
//                             result.len);
//     return result;
// }


// inline gguf_metadata_array_t parseGGUFMetadataArrayT(const uint8_t* buffer, size_t offset)
// {
//     GGUFParsedObject<gguf_metadata_array_t> result;
//     result.offsetInFile = offset;

//     // Parse array type (4 bytes)
//     result.data.type = *reinterpret_cast<const gguf_metadata_value_type*>(buffer + offset);
//     offset += sizeof(gguf_metadata_value_type);

//     // Parse array length (8 bytes)
//     result.data.len = *reinterpret_cast<const uint64_t*>(buffer + offset);
//     offset += sizeof(uint64_t);

//     // Allocate array
//     result.data.array = new gguf_metadata_value_t[result.data.len];

//     // Parse each element based on array type
//     for (uint64_t i = 0; i < result.data.len; ++i)
//     {
//         gguf_metadata_value_t val;

//         switch (result.data.type)
//         {
//             case GGUF_METADATA_VALUE_TYPE_UINT8:
//                 val.value = *reinterpret_cast<const uint8_t*>(buffer + offset);
//                 offset += sizeof(uint8_t);
//                 break;
//             case GGUF_METADATA_VALUE_TYPE_INT8:
//                 val.value = *reinterpret_cast<const int8_t*>(buffer + offset);
//                 offset += sizeof(int8_t);
//                 break;
//             case GGUF_METADATA_VALUE_TYPE_UINT16:
//                 val.value = *reinterpret_cast<const uint16_t*>(buffer + offset);
//                 offset += sizeof(uint16_t);
//                 break;
//             case GGUF_METADATA_VALUE_TYPE_INT16:
//                 val.value = *reinterpret_cast<const int16_t*>(buffer + offset);
//                 offset += sizeof(int16_t);
//                 break;
//             case GGUF_METADATA_VALUE_TYPE_UINT32:
//                 val.value = *reinterpret_cast<const uint32_t*>(buffer + offset);
//                 offset += sizeof(uint32_t);
//                 break;
//             case GGUF_METADATA_VALUE_TYPE_INT32:
//                 val.value = *reinterpret_cast<const int32_t*>(buffer + offset);
//                 offset += sizeof(int32_t);
//                 break;
//             case GGUF_METADATA_VALUE_TYPE_FLOAT32:
//                 val.value = *reinterpret_cast<const float*>(buffer + offset);
//                 offset += sizeof(float);
//                 break;
//             case GGUF_METADATA_VALUE_TYPE_BOOL:
//                 val.value = *reinterpret_cast<const bool*>(buffer + offset);
//                 offset += sizeof(bool);
//                 break;
//             case GGUF_METADATA_VALUE_TYPE_UINT64:
//                 val.value = *reinterpret_cast<const uint64_t*>(buffer + offset);
//                 offset += sizeof(uint64_t);
//                 break;
//             case GGUF_METADATA_VALUE_TYPE_INT64:
//                 val.value = *reinterpret_cast<const int64_t*>(buffer + offset);
//                 offset += sizeof(int64_t);
//                 break;
//             case GGUF_METADATA_VALUE_TYPE_FLOAT64:
//                 val.value = *reinterpret_cast<const double*>(buffer + offset);
//                 offset += sizeof(double);
//                 break;
//             case GGUF_METADATA_VALUE_TYPE_STRING: {
//                 auto strParsed = parseGGUFString(buffer, offset);
//                 val.value = strParsed.data;
//                 offset += sizeof(uint64_t) + strParsed.data.len;
//                 break;
//             }
//             default:
//                 // Unsupported element type in array
//                 // Optionally throw/log error
//                 break;
//         }

//         result.data.array[i] = val;
//     }

//     return result;
// }


// inline GGUFParsedObject<gguf_metadata_kv_t> parseGGUFMetadataKVt(const uint8_t* buffer, size_t offset)
// {
//     GGUFParsedObject<gguf_metadata_kv_t> result;
//     result.offsetInFile = offset;

//     // Parse key
//     auto keyParsed = parseGGUFString(buffer, offset);
//     result.data.key = keyParsed;
//     size_t currentOffset = offset + sizeof(uint64_t) + keyParsed.data.len;

//     // Parse value type
//     result.data.value_type = *reinterpret_cast<const gguf_metadata_value_type*>(buffer + currentOffset);
//     currentOffset += sizeof(gguf_metadata_value_type);

//     // Parse value based on type
//     switch (result.data.value_type)
//     {
//         case GGUF_METADATA_VALUE_TYPE_UINT8:
//             result.data.value.value = *reinterpret_cast<const uint8_t*>(buffer + currentOffset);
//             break;
//         case GGUF_METADATA_VALUE_TYPE_INT8:
//             result.data.value.value = *reinterpret_cast<const int8_t*>(buffer + currentOffset);
//             break;
//         case GGUF_METADATA_VALUE_TYPE_UINT16:
//             result.data.value.value = *reinterpret_cast<const uint16_t*>(buffer + currentOffset);
//             break;
//         case GGUF_METADATA_VALUE_TYPE_INT16:
//             result.data.value.value = *reinterpret_cast<const int16_t*>(buffer + currentOffset);
//             break;
//         case GGUF_METADATA_VALUE_TYPE_UINT32:
//             result.data.value.value = *reinterpret_cast<const uint32_t*>(buffer + currentOffset);
//             break;
//         case GGUF_METADATA_VALUE_TYPE_INT32:
//             result.data.value.value = *reinterpret_cast<const int32_t*>(buffer + currentOffset);
//             break;
//         case GGUF_METADATA_VALUE_TYPE_FLOAT32:
//             result.data.value.value = *reinterpret_cast<const float*>(buffer + currentOffset);
//             break;
//         case GGUF_METADATA_VALUE_TYPE_BOOL:
//             result.data.value.value = *reinterpret_cast<const bool*>(buffer + currentOffset);
//             break;
//         case GGUF_METADATA_VALUE_TYPE_UINT64:
//             result.data.value.value = *reinterpret_cast<const uint64_t*>(buffer + currentOffset);
//             break;
//         case GGUF_METADATA_VALUE_TYPE_INT64:
//             result.data.value.value = *reinterpret_cast<const int64_t*>(buffer + currentOffset);
//             break;
//         case GGUF_METADATA_VALUE_TYPE_FLOAT64:
//             result.data.value.value = *reinterpret_cast<const double*>(buffer + currentOffset);
//             break;
//         case GGUF_METADATA_VALUE_TYPE_STRING: {
//             auto strParsed = parseGGUFString(buffer, currentOffset);
//             result.data.value.value = strParsed.data;
//             break;
//         }
//         case GGUF_METADATA_VALUE_TYPE_ARRAY: {
//             // NOTE: assuming you will define `parseGGUFMetadataArrayT()` that returns GGUFParsedObject<gguf_metadata_array_t>
//             auto arrayParsed = parseGGUFMetadataArrayT(buffer, currentOffset);
//             result.data.value.value = arrayParsed;
//             break;
//         }
//         default:
//             // TODO: Handle unknown/unsupported value_type
//             break;
//     }

//     return result;
// }




CV__DNN_INLINE_NS_END
}} // namespace cv::dnn

#endif  // __OPENCV_DNN_GGUF_IMPORTER_HPP__