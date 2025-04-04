#include "../precomp.hpp"
#include "gguf_buffer.hpp"


namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

GGUFBuffer::GGUFBuffer(const std::string & fileName){
    std::ifstream file(fileName, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: ");
    }

    // Get the size of the file and prepare a buffer
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    buf.resize(size);

    // Read the file content into the buffer
    if (!file.read(reinterpret_cast<char*>(buf.data()), size)) {
        throw std::runtime_error("Error reading file: " );
    }
}

DictValue GGUFBufferReader::readSingleValue(uint32_t type) {
    switch (type) {
        case GGUF_METADATA_VALUE_TYPE_UINT8:
            return DictValue(readSingleValue<uint8_t, int>());
        case GGUF_METADATA_VALUE_TYPE_INT8:
            return DictValue(readSingleValue<int8_t, int>());
        case GGUF_METADATA_VALUE_TYPE_UINT16:
            return DictValue(readSingleValue<uint16_t, int>());
        case GGUF_METADATA_VALUE_TYPE_INT16:
            return DictValue(readSingleValue<int16_t, int>());
        case GGUF_METADATA_VALUE_TYPE_UINT32:
            return DictValue(readSingleValue<uint32_t, int>());
        case GGUF_METADATA_VALUE_TYPE_INT32:
            return DictValue(readSingleValue<int32_t, int>());
        case GGUF_METADATA_VALUE_TYPE_FLOAT32:
            return DictValue(readSingleValue<float, float>());
        case GGUF_METADATA_VALUE_TYPE_BOOL:
            return DictValue(readSingleValue<uint8_t, int>() != 0);
        case GGUF_METADATA_VALUE_TYPE_STRING:
            return DictValue(readString());
        case GGUF_METADATA_VALUE_TYPE_UINT64:
            return DictValue(readSingleValue<uint64_t, int64>());
        case GGUF_METADATA_VALUE_TYPE_FLOAT64:
            return DictValue(readSingleValue<float, double>());
        case GGUF_METADATA_VALUE_TYPE_ARRAY:
            throw std::runtime_error("Tried to parse array as single value");
        default:
            throw std::runtime_error("Unsupported metadata type: " + std::to_string(type));
    }
}

DictValue GGUFBufferReader::readIntArray(int n, gguf_metadata_value_type type) {
    std::vector<int> arr(n);
    for (int i = 0; i < n; ++i) {
        arr[i] = readSingleValueInt(type);
    }
    return DictValue::arrayInt(arr.begin(), arr.size());
}

DictValue GGUFBufferReader::readRealArray(int n, gguf_metadata_value_type type) {
    std::vector<double> arr(n);
    for (int i = 0; i < n; ++i) {
        arr[i] = readSingleValueReal(type);
    }
    return DictValue::arrayReal(arr.begin(), arr.size());
}

DictValue GGUFBufferReader::readStringArray(int n) {
    std::vector<std::string> arr(n);
    for (int i = 0; i < n; ++i) {
        arr[i] = readString();
    }
    return DictValue::arrayString(arr.begin(), arr.size());
}

double GGUFBufferReader::readSingleValueReal(gguf_metadata_value_type type) {
    switch (type) {
        case GGUF_METADATA_VALUE_TYPE_FLOAT32:
            return readSingleValueReal<float>();
        case GGUF_METADATA_VALUE_TYPE_FLOAT64:
            return readSingleValueReal<double>();
        default:
            throw std::runtime_error("Unsupported metadata type: " + std::to_string(type));
    }
}

int64 GGUFBufferReader::readSingleValueInt(gguf_metadata_value_type type) {
    switch (type) {
        case GGUF_METADATA_VALUE_TYPE_UINT8:
            return readSingleValueInt<uint8_t>();
        case GGUF_METADATA_VALUE_TYPE_INT8:
            return readSingleValueInt<int8_t>();
        case GGUF_METADATA_VALUE_TYPE_UINT16:
            return readSingleValueInt<uint16_t>();
        case GGUF_METADATA_VALUE_TYPE_INT16:
            return readSingleValueInt<int16_t>();
        case GGUF_METADATA_VALUE_TYPE_UINT32:
            return readSingleValueInt<uint32_t>();
        case GGUF_METADATA_VALUE_TYPE_INT32:
            return readSingleValueInt<int32_t>();
        default:
            throw std::runtime_error("Unsupported metadata type: " + std::to_string(type));
    }
}

std::string GGUFBufferReader::readString() {
    uint32_t str_len = readSingleValue<uint64_t, int>(); // 28
    std::string str(reinterpret_cast<const char*>(buffer->buf.data() + current_offset), str_len);
    current_offset += str_len;
    return str;
}

Mat GGUFBufferReader::read2DMat(ggml_type type, size_t rows, size_t cols, size_t offset) {
    if (type != GGML_TYPE_F32) {
        throw std::runtime_error("Unsupported tensor type: " + std::to_string(type));
    }   
    const float* dataPtr = reinterpret_cast<const float*>(buffer->buf.data() + current_offset + offset);

    Mat mat((int)cols, (int)rows, CV_32F);

    for (size_t i = 0; i < cols; i++) {
        for (size_t j = 0; j < rows; j++) {
            mat.at<float>((int)i, (int)j) = dataPtr[i * rows + j];
        } 
    }
    return mat;
}

Mat GGUFBufferReader::read1DMat(ggml_type type, size_t rows, size_t offset) {
    if (type != GGML_TYPE_F32) {
        throw std::runtime_error("Unsupported tensor type: " + std::to_string(type));
    }   
    const float* dataPtr = reinterpret_cast<const float*>(buffer->buf.data() + current_offset + offset);

    Mat mat(rows, 1, CV_32F);
    for (size_t row = 0; row < rows; row++) {
        printf("r: %d, -- %f \n", (int)row, dataPtr[row]);
        mat.at<float>((int)row,0) = dataPtr[row];
    }

    return mat;
}

CV__DNN_INLINE_NS_END
}}

