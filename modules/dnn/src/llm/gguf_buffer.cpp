#include "../precomp.hpp"
#include "gguf_def.hpp"
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


std::string GGUFBufferReader::readString(size_t len) {
    std::string str(reinterpret_cast<const char*>(buffer->buf.data() + current_offset), len);
    current_offset += len;
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

