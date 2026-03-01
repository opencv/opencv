#ifndef OPENCV_DNN_TOKENIZER_HPP
#define OPENCV_DNN_TOKENIZER_HPP

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace cv {
namespace dnn {

/**
 * @brief Base class for text tokenizers in the DNN module.
 */
class CV_EXPORTS_W Tokenizer {
public:
    virtual ~Tokenizer() {}

    /** @brief Encodes text into a sequence of token IDs. */
    CV_WRAP virtual std::vector<int> encode(const std::string& text) = 0;

    /** @brief Decodes a sequence of token IDs back into text. */
    CV_WRAP virtual std::string decode(const std::vector<int>& tokens) = 0;

    /** @brief Converts token IDs into an OpenCV Mat (suitable for dnn::Net input). */
    CV_WRAP virtual Mat tokensToMat(const std::vector<int>& tokens);

    /** @brief Load vocabulary/merges from a file (e.g., JSON or YAML). */
    CV_WRAP virtual void load(const std::string& path) = 0;

    // Factory method to create specific instances
    CV_WRAP static Ptr<Tokenizer> createBPE(const std::string& vocabPath);
};

} // namespace dnn
} // namespace cv

#endif