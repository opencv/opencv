#include "../../include/opencv2/dnn/tokenizer.hpp"
#include "gpt2_tokenizer_fast.hpp"

namespace cv { namespace dnn { namespace tokenizer {

Tokenizer Tokenizer::from_pretrained(const std::string& name, const std::string& pretrained_model_path) {
    // We most load files json into FileStorge
    std::shared_ptr<Encoding> enc;
    if (name == "gpt2") {
        enc = std::make_shared<Encoding>(GPT2TokenizerFast::from_pretrained(pretrained_model_path).encoding());
    } else if (name == "cl100k_base") {
      enc = std::make_shared<Encoding>(getEncodingForCl100k_base(name, pretrained_model_path));  
    } else {
        throw std::runtime_error("Unknown model name: " + name);
    }
    return Tokenizer(std::move(enc), name);
}

}}}