#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include "../../../src/tokenizer/core_bpe.hpp"

namespace cv { namespace dnn { namespace tokenizer {

class CV_EXPORTS_W_SIMPLE Tokenizer {
public:

    CV_WRAP Tokenizer();
    Tokenizer(std::shared_ptr<CoreBPE> core);
    CV_WRAP static Tokenizer load(const std::string& model_dir); 
    // Encoding
    CV_WRAP std::vector<int> encode(const std::string& text);
    // Decoding
    std::string decode(const std::vector<int>& tokens);
private:
    std::shared_ptr<CoreBPE> coreBPE_;
};

}}}

