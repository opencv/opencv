#include "tokenizer.hpp"

namespace cv { namespace dnn { namespace tokenizer {

class CV_EXPORTS GPT2TokenizerFast : public Tokenizer {
public:
    static GPT2TokenizerFast from_pretrained(const std::string& pretrain_model_path) {
        return GPT2TokenizerFast(getEncodingForGPT2("gpt2", pretrain_model_path));
    }
    CV_EXPORTS static Encoding getEncodingForGPT2(const std::string &name, const std::string& vocab_file);
    static std::string decodeDataGym(const std::string& s,
                                     const std::unordered_map<uint32_t,uint8_t>& dg2b);
    static std::unordered_map<std::string,int> dataGymToMergeableBpeRanks(
                                        const std::string& vocabBpePath
                                        /*const std::string& encoderJsonPath*/);
private:
    explicit GPT2TokenizerFast(Encoding enc) : Tokenizer(std::make_shared<Encoding>(std::move(enc))) {}
};

}}}