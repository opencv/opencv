#include "../../include/opencv2/dnn/tokenizer.hpp"
#include "encoding.hpp"
#include <opencv2/core.hpp>
#include <iostream>
#include <map>

namespace cv { namespace dnn { namespace tokenizer {

class CV_EXPORTS GPT2TokenizerFast : public Tokenizer {
public:
    static GPT2TokenizerFast from_pretrained(const std::string& pretrain_model_path) {
        return GPT2TokenizerFast(getEncodingForGPT2FromJSON("gpt2", pretrain_model_path));
    }
    
    Tokenizer train_bpe_from_corpus(const std::string& corpus,
                                   int vocab_sz,
                                   const std::string& pattern) = delete;
    Tokenizer train_bpe_from_corpus(const std::vector<std::string>& corpus,
                                   int vocab_sz,
                                   const std::string& pattern,
                                   int min_freq=2, 
                                   int max_token_length=std::numeric_limits<int>::max(),
                                    bool verbose=false) = delete;

    CV_EXPORTS static Encoding getEncodingForGPT2(const std::string &name, const std::string& vocab_file);
    CV_EXPORTS static Encoding getEncodingForGPT2FromJSON(const std::string &name, const std::string& vocab_file);
    static std::string decodeDataGym(const std::string& s,
                                     const std::unordered_map<uint32_t,uint8_t>& dg2b);
    static std::unordered_map<std::string,int> dataGymToMergeableBpeRanks(
                                        const std::string& vocabBpePath
                                        /*const std::string& encoderJsonPath*/);
private:
    explicit GPT2TokenizerFast(Encoding enc) : Tokenizer(std::make_shared<Encoding>(std::move(enc))) {}
};

}}}