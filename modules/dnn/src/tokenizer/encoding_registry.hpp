#pragma once

#include "encoding.hpp"
#include <mutex>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <functional>

namespace cv { namespace dnn { namespace tokenizer {

using EncodingCtor = std::function<Encoding()>;


static std::mutex mtx;
static std::unordered_map<std::string, EncodingCtor> ctors;
static std::unordered_map<std::string, Encoding> instances;


CV_EXPORTS void registerEncoding(const std::string &name, EncodingCtor ctor);

CV_EXPORTS const Encoding& getEncoding(const std::string &name);

CV_EXPORTS std::vector<std::string> listEncodingNames();

}}}
