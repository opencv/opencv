// Helpers for reading and interpreting GPT-2 encoder JSON, handling
//  UTF-8/Unicode, and other small utilities used by the DNN tokenizer.
 

#pragma once
#include <unordered_map>
#include <string>

namespace cv { namespace dnn { namespace tokenizer {
    
// JSON parsing 
void append_utf8(uint32_t codepoint, std::string& out);

std::string unescape_json(const std::string& s);

// Return a mapping: token string (raw bytes) -> rank
std::unordered_map<std::string, int> read_encoder_json(const std::string& path);

}}} // namespace cv namespace dnn namespace tokenizer