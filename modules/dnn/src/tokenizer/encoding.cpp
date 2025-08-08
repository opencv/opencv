/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  Portions of this file are inspired by or adapted from:
//      • tiktoken (Python) implementation:
//          https://github.com/openai/tiktoken/blob/main/tiktoken/core.py
//      • minbpe by Andrej Karpathy:
//          https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py
//
//  This file is part of the OpenCV DNN module for tokenization.
//
////////////////////////////////////////////////////////////////////////////////////////*/

/*M///////////////////////////////////////////////////////////////////////////////////////
// MIT License
//
// Copyright (c) 2022 OpenAI, Shantanu Jain
// Copyright (c) 2024 Andrej Karpathy
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////////////*/
#include <opencv2/core.hpp>

#include "encoding.hpp"
#include <cassert>
#include <regex>
#include <functional>
#include <iostream>
#include "unicode.hpp"
#include "utils.hpp"

namespace cv { namespace dnn { namespace tokenizer {

Encoding::Encoding(const std::string &name, 
                     const std::string &patStr,
                     const ByteVecRankMap &mergeableRanks,
                     const std::unordered_map<std::string, Rank>& specialTokens, 
                     int explicitNvocab) 
    : name_(name)
    , patStr_(patStr)
    , mergeableRanks_(mergeableRanks)
    , specialTokens_(specialTokens)
    , coreBPE_(mergeableRanks_, specialTokens_, patStr_) {
    // compute max token value
    Rank mrMax = 0;
    for (auto& kv : mergeableRanks_) 
        mrMax = std::max(mrMax, kv.second);
    Rank stMax = 0;
    for (auto& kv : specialTokens_) 
        stMax = std::max(stMax, kv.second);
    maxTokenValue_ = std::max(mrMax, stMax);
    if (explicitNvocab > 0) {
        std::cout << "mergeableRanks_.size(): " << mergeableRanks_.size() << std::endl;
        std::cout << "specialTokens_.size(): " << specialTokens_.size() << std::endl;
        std::cout << "explicitNvocab: " << explicitNvocab << std::endl;
        assert(static_cast<int>(mergeableRanks_.size() + specialTokens_.size()) == explicitNvocab);
        assert(maxTokenValue_ == static_cast<Rank>(explicitNvocab-1));
    }
}


std::vector<Rank> Encoding::encode(const std::string& text,
                                     const std::unordered_set<std::string>& allowedSpecial,
                                     const std::unordered_set<std::string>& disallowedSpecial) const {
    
    // 1. Handle "all" for allowed/disallowed (simulate with a special value or overload)
    std::unordered_set<std::string> allowed = allowedSpecial;
    std::unordered_set<std::string> disallowed = disallowedSpecial;

    // If allowedSpecial is "all", allow all special tokens
    // Example: if allowedSpecial contains "__ALL__", treat as all
    if (allowed.size() == 1 && allowed.count("__ALL__")) {
        for (const auto& kv : specialTokens_)
            allowed.insert(kv.first);
        allowed.erase("__ALL__");
    }

    if (disallowed.size() == 1 && disallowed.count("__ALL__")) {
        for (const auto& kv : specialTokens_) {
            if (!allowed.count(kv.first))
                disallowed.insert(kv.first);
        }
        disallowed.erase("__ALL__");
    }

    // disallow all special tokens if both sets are empty
    if (allowed.empty() && disallowed.empty()) {
        for (const auto& kv : specialTokens_)
            disallowed.insert(kv.first);
    }

    // If disallowed is empty, allow all text (even if it matches a special token)
    if (disallowed.empty()) {
        return coreBPE_.encode(text, allowed).first;
    }

    //  Regex check for disallowed special tokens
    std::string pattern;
    for (auto it = disallowed.begin(); it != disallowed.end(); ++it) {
        if (it != disallowed.begin()) pattern += "|";
        pattern += escape_regex(*it);
    }
    std::regex spec_re(pattern);
    std::smatch m;
    if (std::regex_search(text, m, spec_re)) {
        throw std::invalid_argument("Encountered disallowed special token: " + m.str());
    }
    
    return coreBPE_.encode(text, allowed).first;
}


std::string Encoding::decode(const std::vector<Rank>& tokens, const std::string& errors) const {
    auto opt_bytes = coreBPE_.decodeBytes(tokens);
    if (!opt_bytes) throw std::runtime_error("Invalid decode.");
    const ByteVec& bytes = *opt_bytes;

    // Convert bytes to std::string (UTF-8)
    std::string result(reinterpret_cast<const char*>(bytes.data()), bytes.size());

    // If strict, validate UTF-8
    if (errors == "strict") {
        // Simple UTF-8 validation
        size_t i = 0;
        while (i < result.size()) {
            unsigned char c = static_cast<unsigned char>(result[i]);
            size_t len = 0;
            if (c < 0x80) len = 1;
            else if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            else throw std::runtime_error("Invalid UTF-8 sequence in decoded bytes");
            if (i + len > result.size())
                throw std::runtime_error("Truncated UTF-8 sequence in decoded bytes");
            // Check continuation bytes
            for (size_t j = 1; j < len; ++j)
                if ((static_cast<unsigned char>(result[i + j]) & 0xC0) != 0x80)
                    throw std::runtime_error("Invalid UTF-8 continuation byte in decoded bytes");
            i += len;
        }
    }
    return result;
}

std::vector<std::uint8_t> Encoding::decodeBytes(const std::vector<Rank>& tokens) const {
    auto opt_bytes = coreBPE_.decodeBytes(tokens);
    if (!opt_bytes) throw std::runtime_error("Invalid decode.");
    return *opt_bytes;
}


Encoding getEncodingForCl100k_base(const std::string &name, const std::string& cl00k_case_file) {
    if (name != "cl100k_base")
        throw std::runtime_error("Wrong model name. This model is cl100k_base");
    
    auto bpeText = readFile(cl00k_case_file);
    ByteVecRankMap mergeableRanks;

    std::istringstream lines(bpeText);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string token_b64;
        int rank;
        if (!(iss >> token_b64 >> rank)) {
            throw std::runtime_error("Error parsing line: " + line);
        }
        // Decode base64 token 
        auto bytes = base64_decode(token_b64);
        ByteVec bv(bytes.begin(), bytes.end());
        mergeableRanks.emplace(std::move(bv), static_cast<Rank>(rank));
    }
    // Special tokens for cl100k_base
    std::unordered_map<std::string, Rank> specialTokens = {
        {"<|endoftext|>", 100257},
        {"<|fim_prefix|>", 100258},
        {"<|fim_middle|>", 100259},
        {"<|fim_suffix|>", 100260},
        {"<|endofprompt|>", 100276}
    };

    return Encoding(name, CL100K_BASE, std::move(mergeableRanks), std::move(specialTokens), -1);
}

Encoding getEncodingForCl100k_baseFromJSON_FS(const std::string &name,
                                                         const std::string &json_path)
{
    if (name != "cl100k_base")
        CV_Error(cv::Error::StsError, "Wrong model name. This model is cl100k_base");

    cv::FileStorage fs(json_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened())
        CV_Error(cv::Error::StsError, "Failed to open tokenizer.json: " + json_path);

    // model.vocab (HF-style)
    cv::FileNode model = fs["model"];
    if (model.empty()) CV_Error(cv::Error::StsError, "tokenizer.json missing 'model'");

    cv::FileNode vocab = model["vocab"];
    if (vocab.empty()) CV_Error(cv::Error::StsError, "tokenizer.json missing model.vocab");

    ByteVecRankMap mergeableRanks;
    mergeableRanks.reserve((size_t)vocab.size());
    int max_id = -1;

    auto token_to_bytes = [&](const std::string& token_utf8) -> std::vector<uint8_t> {
        ByteVec out;
        auto cps = unicode_cpts_from_utf8(token_utf8); 
        out.reserve(cps.size());
        for (uint32_t cp : cps) {
            const std::string one = unicode_cpt_to_utf8(cp);
            out.push_back(unicode_utf8_to_byte(one));    
        }
        return out;
    };

    for (cv::FileNodeIterator it = vocab.begin(); it != vocab.end(); ++it) {
        cv::FileNode val = *it;
        std::string token = val.name();    
        int id = (int)val;
        mergeableRanks.emplace(token_to_bytes(token), (Rank)id);
        if (id > max_id) max_id = id;
    }

    std::unordered_map<std::string, Rank> specialTokens;
    cv::FileNode added = fs["added_tokens"];
    if (!added.empty()) {
        for (auto it = added.begin(); it != added.end(); ++it) {
            cv::FileNode t = *it;
            bool special = false; t["special"] >> special;
            int id = -1;          t["id"]      >> id;
            std::string content;  t["content"] >> content;
            if (special && id >= 0 && !content.empty()) {
                specialTokens.emplace(content, (Rank)id);
                if (id > max_id) max_id = id;
            }
        }
    }

    for (int b = 0; b < 256; ++b) {
        ByteVec key{ (uint8_t)b };
        if (mergeableRanks.find(key) == mergeableRanks.end()) {
            std::ostringstream oss; oss << "Missing singleton byte token 0x" << std::hex << b;
            CV_Error(cv::Error::StsError, oss.str());
        }
    }

    const int explicitNVocab = -1; // We pass -1 so we dont need to check the vocab size. We only do this for gpt2.
    return Encoding(name, CL100K_BASE, std::move(mergeableRanks), std::move(specialTokens), explicitNVocab);
}


}}}