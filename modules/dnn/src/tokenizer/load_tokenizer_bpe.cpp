// load_tokenizer_bpe.cpp — helper routines to lazily fetch & verify remote BPE tables
// -----------------------------------------------------------------------------
// These utilities let our OpenCV tokenizer load either local or remote
// copies of encoder.json and vocab.bpe (or any other BPE table) while
// guaranteeing SHA‑256 integrity. 
//
// Dependencies (all are header‑only or widely available system libraries):
// If Opencv has other methods to deal with this without these depedencies 
// will change later. 
//   • <curl/curl.h>     — HTTP GET (Linux/macOS: sudo apt/brew install curl)
//   • <openssl/sha.h>   — SHA‑256 (ships with OpenSSL)
//   • nlohmann/json.hpp — tiny JSON parser (single‑header, BSD‑3)
//
// Build flags example (add to modules/dnn/CMakeLists.txt):
//   find_package(CURL REQUIRED)
//   find_package(OpenSSL REQUIRED)
//   target_link_libraries(opencv_dnn PRIVATE CURL::libcurl OpenSSL::Crypto)
// -----------------------------------------------------------------------------

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <filesystem>
#include <iomanip>
#include <cctype>

#include <curl/curl.h>
#include <openssl/sha.h>
#include "nlohmann/json.hpp"

#include "core_bpe.hpp"   
#include "load_tokenizer_bpe.hpp"


namespace cv { namespace dnn { namespace tokenizer {

static inline bool isUrl(const std::string &s) {
    return s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0;
}

static std::string sha256(const std::string& data) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(data.data()), data.size(), hash);
    std::ostringstream oss;
    for (unsigned char b : hash) oss << std::hex << std::setw(2) << std::setfill('0') << (int)b;
    return oss.str();
}

static size_t curlWrite(void *ptr, size_t size, size_t nmemb, void *userData) {
    auto *buf = static_cast<std::string*>(userData);
    buf->append(static_cast<char*>(ptr), size * nmemb);
    return size * nmemb;
}

std::string readFileOrUrl(const std::string& pathOrUrl,
                          const std::string& expextedHash) {
                            
    std::string contents;
    if (isUrl(pathOrUrl)) {
        CURL *curl = curl_easy_init();
        if (!curl) throw std::runtime_error("curl_easy_init() failed");
        curl_easy_setopt(curl, CURLOPT_URL, pathOrUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWrite);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &contents);
        auto res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        if (res != CURLE_OK) 
            throw std::runtime_error("HTTP GET failed for " + pathOrUrl + ": " + curl_easy_strerror(res));
    } else {
        std::fstream in(pathOrUrl, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open file: " + pathOrUrl);
        std::ostringstream oss;
        oss << in.rdbuf();
        contents = oss.str();
    }

    if (!expextedHash.empty()) {
        std::string got = sha256(contents);
        for (auto &c : got) c = static_cast<char>(std::tolower(c));
        std::string expect = expextedHash;
        for (auto& c : expect) c = static_cast<char>(std::tolower(c));
        if (got != expect)
            throw std::runtime_error("SHA‑256 mismatch for " + pathOrUrl + " (expected " + expect + ", got " + got + ")");
    }
    return contents;
}

static inline int b64Index(char c) {
    // Ignores whitespace
    if ('A' <= c && c <= 'Z') return c - 'A';
    if ('a' <= c && c <= 'z') return c - 'a' + 26;
    if ('0' <= c && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static ByteVec base64Decode(const std::string& in) {
    ByteVec out;
    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (std::isspace(c)) continue;
        if (c == '=') break;
        int idx = b64Index(c);
        if (idx < 0) throw std::runtime_error("Invalid base64 character");
        val = (val << 6) + idx;
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<std::uint8_t>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// Parse a “.tiktoken” BPE file into mergeable_ranks.
ByteVecRankMap loadTokenizerBPE(const std::string& filePath,
                                const std::string& expectedHash) {

    std::string contents = readFileOrUrl(filePath, expectedHash);
    std::istringstream ls(contents);
    std::string line;

    if (std::getline(ls, line) && line.rfind("#", 0) != 0) {
        ls.clear();
        ls.seekg(0);
    }

    ByteVecRankMap ranks;
    while (std::getline(ls, line)) {
        if (line.empty()) continue;
        auto sep = line.find(' ');
        if (sep == std::string::npos) 
            throw std::runtime_error("Malformed .bpe line: " + line);
        std::string tokenB64 = line.substr(0, sep);
        int rank = std::stoi(line.substr(sep + 1));
        ranks.emplace(base64Decode(tokenB64), static_cast<Rank>(rank));
    }
    return ranks;
}

// ------------------------------------------------------------
//    • Reads encoder.json (token -> rank) and vocab.bpe (merge table)
//    • Converts the GPT‑2 printable‑ASCII trick back to raw bytes
//    • Returns ByteVecRankMap where all 50257 entries map to Rank
// ------------------------------------------------------------
ByteVecRankMap dataGymToMergeableBpeRanks(const std::string& vocabBpeFile,
                                          const std::string& encoderJsonFile,
                                          const std::string& vocabBpeHash,
                                          const std::string& encoderJsonHash) {
    
    // load and decode encoder.json
    std::string encJson = readFileOrUrl(encoderJsonFile, encoderJsonHash);
    std::unordered_map<std::string, int> encoderMap = nlohmann::json::parse(encJson).get<std::unordered_map<std::string, int>>();

    //    Build reverse map token string -> ByteVec (using Data‑Gym escaping)
    //    GPT‑2 training treated bytes 0‑255 as 512 printable chars.  First 188 are
    //    printable ASCII (minus space), the rest are mapped to U+0100‑U+01FF.
    std::unordered_map<std::string, ByteVec> str2bytes;
    for (int b = 0; b < 256; ++b) {
        if (std::isprint(static_cast<unsigned char>(b)) && b != ' ') {
            str2bytes[std::string(1, static_cast<char>(b))] = { static_cast<uint8_t>(b) };
        }
    }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!(std::isprint(static_cast<char>(b)) && b != ' ')) {
            char ch = static_cast<char>(256 + n);
            str2bytes[std::string(1, ch)] = { static_cast<uint8_t>(b) };
            ++n;
        }
    }

    // Parse vocab.bpe merge list
    std::string bpeTxt = readFileOrUrl(vocabBpeFile, vocabBpeHash);
    std::istringstream ls(bpeTxt);
    std::string line;
    // skip header line
    std::getline(ls, line);

    ByteVecRankMap mergeable;

    for (const auto& kv : encoderMap) {
        auto it = str2bytes.find(kv.first);
        if (it == str2bytes.end())
            throw std::runtime_error("Unknown byte in encoder.json: " + kv.first);
        mergeable.emplace(it->second, static_cast<Rank>(kv.second));
    }

    Rank rank = static_cast<Rank>(encoderMap.size());
    while (std::getline(ls, line)) {
        if (line.empty()) continue;
        std::istringstream pairStream(line);
        std::string s1, s2;
        pairStream >> s1 >> s2;
        if (s1.empty() || s2.empty()) 
            throw std::runtime_error("Malformed merge line: " + line);
        ByteVec bytes;
        const auto it1 = str2bytes.find(s1); 
        if (it1 == str2bytes.end()) 
            throw std::runtime_error("Token missing: " + s1);
        const auto it2 = str2bytes.find(s2);
        if (it2 == str2bytes.end())
            throw std::runtime_error("Token missing: " + s2);
        bytes = it1->second;
        bytes.insert(bytes.end(), it2->second.begin(), it2->second.end());
        mergeable.emplace(std::move(bytes), rank++);
    }
    return mergeable;
}

}}}