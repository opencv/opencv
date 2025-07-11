#include "utils.hpp"
#include <fstream>
#include <string>
#include <unordered_map>
#include <regex>

#include "core_bpe.hpp"

namespace cv { namespace dnn { namespace tokenizer {
    
// ------------------------------- JSON Parsing --------------------------------
void append_utf8(uint32_t codepoint, std::string& out) {
    if (codepoint <= 0x7F)          out.push_back(char(codepoint));
    else if (codepoint <= 0x7FF)    { out.push_back(char(0xC0 | (codepoint>>6)));
                                      out.push_back(char(0x80 | (codepoint & 0x3F))); }
    else if (codepoint <= 0xFFFF)   { out.push_back(char(0xE0 | (codepoint>>12)));
                                      out.push_back(char(0x80 | ((codepoint>>6)&0x3F)));
                                      out.push_back(char(0x80 | (codepoint & 0x3F))); }
    else {                           out.push_back(char(0xF0 | (codepoint>>18)));
                                      out.push_back(char(0x80 | ((codepoint>>12)&0x3F)));
                                      out.push_back(char(0x80 | ((codepoint>>6)&0x3F)));
                                      out.push_back(char(0x80 | (codepoint & 0x3F))); }
}

std::string unescape_json(const std::string& s) {
    std::string out;  out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        char c = s[i++];
        if (c != '\\') { out.push_back(c); continue; }

        char esc = s[i++];
        switch (esc) {
            case '"':  out.push_back('"');  break;
            case '\\': out.push_back('\\'); break;
            case '/':  out.push_back('/');  break;
            case 'b':  out.push_back('\b'); break;
            case 'f':  out.push_back('\f'); break;
            case 'n':  out.push_back('\n'); break;
            case 'r':  out.push_back('\r'); break;
            case 't':  out.push_back('\t'); break;
            case 'u': {
                uint32_t cp = std::stoul(s.substr(i,4), nullptr, 16);
                i += 4;
                if (cp <= 0xFF) {
                    // raw byte
                    out.push_back(static_cast<char>(cp));
                }
                else {
                    // real Unicode → emit UTF-8
                    if (0xD800 <= cp && cp <= 0xDBFF 
                        && s[i]=='\\' && s[i+1]=='u') {
                        uint32_t cp2 = std::stoul(s.substr(i+2,4),nullptr,16);
                        i += 6;
                        cp = 0x10000 + ((cp-0xD800)<<10) + (cp2-0xDC00);
                    }
                    append_utf8(cp, out);
                }
                break;
            }
            default:   out.push_back(esc); break; 
        }
    }
    return out;
}

/* Return mapping: token string (raw bytes)  -> rank */
std::unordered_map<std::string,int>
read_encoder_json(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("cannot open " + path);

    std::string blob((std::istreambuf_iterator<char>(ifs)),
                     std::istreambuf_iterator<char>());

    static const std::regex pair_re(
    R"PAIR("((?:\\.|[^"])*)"\s*:\s*(\d+))PAIR",
    std::regex::optimize);


    std::unordered_map<std::string,int> table;
    for (auto it = std::sregex_iterator(blob.begin(), blob.end(), pair_re);
         it != std::sregex_iterator(); ++it)
    {
        std::string key = unescape_json((*it)[1].str());
        int         id  = std::stoi((*it)[2].str());
        table.emplace(std::move(key), id);
    }
    return table;
}


ByteVec textToBytes(const std::string& txt) {
    ByteVec bytes;
    for (size_t i = 0; i < txt.size();) {
        unsigned char c = txt[i];
        if (c < 128) {
            // ASCII
            bytes.push_back(c);
            ++i;
        }
        else if ((c & 0xE0) == 0xC0 && i + 1 < txt.size()) {
            // two-byte sequence: maybe U+0120?
            uint32_t cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(txt[i+1]) & 0x3F);
            if (cp == 0x0120) {
                // U+0120 → single space byte in GPT-2
                bytes.push_back(static_cast<uint8_t>(' '));
            }
            else {
                throw std::runtime_error("Unsupported UTF-8 codepoint in BPE prefix");
            }
            i += 2;
        }
        else {
            throw std::runtime_error("Unsupported UTF-8 sequence in BPE token");
        }
    }
    return bytes;
}

}}}