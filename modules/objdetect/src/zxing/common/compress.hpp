#pragma once
#include <string>
#include <map>


namespace zxing
{
#define COMPRESS_BASE 256
#define ARRAY_LEN 10000
class CompressTools
{
public:
    CompressTools();
    
    ~CompressTools();
    
    std::string Compress(const std::string &sText);
    
    std::string Revert(const std::string &sCode);
    
    bool CanBeCompress(const std::string &sText);
    
    bool CanBeRevert(const std::string &sText);
    
private:
    int Encode(int iBase, const std::string &sBefore, std::string &sAfter);
    int Decode(int iBase, const std::string &sBefore, std::string &sAfter);
    std::map<int, char> m_tIntToChar[COMPRESS_BASE];
    std::map<char, int> m_tCharToInt[COMPRESS_BASE];
    bool m_bSetFlag[COMPRESS_BASE];

    int SetMap(int iBase, const std::string &sKey);
};
}  // namespace zxing
