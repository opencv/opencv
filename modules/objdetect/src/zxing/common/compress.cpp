#include "compress.hpp"

#include <assert.h>
#include <sys/types.h>
#include <string.h>

#include <iostream>
#include <cstdio>
#include <algorithm>

namespace zxing
{
const int BEFORE_BASE = 45;
const int AFTER_BASE = 82;
CompressTools::CompressTools()
{
    SetMap(BEFORE_BASE, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:");
    SetMap(AFTER_BASE, "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#$&'()*+,/:;=?@-._~");
}

CompressTools::~CompressTools()
{
}

bool CompressTools::CanBeCompress(const std::string &sText)
{
    for (size_t i = 0; i < sText.size(); ++i)
    {
        if (m_tCharToInt[BEFORE_BASE].find(sText[i]) == m_tCharToInt[BEFORE_BASE].end())
            return false;
    }
    return true;
}

bool CompressTools::CanBeRevert(const std::string &sText)
{
    for (size_t i = 0; i < sText.size(); ++i)
    {
        if (m_tCharToInt[AFTER_BASE].find(sText[i]) == m_tCharToInt[AFTER_BASE].end())
            return false;
    }
    return true;
}


int CompressTools::SetMap(int iBase, const std::string &sKey)
{
    if (static_cast<int>(sKey.size()) != iBase) return -1;
    for (int i = 0; i < iBase; ++i)
    {
        for (int j = i + 1; j < iBase; ++j)
        {
            if (sKey[i] == sKey[j])
            {
                return -2;
            }
        }
    }
    for (int i = 0; i < iBase; ++i)
    {
        m_tIntToChar[iBase][i] = sKey[i];
        m_tCharToInt[iBase][sKey[i]] = i;
    }
    m_bSetFlag[iBase] = true;
    return 0;
}

int CompressTools::Encode(int iBase, const std::string &sBefore, std::string &sAfter)
{
    if (!m_bSetFlag[iBase]) return 1;
    
    int aiBefore[ARRAY_LEN], iBeforeLen = sBefore.size();
    int aiAfter[ARRAY_LEN], iAfterLen = 0;
    for (size_t i = 0; i < sBefore.size(); ++i)
    {
        if (m_tCharToInt[iBase].find(sBefore[i]) == m_tCharToInt[iBase].end()) return -1;
        aiBefore[i] = m_tCharToInt[iBase][sBefore[i]];
    }
    int iBaseSum = 0;
    bool bZeroFlag = false;
    int iZeroCount = 0;
    for (int i = 0; i < iBeforeLen; ++i)
    {
        if (aiBefore[i] == 0)
        {
            if (!bZeroFlag)
            {
                iZeroCount++;
                continue;
            }
        }
        else bZeroFlag = true;
        iBaseSum *= iBase;
        iBaseSum += aiBefore[i];
        while (iBaseSum >= COMPRESS_BASE)
        {
            aiAfter[iAfterLen++] = iBaseSum % COMPRESS_BASE;
            iBaseSum /= COMPRESS_BASE;
        }
    }
    if (iZeroCount > 255) return -2;  // leading zero is not more than 255(sizeof (char))
    if (iBaseSum) aiAfter[iAfterLen++] = iBaseSum;
    sAfter.resize(iAfterLen + 1);
    for (int i = 0; i < iAfterLen; ++i)
    {
        sAfter[i] = aiAfter[i];
    }
    sAfter[iAfterLen] = iZeroCount;
    reverse(sAfter.begin(), sAfter.end());
    return 0;
}

int CompressTools::Decode(int iBase, const std::string &sBefore, std::string &sAfter)
{
    if (!m_bSetFlag[iBase]) return 1;
    
    int aiBefore[ARRAY_LEN], iBeforeLen = sBefore.size();
    int aiAfter[ARRAY_LEN], iAfterLen = 0;
    for (int i = 0; i < iBeforeLen; ++i)
    {
        aiBefore[i] = (unsigned char)sBefore[i];
    }
    
    int iLastSum = 0;
    int iZeroCount = aiBefore[0];
    if (iBeforeLen > 1)
    {
        iLastSum = aiBefore[1];
        if (iLastSum >= iBase)
        {
            while (iLastSum >= iBase)
            {
                aiAfter[iAfterLen++] = iLastSum % iBase;
                iLastSum /= iBase;
            }
        }
    }
    for (int i = 2; i < iBeforeLen; ++i)
    {
        iLastSum = iLastSum * COMPRESS_BASE + aiBefore[i];
        while (iLastSum >= iBase)
        {
            aiAfter[iAfterLen++] = iLastSum % iBase;
            iLastSum /= iBase;
        }
    }
    if (iLastSum) aiAfter[iAfterLen++] = iLastSum;
    
    sAfter.resize(iAfterLen + iZeroCount);
    for (int i = 0; i < iAfterLen; ++i)
    {
        if (m_tIntToChar[iBase].find(aiAfter[i]) == m_tIntToChar[iBase].end()) return -1;
        sAfter[i] = m_tIntToChar[iBase][aiAfter[i]];
    }
    for (int i = 0; i < iZeroCount; ++i)
    {
        sAfter[i + iAfterLen] = m_tIntToChar[iBase][0];
    }
    reverse(sAfter.begin(), sAfter.end());
    return 0;
}

std::string CompressTools::Compress(const std::string &sText)
{
    std::string sCode;
    int iRet = Encode(BEFORE_BASE, sText, sCode);
    if (iRet)
    {
        printf("compress.encode err! ret = %d\n", iRet);
        return "";
    }
    std::string sNewText;
    iRet = Decode(AFTER_BASE, sCode, sNewText);
    if (iRet)
    {
        printf("compress.decode err! ret = %d\n", iRet);
        return "";
    }
    return sNewText;
}

std::string CompressTools::Revert(const std::string &sCode)
{
    std::string sText;
    int iRet = Encode(AFTER_BASE, sCode, sText);
    if (iRet)
    {
        printf("revert.encode err! ret = %d\n", iRet);
        return "";
    }
    std::string sNewCode;
    iRet = Decode(BEFORE_BASE, sText, sNewCode);
    if (iRet)
    {
        printf("revert.decode err! ret = %d\n", iRet);
        return "";
    }
    return sNewCode;
}
}  // namespace zxing
