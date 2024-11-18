// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_COMPRESS_HPP__
#define __ZXING_COMMON_COMPRESS_HPP__

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
    
    std::string compress(const std::string &sText);
    
    std::string revert(const std::string &sCode);
    
    bool canBeCompress(const std::string &sText);
    
    bool canBeRevert(const std::string &sText);
    
private:
    int encode(int iBase, const std::string &sBefore, std::string &sAfter);
    int decode(int iBase, const std::string &sBefore, std::string &sAfter);
    std::map<int, char> m_tIntToChar[COMPRESS_BASE];
    std::map<char, int> m_tCharToInt[COMPRESS_BASE];
    bool m_bSetFlag[COMPRESS_BASE];

    int SetMap(int iBase, const std::string &sKey);
};
}  // namespace zxing

#endif // __ZXING_COMMON_COMPRESS_HPP__