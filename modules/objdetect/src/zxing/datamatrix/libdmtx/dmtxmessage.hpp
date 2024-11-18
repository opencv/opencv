// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

//
//  dmtxmessage.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/5.
//

#ifndef __ZXING_DATAMATRIX_LIBDMTX_DMTXMESSAGE_HPP__
#define __ZXING_DATAMATRIX_LIBDMTX_DMTXMESSAGE_HPP__

#include <stdio.h>
#include <vector>
#include "common.hpp"

namespace dmtx {

class DmtxMessage {
public:
    DmtxMessage() {}
    ~DmtxMessage();
    
    int init(int sizeIdx, int symbolFormat);
    
    unsigned int decodeDataStream(int sizeIdx, unsigned char *outputStart);
    
private:
    int getEncodationScheme(unsigned char cw);
    DmtxBoolean validOutputWord(int value);
    unsigned int pushOutputWord(int value);
    unsigned int pushOutputC40TextWord(C40TextState *state, int value);
    unsigned int pushOutputMacroHeader(int macroType);
    void pushOutputMacroTrailer();
    unsigned char *decodeSchemeAscii(unsigned char *ptr, unsigned char *dataEnd);
    unsigned char *decodeSchemeC40Text(unsigned char *ptr, unsigned char *dataEnd, DmtxScheme encScheme);
    unsigned char *decodeSchemeX12(unsigned char *ptr, unsigned char *dataEnd);
    unsigned char *decodeSchemeEdifact(unsigned char *ptr, unsigned char *dataEnd);
    unsigned char *decodeSchemeBase256(unsigned char *ptr, unsigned char *dataEnd);
    unsigned char unRandomize255State(unsigned char value, int idx);
    
public:
    size_t          arraySize;     /* mappingRows * mappingCols */
    size_t          codeSize;      /* Size of encoded data (data words + error words) */
    size_t          outputSize;    /* Size of buffer used to hold decoded data */
    int             outputIdx;     /* Internal index used to store output progress */
    int             padCount;
    int             fnc1;          /* Character to represent FNC1, or DmtxUndefined */
    unsigned char  *array;         /* Pointer to internal representation of Data Matrix modules */
    unsigned char  *code;          /* Pointer to internal storage of code words (data and error) */
    unsigned char  *output;        /* Pointer to internal storage of decoded output */
    
    std::vector<DmtxPixelLoc> points;
};

}  // namespace dmtx

#endif // __ZXING_DATAMATRIX_LIBDMTX_DMTXMESSAGE_HPP__
