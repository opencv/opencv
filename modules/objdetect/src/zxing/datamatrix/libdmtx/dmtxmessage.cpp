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
//  dmtxmessage.cpp
//  test_dm
//
//  Created by wechatcv on 2022/5/5.
//

#include "dmtxmessage.hpp"
#include "dmtxsymbol.hpp"
#include "common.hpp"

#include <limits.h>
#include "stdlib.h"

namespace dmtx {

DmtxMessage::~DmtxMessage()
{
    if (this->array != NULL)
        free(this->array);
    
    if (this->code != NULL)
        free(this->code);
    
    if (this->output != NULL)
        free(this->output);
}

int DmtxMessage::init(int sizeIdx, int symbolFormat)
{
    int mappingRows, mappingCols;
    
    if (symbolFormat != DmtxFormatMatrix && symbolFormat != DmtxFormatMosaic)
        return -1;
    
    mappingRows = dmtxGetSymbolAttribute(DmtxSymAttribMappingMatrixRows, sizeIdx);
    mappingCols = dmtxGetSymbolAttribute(DmtxSymAttribMappingMatrixCols, sizeIdx);
    
    this->arraySize = sizeof(unsigned char) * mappingRows * mappingCols;
    
    this->array = (unsigned char *)calloc(1, this->arraySize);
    if (this->array == NULL)
    {
        return -1;
    }
    
    this->codeSize = sizeof(unsigned char) *
    dmtxGetSymbolAttribute(DmtxSymAttribSymbolDataWords, sizeIdx) +
    dmtxGetSymbolAttribute(DmtxSymAttribSymbolErrorWords, sizeIdx);
    
    if (symbolFormat == DmtxFormatMosaic)
        this->codeSize *= 3;
    
    this->code = (unsigned char *)calloc(this->codeSize, sizeof(unsigned char));
    if (this->code == NULL)
    {
        return -1;
    }
    
    /* XXX not sure if this is the right place or even the right approach.
     Trying to allocate memory for the decoded data stream and will
     initially assume that decoded data will not be larger than 2x encoded data */
    this->outputSize = sizeof(unsigned char) * this->codeSize * 10;
    this->output = (unsigned char *)calloc(this->outputSize, sizeof(unsigned char));
    if (this->output == NULL)
    {
        return -1;
    }
    
    return 0;
}

unsigned int DmtxMessage::decodeDataStream(int sizeIdx, unsigned char *outputStart)
{
    DmtxBoolean macro = DmtxFalse;
    DmtxScheme encScheme;
    unsigned char *ptr, *dataEnd;
    
    this->output = (outputStart == NULL) ? this->output : outputStart;
    this->outputIdx = 0;
    
    ptr = this->code;
    dataEnd = ptr + dmtxGetSymbolAttribute(DmtxSymAttribSymbolDataWords, sizeIdx);

    if (ptr == NULL)
        return DmtxFail;

    /* Print macro header if first codeword triggers it */
    if (*ptr == DmtxValue05Macro || *ptr == DmtxValue06Macro) {
        pushOutputMacroHeader(*ptr);
        macro = DmtxTrue;
    }
    
    if (ptr == NULL)
        return DmtxFail;
    
    while (ptr < dataEnd) {
        encScheme = (DmtxScheme)getEncodationScheme(*ptr);
        if (encScheme != DmtxSchemeAscii)
            ptr++;
        
        if (ptr == NULL)
            return DmtxFail;
        
        switch (encScheme) {
            case DmtxSchemeAscii:
                ptr = decodeSchemeAscii(ptr, dataEnd);
                break;
            case DmtxSchemeC40:
            case DmtxSchemeText:
                ptr = decodeSchemeC40Text(ptr, dataEnd, encScheme);
                break;
            case DmtxSchemeX12:
                ptr = decodeSchemeX12(ptr, dataEnd);
                break;
            case DmtxSchemeEdifact:
                ptr = decodeSchemeEdifact(ptr, dataEnd);
                break;
            case DmtxSchemeBase256:
                ptr = decodeSchemeBase256(ptr, dataEnd);
                break;
            default:
                /* error */
                break;
        }
        
        if (ptr == NULL)
            return DmtxFail;
    }
    
    /* Print macro trailer if required */
    if (macro == DmtxTrue)
        pushOutputMacroTrailer();
    
    return DmtxPass;
}

int DmtxMessage::getEncodationScheme(unsigned char cw)
{
    DmtxScheme encScheme;
    
    switch (cw) {
        case DmtxValueC40Latch:
            encScheme = DmtxSchemeC40;
            break;
        case DmtxValueTextLatch:
            encScheme = DmtxSchemeText;
            break;
        case DmtxValueX12Latch:
            encScheme = DmtxSchemeX12;
            break;
        case DmtxValueEdifactLatch:
            encScheme = DmtxSchemeEdifact;
            break;
        case DmtxValueBase256Latch:
            encScheme = DmtxSchemeBase256;
            break;
        default:
            encScheme = DmtxSchemeAscii;
            break;
    }
    
    return encScheme;
}

unsigned int DmtxMessage::pushOutputWord(int value)
{
    if (value < 0 || value >= 256) return DmtxFail;
    
    this->output[this->outputIdx++] = (unsigned char)value;
    return DmtxPass;
}

DmtxBoolean DmtxMessage::validOutputWord(int value)
{
    return (value >= 0 && value < 256) ? DmtxTrue : DmtxFalse;
}

unsigned int DmtxMessage::pushOutputC40TextWord(C40TextState *state, int value)
{
    if (value < 0 || value >= 256) return DmtxFail;
    
    this->output[this->outputIdx] = (unsigned char)value;
    
    if (state->upperShift == DmtxTrue)
    {
        if (value >= 128) return DmtxFail;
        this->output[this->outputIdx] += 128;
    }
    
    this->outputIdx++;
    
    state->shift = DmtxC40TextBasicSet;
    state->upperShift = DmtxFalse;
    
    return DmtxPass;
}

unsigned int DmtxMessage::pushOutputMacroHeader(int macroType)
{
    pushOutputWord('[');
    pushOutputWord(')');
    pushOutputWord('>');
    pushOutputWord(30); /* ASCII RS */
    pushOutputWord('0');
    
    if (macroType != DmtxValue05Macro && macroType != DmtxValue06Macro)
        return DmtxFail;
    if (macroType == DmtxValue05Macro)
        pushOutputWord('5');
    else
        pushOutputWord('6');
    
    pushOutputWord(29); /* ASCII GS */
    return DmtxPass;
}

void DmtxMessage::pushOutputMacroTrailer()
{
    pushOutputWord(30); /* ASCII RS */
    pushOutputWord(4);  /* ASCII EOT */
}

unsigned char * DmtxMessage::decodeSchemeAscii(unsigned char *ptr, unsigned char *dataEnd)
{
    int upperShift = DmtxFalse;
    
    while (ptr < dataEnd) {
        int codeword = static_cast<int>(*ptr);
        
        if (getEncodationScheme(*ptr) != DmtxSchemeAscii)
            return ptr;
        else
            ptr++;
        
        if (upperShift == DmtxTrue)
        {
            int pushword = codeword + 127;
            if (validOutputWord(pushword) != DmtxTrue)
                return NULL;
            pushOutputWord(pushword);
            upperShift = DmtxFalse;
        }
        else if (codeword == DmtxValueAsciiUpperShift)
        {
            upperShift = DmtxTrue;
        }
        else if (codeword == DmtxValueAsciiPad)
        {
            if (dataEnd < ptr) return NULL;
            if (dataEnd - ptr > INT_MAX) return NULL;
            this->padCount = static_cast<int>(dataEnd - ptr);
            return dataEnd;
        }
        else if (codeword == 0 || codeword >= 242)
        {
            return ptr;
        }
        else if (codeword <= 128)
        {
            if (pushOutputWord(codeword - 1) == DmtxFail)
                return NULL;
        }
        else if (codeword <= 229)
        {
            int digits = codeword - 130;
            if (pushOutputWord(digits/10 + '0') == DmtxFail)
                return NULL;
            if (pushOutputWord(digits - (digits/10)*10 + '0') == DmtxFail)
                return NULL;
        }
        else if (codeword == DmtxValueFNC1)
        {
            if (this->fnc1 != DmtxUndefined)
            {
                int pushword = this->fnc1;
                if (validOutputWord(pushword) != DmtxTrue)
                    return NULL;
                pushOutputWord(pushword);
            }
        }
    }
    
    return ptr;
}

unsigned char * DmtxMessage::decodeSchemeC40Text(unsigned char *ptr, unsigned char *dataEnd, DmtxScheme encScheme)
{
    int i;
    int packed;
    int c40Values[3];
    C40TextState state;
    
    state.shift = DmtxC40TextBasicSet;
    state.upperShift = DmtxFalse;
    
    if (encScheme != DmtxSchemeC40 && encScheme != DmtxSchemeText) return NULL;
    
    /* Unlatch is implied if only one codeword remains */
    if (dataEnd - ptr < 2)
        return ptr;
    
    while (ptr < dataEnd) {
        /* FIXME Also check that ptr+1 is safe to access */
        packed = (*ptr << 8) | *(ptr+1);
        c40Values[0] = ((packed - 1)/1600);
        c40Values[1] = ((packed - 1)/40) % 40;
        c40Values[2] =  (packed - 1) % 40;
        ptr += 2;
        
        for (i = 0; i < 3; i++) {
            if (state.shift == DmtxC40TextBasicSet)
            { /* Basic set */
                if (c40Values[i] <= 2)
                {
                    state.shift = c40Values[i] + 1;
                }
                else if (c40Values[i] == 3)
                {
                    if (pushOutputC40TextWord(&state, ' ') == DmtxFail)
                        return NULL;
                }
                else if (c40Values[i] <= 13)
                {
                    if (pushOutputC40TextWord(&state, c40Values[i] - 13 + '9') == DmtxFail)
                        return NULL; /* 0-9 */
                }
                else if (c40Values[i] <= 39)
                {
                    if (encScheme == DmtxSchemeC40)
                    {
                        if (pushOutputC40TextWord(&state, c40Values[i] - 39 + 'Z') == DmtxFail)
                            return NULL; /* A-Z */
                    }
                    else if (encScheme == DmtxSchemeText)
                    {
                        if (pushOutputC40TextWord(&state, c40Values[i] - 39 + 'z') == DmtxFail)
                            return NULL; /* a-z */
                    }
                }
            }
            else if (state.shift == DmtxC40TextShift1)
            { /* Shift 1 set */
                if (pushOutputC40TextWord(&state, c40Values[i]) == DmtxFail)
                    return NULL; /* ASCII 0 - 31 */
            }
            else if (state.shift == DmtxC40TextShift2)
            { /* Shift 2 set */
                if (c40Values[i] <= 14)
                {
                    if (pushOutputC40TextWord(&state, c40Values[i] + 33) == DmtxFail)
                        return NULL;  /* ASCII 33 - 47 */
                }
                else if (c40Values[i] <= 21)
                {
                    if (pushOutputC40TextWord(&state, c40Values[i] + 43) == DmtxFail)
                        return NULL; /* ASCII 58 - 64 */
                }
                else if (c40Values[i] <= 26)
                {
                    if (pushOutputC40TextWord(&state, c40Values[i] + 69) == DmtxFail)
                        return NULL; /* ASCII 91 - 95 */
                }
                else if (c40Values[i] == 27)
                {
                    if (this->fnc1 != DmtxUndefined)
                    {
                        if (pushOutputC40TextWord(&state, this->fnc1) == DmtxFail)
                            return NULL;
                    }
                }
                else if (c40Values[i] == 30)
                {
                    state.upperShift = DmtxTrue;
                    state.shift = DmtxC40TextBasicSet;
                }
            }
            else if (state.shift == DmtxC40TextShift3)
            { /* Shift 3 set */
                if (encScheme == DmtxSchemeC40)
                {
                    if (pushOutputC40TextWord(&state, c40Values[i] + 96) == DmtxFail)
                        return NULL;
                }
                else if (encScheme == DmtxSchemeText)
                {
                    if (c40Values[i] == 0)
                    {
                        if (pushOutputC40TextWord(&state, c40Values[i] + 96) == DmtxFail)
                            return NULL;
                    }
                    else if (c40Values[i] <= 26)
                    {
                        if (pushOutputC40TextWord(&state, c40Values[i] - 26 + 'Z') == DmtxFail)
                            return NULL; /* A-Z */
                    }
                    else
                    {
                        if (pushOutputC40TextWord(&state, c40Values[i] - 31 + 127) == DmtxFail)
                            return NULL; /* { | } ~ DEL */
                    }
                }
            }
        }
        
        /* Unlatch if codeword 254 follows 2 codewords in C40/Text encodation */
        if (*ptr == DmtxValueCTXUnlatch)
            return ptr + 1;
        
        /* Unlatch is implied if only one codeword remains */
        if (dataEnd - ptr < 2)
            return ptr;
    }
    
    return ptr;
}

unsigned char * DmtxMessage::decodeSchemeX12(unsigned char *ptr, unsigned char *dataEnd)
{
    int i;
    int packed;
    int x12Values[3];
    
    /* Unlatch is implied if only one codeword remains */
    if (dataEnd - ptr < 2)
        return ptr;
    
    while (ptr < dataEnd) {
        /* FIXME Also check that ptr+1 is safe to access */
        packed = (*ptr << 8) | *(ptr+1);
        x12Values[0] = ((packed - 1)/1600);
        x12Values[1] = ((packed - 1)/40) % 40;
        x12Values[2] =  (packed - 1) % 40;
        ptr += 2;
        
        for (i = 0; i < 3; i++) {
            if (x12Values[i] == 0)
                pushOutputWord(13);
            else if (x12Values[i] == 1)
                pushOutputWord(42);
            else if (x12Values[i] == 2)
                pushOutputWord(62);
            else if (x12Values[i] == 3)
                pushOutputWord(32);
            else if (x12Values[i] <= 13)
            {
                if (pushOutputWord(x12Values[i] + 44) == DmtxFail)
                    return NULL;
            }
            else if (x12Values[i] <= 90)
            {
                if (pushOutputWord(x12Values[i] + 51) == DmtxFail)
                    return NULL;
            }
        }
        
        /* Unlatch if codeword 254 follows 2 codewords in C40/Text encodation */
        if (*ptr == DmtxValueCTXUnlatch)
            return ptr + 1;
        
        /* Unlatch is implied if only one codeword remains */
        if (dataEnd - ptr < 2)
            return ptr;
    }
    
    return ptr;
}

unsigned char * DmtxMessage::decodeSchemeEdifact(unsigned char *ptr, unsigned char *dataEnd)
{
    int i;
    unsigned char unpacked[4];
    
    /* Unlatch is implied if fewer than 3 codewords remain */
    if (dataEnd - ptr < 3)
        return ptr;
    
    while (ptr < dataEnd) {
        /* FIXME Also check that ptr+2 is safe to access -- shouldn't be a
         problem because I'm guessing you can guarantee there will always
         be at least 3 error codewords */
        unpacked[0] = (*ptr & 0xfc) >> 2;
        unpacked[1] = (*ptr & 0x03) << 4 | (*(ptr+1) & 0xf0) >> 4;
        unpacked[2] = (*(ptr+1) & 0x0f) << 2 | (*(ptr+2) & 0xc0) >> 6;
        unpacked[3] = *(ptr+2) & 0x3f;
        
        for (i = 0; i < 4; i++) {
            
            /* Advance input ptr (4th value comes from already-read 3rd byte) */
            if (i < 3)
                ptr++;
            
            /* Test for unlatch condition */
            if (unpacked[i] == DmtxValueEdifactUnlatch)
            {
                if (this->output[this->outputIdx] != 0) return NULL; /* XXX dirty why? */
                return ptr;
            }
            
            if (pushOutputWord(unpacked[i] ^ (((unpacked[i] & 0x20) ^ 0x20) << 1)) == DmtxFail)
                return NULL;
        }
        
        /* Unlatch is implied if fewer than 3 codewords remain */
        if (dataEnd - ptr < 3)
            return ptr;
    }
    
    return ptr;
}

unsigned char * DmtxMessage::decodeSchemeBase256(unsigned char *ptr, unsigned char *dataEnd)
{
    int d0, d1;
    int idx;
    unsigned char *ptrEnd;
    
    /* Find positional index used for unrandomizing */
    if (ptr + 1 < this->code) return NULL;
    if (ptr + 1 - this->code > INT_MAX) return NULL;
    idx = static_cast<int>(ptr + 1 - this->code);
    
    d0 = unRandomize255State(*(ptr++), idx++);
    if (d0 == 0) {
        ptrEnd = dataEnd;
    }
    else if (d0 <= 249)
    {
        ptrEnd = ptr + d0;
    }
    else
    {
        d1 = unRandomize255State(*(ptr++), idx++);
        ptrEnd = ptr + (d0 - 249) * 250 + d1;
    }
    
    if (ptrEnd > dataEnd)
        return NULL;
    
    while (ptr < ptrEnd) {
        if (pushOutputWord(unRandomize255State(*(ptr++), idx++)) == DmtxFail)
            return NULL;
    }
    
    return ptr;
}

unsigned char DmtxMessage::unRandomize255State(unsigned char value, int idx)
{
    int pseudoRandom;
    int tmp;
    
    pseudoRandom = ((149 * idx) % 255) + 1;
    tmp = value - pseudoRandom;
    if (tmp < 0)
        tmp += 256;
    
    return (unsigned char)tmp;
}

}  // namespace dmtx
