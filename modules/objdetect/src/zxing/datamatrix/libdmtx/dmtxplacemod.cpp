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
//  dmtxplacemod.cpp
//  test_dm
//
//  Created by wechatcv on 2022/5/7.
//

#include "dmtxplacemod.hpp"
#include "dmtxsymbol.hpp"

#include "common.hpp"

namespace dmtx {

int modulePlacementEcc200(unsigned char *modules, unsigned char *codewords, int sizeIdx, int moduleOnColor)
{
    int row, col, chr;
    int mappingRows, mappingCols;
    
    // assert(moduleOnColor & (DmtxModuleOnRed | DmtxModuleOnGreen | DmtxModuleOnBlue));
    if (!(moduleOnColor & (DmtxModuleOnRed | DmtxModuleOnGreen | DmtxModuleOnBlue)))
        return -1;
    
    mappingRows = dmtxGetSymbolAttribute(DmtxSymAttribMappingMatrixRows, sizeIdx);
    mappingCols = dmtxGetSymbolAttribute(DmtxSymAttribMappingMatrixCols, sizeIdx);
    
    /* Start in the nominal location for the 8th bit of the first character */
    chr = 0;
    row = 4;
    col = 0;
    
    do {
        /* Repeatedly first check for one of the special corner cases */
        if ((row == mappingRows) && (col == 0))
            patternShapeSpecial1(modules, mappingRows, mappingCols, &(codewords[chr++]), moduleOnColor);
        else if ((row == mappingRows-2) && (col == 0) && (mappingCols%4 != 0))
            patternShapeSpecial2(modules, mappingRows, mappingCols, &(codewords[chr++]), moduleOnColor);
        else if ((row == mappingRows-2) && (col == 0) && (mappingCols%8 == 4))
            patternShapeSpecial3(modules, mappingRows, mappingCols, &(codewords[chr++]), moduleOnColor);
        else if ((row == mappingRows+4) && (col == 2) && (mappingCols%8 == 0))
            patternShapeSpecial4(modules, mappingRows, mappingCols, &(codewords[chr++]), moduleOnColor);
        
        /* Sweep upward diagonally, inserting successive characters */
        do {
            if ((row < mappingRows) && (col >= 0) &&
               !(modules[row*mappingCols+col] & DmtxModuleVisited))
                patternShapeStandard(modules, mappingRows, mappingCols, row, col, &(codewords[chr++]), moduleOnColor);
            row -= 2;
            col += 2;
        } while ((row >= 0) && (col < mappingCols));
        row += 1;
        col += 3;
        
        /* Sweep downward diagonally, inserting successive characters */
        do {
            if ((row >= 0) && (col < mappingCols) &&
               !(modules[row*mappingCols+col] & DmtxModuleVisited))
                patternShapeStandard(modules, mappingRows, mappingCols, row, col, &(codewords[chr++]), moduleOnColor);
            row += 2;
            col -= 2;
        } while ((row < mappingRows) && (col >= 0));
        row += 3;
        col += 1;
        /* ... until the entire modules array is scanned */
    } while ((row < mappingRows) || (col < mappingCols));
    
    /* If lower righthand corner is untouched then fill in the fixed pattern */
    if (!(modules[mappingRows * mappingCols - 1] &
         DmtxModuleVisited)) {
        
        modules[mappingRows * mappingCols - 1] |= moduleOnColor;
        modules[(mappingRows * mappingCols) - mappingCols - 2] |= moduleOnColor;
    } /* XXX should this fixed pattern also be used in reading somehow? */
    
    /* XXX compare that chr == region->dataSize here */
    return chr; /* XXX number of codewords read off */
}

static void patternShapeStandard(unsigned char *modules, int mappingRows, int mappingCols, int row, int col, unsigned char *codeword, int moduleOnColor)
{
    placeModule(modules, mappingRows, mappingCols, row-2, col-2, codeword, DmtxMaskBit1, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, row-2, col-1, codeword, DmtxMaskBit2, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, row-1, col-2, codeword, DmtxMaskBit3, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, row-1, col-1, codeword, DmtxMaskBit4, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, row-1, col,   codeword, DmtxMaskBit5, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, row,   col-2, codeword, DmtxMaskBit6, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, row,   col-1, codeword, DmtxMaskBit7, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, row,   col,   codeword, DmtxMaskBit8, moduleOnColor);
}

static void patternShapeSpecial1(unsigned char *modules, int mappingRows, int mappingCols, unsigned char *codeword, int moduleOnColor)
{
    placeModule(modules, mappingRows, mappingCols, mappingRows-1, 0, codeword, DmtxMaskBit1, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, mappingRows-1, 1, codeword, DmtxMaskBit2, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, mappingRows-1, 2, codeword, DmtxMaskBit3, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 0, mappingCols-2, codeword, DmtxMaskBit4, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 0, mappingCols-1, codeword, DmtxMaskBit5, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 1, mappingCols-1, codeword, DmtxMaskBit6, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 2, mappingCols-1, codeword, DmtxMaskBit7, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 3, mappingCols-1, codeword, DmtxMaskBit8, moduleOnColor);
}

static void patternShapeSpecial2(unsigned char *modules, int mappingRows, int mappingCols, unsigned char *codeword, int moduleOnColor)
{
    placeModule(modules, mappingRows, mappingCols, mappingRows-3, 0, codeword, DmtxMaskBit1, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, mappingRows-2, 0, codeword, DmtxMaskBit2, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, mappingRows-1, 0, codeword, DmtxMaskBit3, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 0, mappingCols-4, codeword, DmtxMaskBit4, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 0, mappingCols-3, codeword, DmtxMaskBit5, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 0, mappingCols-2, codeword, DmtxMaskBit6, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 0, mappingCols-1, codeword, DmtxMaskBit7, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 1, mappingCols-1, codeword, DmtxMaskBit8, moduleOnColor);
}

static void patternShapeSpecial3(unsigned char *modules, int mappingRows, int mappingCols, unsigned char *codeword, int moduleOnColor)
{
    placeModule(modules, mappingRows, mappingCols, mappingRows-3, 0, codeword, DmtxMaskBit1, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, mappingRows-2, 0, codeword, DmtxMaskBit2, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, mappingRows-1, 0, codeword, DmtxMaskBit3, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 0, mappingCols-2, codeword, DmtxMaskBit4, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 0, mappingCols-1, codeword, DmtxMaskBit5, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 1, mappingCols-1, codeword, DmtxMaskBit6, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 2, mappingCols-1, codeword, DmtxMaskBit7, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 3, mappingCols-1, codeword, DmtxMaskBit8, moduleOnColor);
}

static void patternShapeSpecial4(unsigned char *modules, int mappingRows, int mappingCols, unsigned char *codeword, int moduleOnColor)
{
    placeModule(modules, mappingRows, mappingCols, mappingRows-1, 0, codeword, DmtxMaskBit1, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, mappingRows-1, mappingCols-1, codeword, DmtxMaskBit2, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 0, mappingCols-3, codeword, DmtxMaskBit3, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 0, mappingCols-2, codeword, DmtxMaskBit4, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 0, mappingCols-1, codeword, DmtxMaskBit5, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 1, mappingCols-3, codeword, DmtxMaskBit6, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 1, mappingCols-2, codeword, DmtxMaskBit7, moduleOnColor);
    placeModule(modules, mappingRows, mappingCols, 1, mappingCols-1, codeword, DmtxMaskBit8, moduleOnColor);
}

static void placeModule(unsigned char *modules, int mappingRows, int mappingCols, int row, int col, unsigned char *codeword, int mask, int moduleOnColor)
{
    if (row < 0)
    {
        row += mappingRows;
        col += 4 - ((mappingRows+4)%8);
    }
    if (col < 0)
    {
        col += mappingCols;
        row += 4 - ((mappingCols+4)%8);
    }
    
    /* If module has already been assigned then we are decoding the pattern into codewords */
    if ((modules[row*mappingCols+col] & DmtxModuleAssigned) != 0)
    {
        if ((modules[row*mappingCols+col] & moduleOnColor) != 0)
            *codeword |= mask;
        else
            *codeword &= (0xff ^ mask);
    }
    /* Otherwise we are encoding the codewords into a pattern */
    else
    {
        if ((*codeword & mask) != 0x00)
            modules[row*mappingCols+col] |= moduleOnColor;
        
        modules[row*mappingCols+col] |= DmtxModuleAssigned;
    }
    
    modules[row*mappingCols+col] |= DmtxModuleVisited;
}

}  // namespace dmtx
