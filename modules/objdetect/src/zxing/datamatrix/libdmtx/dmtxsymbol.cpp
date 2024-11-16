//
//  dmtxsymbol.cpp
//  test_dm
//
//  Created by wechatcv on 2022/5/7.
//

#include "dmtxsymbol.hpp"

#include "common.hpp"

namespace dmtx {

int getSizeIdxFromSymbolDimension(int rows, int cols)
{
    int symbolRows, symbolCols, i;
    for (i = 0; i < 30; i++){
        symbolRows = dmtxGetSymbolAttribute(DmtxSymAttribSymbolRows, i);
        symbolCols = dmtxGetSymbolAttribute(DmtxSymAttribSymbolCols, i);
        if (rows==symbolRows && cols==symbolCols){
            return i;
        }
    }
    return -1;
}

int dmtxGetSymbolAttribute(int attribute, int sizeIdx)
{
    static const int symbolRows[] = { 10, 12, 14, 16, 18, 20,  22,  24,  26,
        32, 36, 40,  44,  48,  52,
        64, 72, 80,  88,  96, 104,
        120, 132, 144,
        8,  8, 12,  12,  16,  16 };
    
    static const int symbolCols[] = { 10, 12, 14, 16, 18, 20,  22,  24,  26,
        32, 36, 40,  44,  48,  52,
        64, 72, 80,  88,  96, 104,
        120, 132, 144,
        18, 32, 26,  36,  36,  48 };
    
    static const int dataRegionRows[] = { 8, 10, 12, 14, 16, 18, 20, 22, 24,
        14, 16, 18, 20, 22, 24,
        14, 16, 18, 20, 22, 24,
        18, 20, 22,
        6,  6, 10, 10, 14, 14 };
    
    static const int dataRegionCols[] = { 8, 10, 12, 14, 16, 18, 20, 22, 24,
        14, 16, 18, 20, 22, 24,
        14, 16, 18, 20, 22, 24,
        18, 20, 22,
        16, 14, 24, 16, 16, 22 };
    
    static const int horizDataRegions[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2,
        4, 4, 4, 4, 4, 4,
        6, 6, 6,
        1, 2, 1, 2, 2, 2 };
    
    static const int interleavedBlocks[] = { 1, 1, 1, 1, 1, 1, 1,  1, 1,
        1, 1, 1, 1,  1, 2,
        2, 4, 4, 4,  4, 6,
        6, 8, 10,
        1, 1, 1, 1,  1, 1 };
    
    static const int symbolDataWords[] = { 3, 5, 8,  12,   18,   22,   30,   36,  44,
        62,   86,  114,  144,  174, 204,
        280,  368,  456,  576,  696, 816,
        1050, 1304, 1558,
        5,   10,   16,   22,   32,  49 };
    
    static const int blockErrorWords[] = { 5, 7, 10, 12, 14, 18, 20, 24, 28,
        36, 42, 48, 56, 68, 42,
        56, 36, 48, 56, 68, 56,
        68, 62, 62,
        7, 11, 14, 18, 24, 28 };
    
    static const int blockMaxCorrectable[] = { 2, 3, 5,  6,  7,  9,  10,  12,  14,
        18, 21, 24,  28,  34,  21,
        28, 18, 24,  28,  34,  28,
        34,  31,  31,
        3,  5,  7,   9,  12,  14 };
    
    if (sizeIdx < 0 || sizeIdx >= DmtxSymbolSquareCount + DmtxSymbolRectCount)
        return DmtxUndefined;
    
    switch (attribute) {
        case DmtxSymAttribSymbolRows:
            return symbolRows[sizeIdx];
        case DmtxSymAttribSymbolCols:
            return symbolCols[sizeIdx];
        case DmtxSymAttribDataRegionRows:
            return dataRegionRows[sizeIdx];
        case DmtxSymAttribDataRegionCols:
            return dataRegionCols[sizeIdx];
        case DmtxSymAttribHorizDataRegions:
            return horizDataRegions[sizeIdx];
        case DmtxSymAttribVertDataRegions:
            return (sizeIdx < DmtxSymbolSquareCount) ? horizDataRegions[sizeIdx] : 1;
        case DmtxSymAttribMappingMatrixRows:
            return dataRegionRows[sizeIdx] *
            dmtxGetSymbolAttribute(DmtxSymAttribVertDataRegions, sizeIdx);
        case DmtxSymAttribMappingMatrixCols:
            return dataRegionCols[sizeIdx] * horizDataRegions[sizeIdx];
        case DmtxSymAttribInterleavedBlocks:
            return interleavedBlocks[sizeIdx];
        case DmtxSymAttribBlockErrorWords:
            return blockErrorWords[sizeIdx];
        case DmtxSymAttribBlockMaxCorrectable:
            return blockMaxCorrectable[sizeIdx];
        case DmtxSymAttribSymbolDataWords:
            return symbolDataWords[sizeIdx];
        case DmtxSymAttribSymbolErrorWords:
            return blockErrorWords[sizeIdx] * interleavedBlocks[sizeIdx];
        case DmtxSymAttribSymbolMaxCorrectable:
            return blockMaxCorrectable[sizeIdx] * interleavedBlocks[sizeIdx];
    }
    
    return DmtxUndefined;
}

int dmtxGetBlockDataSize(int sizeIdx, int blockIdx)
{
    int symbolDataWords;
    int interleavedBlocks;
    int count;
    
    symbolDataWords = dmtxGetSymbolAttribute(DmtxSymAttribSymbolDataWords, sizeIdx);
    interleavedBlocks = dmtxGetSymbolAttribute(DmtxSymAttribInterleavedBlocks, sizeIdx);
    
    if (symbolDataWords < 1 || interleavedBlocks < 1)
        return DmtxUndefined;
    
    count = static_cast<int>(symbolDataWords/interleavedBlocks);
    
    return (sizeIdx == DmtxSymbol144x144 && blockIdx < 8) ? count + 1 : count;
}

}  // namespace dmtx
