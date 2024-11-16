//
//  dmtxplacemod.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/7.
//

#ifndef dmtxplacemod_hpp
#define dmtxplacemod_hpp

#include <stdio.h>

namespace dmtx {
int ModulePlacementEcc200(unsigned char *modules, unsigned char *codewords, int sizeIdx, int moduleOnColor);
static void PatternShapeStandard(unsigned char *modules, int mappingRows, int mappingCols, int row, int col, unsigned char *codeword, int moduleOnColor);
static void PatternShapeSpecial1(unsigned char *modules, int mappingRows, int mappingCols, unsigned char *codeword, int moduleOnColor);
static void PatternShapeSpecial2(unsigned char *modules, int mappingRows, int mappingCols, unsigned char *codeword, int moduleOnColor);
static void PatternShapeSpecial3(unsigned char *modules, int mappingRows, int mappingCols, unsigned char *codeword, int moduleOnColor);
static void PatternShapeSpecial4(unsigned char *modules, int mappingRows, int mappingCols, unsigned char *codeword, int moduleOnColor);
static void PlaceModule(unsigned char *modules, int mappingRows, int mappingCols, int row, int col,
      unsigned char *codeword, int mask, int moduleOnColor);
}  // namespace dmtx
#endif /* dmtxplacemod_hpp */
