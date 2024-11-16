//
//  dmtxsymbol.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/7.
//

#ifndef dmtxsymbol_hpp
#define dmtxsymbol_hpp

#include <stdio.h>

namespace dmtx {

int dmtxGetSymbolAttribute(int attribute, int sizeIdx);
int dmtxGetBlockDataSize(int sizeIdx, int blockIdx);
int getSizeIdxFromSymbolDimension(int rows, int cols);
int dmtxGetSymbolAttribute(int attribute, int sizeIdx);
}  // namespace dmtx
#endif /* dmtxsymbol_hpp */
