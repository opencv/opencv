//
//  dmtxbytelist.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/7.
//

#ifndef dmtxbytelist_hpp
#define dmtxbytelist_hpp

#include <stdio.h>
#include "common.hpp"


namespace dmtx {

DmtxByteList dmtxByteListBuild(DmtxByte *storage, int capacity);
unsigned int dmtxByteListInit(DmtxByteList *list, int length, DmtxByte value);
// void dmtxByteListClear(DmtxByteList *list);
// unsigned int dmtxByteListHasCapacity(DmtxByteList *list);
unsigned int dmtxByteListCopy(DmtxByteList *dst, const DmtxByteList *src);
unsigned int dmtxByteListPush(DmtxByteList *list, DmtxByte value);
DmtxByte dmtxByteListPop(DmtxByteList *list, unsigned int *passFail);

}  // namespace dmtx

#endif /* dmtxbytelist_hpp */
