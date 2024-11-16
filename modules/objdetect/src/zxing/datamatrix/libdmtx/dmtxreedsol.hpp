//
//  dmtxreedsol.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/7.
//

#ifndef dmtxreedsol_hpp
#define dmtxreedsol_hpp

#include <stdio.h>
#include "common.hpp"

namespace dmtx {

unsigned int RsDecode(unsigned char *code, int sizeIdx, int fix);

static DmtxBoolean RsComputeSyndromes(DmtxByteList *syn, const DmtxByteList *rec, int blockErrorWords);
static DmtxBoolean RsFindErrorLocatorPoly(DmtxByteList *elp, const DmtxByteList *syn, int errorWordCount, int maxCorrectable);
static DmtxBoolean RsFindErrorLocations(DmtxByteList *loc, const DmtxByteList *elp);
static unsigned int RsRepairErrors(DmtxByteList *rec, const DmtxByteList *loc, const DmtxByteList *elp, const DmtxByteList *syn);

}  // namespace dmtx

#endif /* dmtxreedsol_hpp */
