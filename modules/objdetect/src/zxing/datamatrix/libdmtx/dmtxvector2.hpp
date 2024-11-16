//
//  dmtxvector2.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/5.
//

#ifndef dmtxvector2_hpp
#define dmtxvector2_hpp

#include <stdio.h>
#include "common.hpp"

namespace dmtx {

DmtxVector2* dmtxVector2Sub(DmtxVector2 *vOut,const DmtxVector2 *v1,const DmtxVector2 *v2);

double dmtxVector2Cross(const DmtxVector2 *v1, const DmtxVector2 *v2);

double dmtxVector2Norm(DmtxVector2 *v);

double dmtxVector2Dot(const DmtxVector2 *v1,const DmtxVector2 *v2);

double dmtxVector2Mag(const DmtxVector2 *v);

unsigned int dmtxRay2Intersect(DmtxVector2 *point, const DmtxRay2 *p0, const DmtxRay2 *p1);

unsigned int dmtxPointAlongRay2(DmtxVector2 *point, const DmtxRay2 *r, double t);

}  // namespace dmtx

#endif /* dmtxvector2_hpp */
