//
//  dmtxmatrix3.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/5.
//

#ifndef dmtxmatrix3_hpp
#define dmtxmatrix3_hpp

#include <stdio.h>
#include "common.hpp"

namespace dmtx {

void dmtxMatrix3Translate(/*@out@*/ DmtxMatrix3 m, double tx, double ty);
void dmtxMatrix3Rotate(/*@out@*/ DmtxMatrix3 m, double angle);
void dmtxMatrix3Scale(/*@out@*/ DmtxMatrix3 m, double sx, double sy);
void dmtxMatrix3Shear(/*@out@*/ DmtxMatrix3 m, double shx, double shy);
unsigned int dmtxMatrix3LineSkewTop(DmtxMatrix3 m, double b0, double b1, double sz);
unsigned int dmtxMatrix3LineSkewTopInv(/*@out@*/ DmtxMatrix3 m, double b0, double b1, double sz);
unsigned int dmtxMatrix3LineSkewSide(/*@out@*/ DmtxMatrix3 m, double b0, double b1, double sz);
unsigned int dmtxMatrix3LineSkewSideInv(/*@out@*/ DmtxMatrix3 m, double b0, double b1, double sz);
void dmtxMatrix3Multiply(/*@out@*/ DmtxMatrix3 mOut, DmtxMatrix3 m0, DmtxMatrix3 m1);
void dmtxMatrix3MultiplyBy(DmtxMatrix3 m0, DmtxMatrix3 m1);
int dmtxMatrix3VMultiply(/*@out@*/ DmtxVector2 *vOut, DmtxVector2 *vIn, DmtxMatrix3 m);
int dmtxMatrix3VMultiplyBy(DmtxVector2 *v, DmtxMatrix3 m);

}  // namespace dmtx


#endif /* dmtxmatrix3_hpp */
