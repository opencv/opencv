//
//  dmtxmatrix3.cpp
//  test_dm
//
//  Created by wechatcv on 2022/5/5.
//

#include "dmtxmatrix3.hpp"

#include <string.h>
#include <math.h>
#include <float.h>

namespace dmtx {

static void dmtxMatrix3Copy(DmtxMatrix3 m0, DmtxMatrix3 m1)
{
    memcpy(m0, m1, sizeof(DmtxMatrix3));
}

static void dmtxMatrix3Identity(DmtxMatrix3 m)
{
    static DmtxMatrix3 tmp = { {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1} };
    dmtxMatrix3Copy(m, tmp);
}

void dmtxMatrix3Translate(DmtxMatrix3 m, double tx, double ty)
{
    dmtxMatrix3Identity(m);
    m[2][0] = tx;
    m[2][1] = ty;
}

void dmtxMatrix3Rotate(DmtxMatrix3 m, double angle)
{
    double sinAngle, cosAngle;
    
    sinAngle = sin(angle);
    cosAngle = cos(angle);
    
    dmtxMatrix3Identity(m);
    m[0][0] = cosAngle;
    m[0][1] = sinAngle;
    m[1][0] = -sinAngle;
    m[1][1] = cosAngle;
}

void dmtxMatrix3Scale(DmtxMatrix3 m, double sx, double sy)
{
    dmtxMatrix3Identity(m);
    m[0][0] = sx;
    m[1][1] = sy;
}

void dmtxMatrix3Shear(DmtxMatrix3 m, double shx, double shy)
{
    dmtxMatrix3Identity(m);
    m[1][0] = shx;
    m[0][1] = shy;
}

unsigned int dmtxMatrix3LineSkewTop(DmtxMatrix3 m, double b0, double b1, double sz)
{
    if (b0 < DmtxAlmostZero) return DmtxFail;
    
    dmtxMatrix3Identity(m);
    m[0][0] = b1 / b0;
    m[1][1] = sz / b0;
    m[0][2] = (b1 - b0) / (sz*b0);
    return DmtxPass;
}

unsigned int dmtxMatrix3LineSkewTopInv(DmtxMatrix3 m, double b0, double b1, double sz)
{
    if (b1 < DmtxAlmostZero) return DmtxFail;
    
    dmtxMatrix3Identity(m);
    m[0][0] = b0 / b1;
    m[1][1] = b0 / sz;
    m[0][2] = (b0 - b1) / (sz*b1);
    return DmtxPass;
}

unsigned int dmtxMatrix3LineSkewSide(DmtxMatrix3 m, double b0, double b1, double sz)
{
    if (b0 < DmtxAlmostZero) return DmtxFail;
    
    dmtxMatrix3Identity(m);
    m[0][0] = sz/b0;
    m[1][1] = b1/b0;
    m[1][2] = (b1 - b0)/(sz*b0);
    return DmtxPass;
}

unsigned int dmtxMatrix3LineSkewSideInv(DmtxMatrix3 m, double b0, double b1, double sz)
{
    if (b1 < DmtxAlmostZero) return DmtxFail;
    
    dmtxMatrix3Identity(m);
    m[0][0] = b0/sz;
    m[1][1] = b0/b1;
    m[1][2] = (b0 - b1)/(sz*b1);
    return DmtxPass;
}

void dmtxMatrix3Multiply(DmtxMatrix3 mOut, DmtxMatrix3 m0, DmtxMatrix3 m1)
{
    int i, j, k;
    double val;
    
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            val = 0.0;
            for (k = 0; k < 3; k++) {
                val += m0[i][k] * m1[k][j];
            }
            mOut[i][j] = val;
        }
    }
}

void dmtxMatrix3MultiplyBy(DmtxMatrix3 m0, DmtxMatrix3 m1)
{
    DmtxMatrix3 mTmp;
    
    dmtxMatrix3Copy(mTmp, m0);
    dmtxMatrix3Multiply(m0, mTmp, m1);
}

int dmtxMatrix3VMultiply(DmtxVector2 *vOut, DmtxVector2 *vIn, DmtxMatrix3 m)
{
    double w;
    
    w = vIn->X*m[0][2] + vIn->Y*m[1][2] + m[2][2];
    if (fabs(w) <= DmtxAlmostZero) {
        vOut->X = FLT_MAX;
        vOut->Y = FLT_MAX;
        return DmtxFail;
    }
    
    vOut->X = (vIn->X*m[0][0] + vIn->Y*m[1][0] + m[2][0])/w;
    vOut->Y = (vIn->X*m[0][1] + vIn->Y*m[1][1] + m[2][1])/w;
    
    return DmtxPass;
}

int dmtxMatrix3VMultiplyBy(DmtxVector2 *v, DmtxMatrix3 m)
{
    int success;
    DmtxVector2 vOut;
    
    success = dmtxMatrix3VMultiply(&vOut, v, m);
    *v = vOut;
    
    return success;
}

}  // namespace dmtx


