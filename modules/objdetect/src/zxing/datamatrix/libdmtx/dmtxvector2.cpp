//
//  dmtxvector2.cpp
//  test_dm
//
//  Created by wechatcv on 2022/5/5.
//

#include "dmtxvector2.hpp"

#include <math.h>

namespace dmtx {

static DmtxVector2* dmtxVector2AddTo(DmtxVector2 *v1, const DmtxVector2 *v2)
{
    v1->X += v2->X;
    v1->Y += v2->Y;
    
    return v1;
}

static DmtxVector2* dmtxVector2Add(DmtxVector2 *vOut, const DmtxVector2 *v1, const DmtxVector2 *v2)
{
    *vOut = *v1;
    
    return dmtxVector2AddTo(vOut, v2);
}

static DmtxVector2* dmtxVector2SubFrom(DmtxVector2 *v1, const DmtxVector2 *v2)
{
    v1->X -= v2->X;
    v1->Y -= v2->Y;
    
    return v1;
}

DmtxVector2 * dmtxVector2Sub(DmtxVector2 *vOut, const DmtxVector2 *v1, const DmtxVector2 *v2)
{
    *vOut = *v1;
    
    return dmtxVector2SubFrom(vOut, v2);
}

static DmtxVector2 * dmtxVector2ScaleBy(DmtxVector2 *v, double s)
{
    v->X *= s;
    v->Y *= s;
    
    return v;
}

static DmtxVector2 * dmtxVector2Scale(DmtxVector2 *vOut, const DmtxVector2 *v, double s)
{
    *vOut = *v;
    
    return dmtxVector2ScaleBy(vOut, s);
}

double dmtxVector2Cross(const DmtxVector2 *v1, const DmtxVector2 *v2)
{
    return (v1->X * v2->Y) - (v1->Y * v2->X);
}

double dmtxVector2Norm(DmtxVector2 *v)
{
    double mag;
    
    mag = dmtxVector2Mag(v);
    
    if (mag <= DmtxAlmostZero)
        return -1.0; /* XXX this doesn't look clean */
    
    dmtxVector2ScaleBy(v, 1/mag);
    
    return mag;
}

double dmtxVector2Dot(const DmtxVector2 *v1, const DmtxVector2 *v2)
{
    return (v1->X * v2->X) + (v1->Y * v2->Y);
}


double dmtxVector2Mag(const DmtxVector2 *v)
{
    return sqrt(v->X * v->X + v->Y * v->Y);
}

static double dmtxDistanceAlongRay2(const DmtxRay2 *r, const DmtxVector2 *q)
{
    DmtxVector2 vSubTmp;
    
    return dmtxVector2Dot(dmtxVector2Sub(&vSubTmp, q, &(r->p)), &(r->v));
}

unsigned int dmtxRay2Intersect(DmtxVector2 *point, const DmtxRay2 *p0, const DmtxRay2 *p1)
{
    double numer, denom;
    DmtxVector2 w;
    
    denom = dmtxVector2Cross(&(p1->v), &(p0->v));
    if (fabs(denom) <= DmtxAlmostZero)
        return DmtxFail;
    
    dmtxVector2Sub(&w, &(p1->p), &(p0->p));
    numer = dmtxVector2Cross(&(p1->v), &w);
    
    return dmtxPointAlongRay2(point, p0, numer/denom);
}

unsigned int dmtxPointAlongRay2(DmtxVector2 *point, const DmtxRay2 *r, double t)
{
    DmtxVector2 vTmp;
    
    /* Ray should always have unit length of 1 */
    if (fabs(1.0 - dmtxVector2Mag(&(r->v))) > DmtxAlmostZero)
        return DmtxFail;
    
    dmtxVector2Scale(&vTmp, &(r->v), t);
    dmtxVector2Add(point, &(r->p), &vTmp);
    
    return DmtxPass;
}

}  // namespace dmtx
