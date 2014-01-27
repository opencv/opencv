///////////////////////////// OpenCL kernels for face detection //////////////////////////////
////////////////////////////// see the opencv/doc/license.txt ///////////////////////////////

typedef struct __attribute__((aligned(4))) OptHaarFeature
{
    int4 ofs[3] __attribute__((aligned (4)));
    float4 weight __attribute__((aligned (4)));
}
OptHaarFeature;

typedef struct __attribute__((aligned(4))) OptLBPFeature
{
    int16 ofs __attribute__((aligned (4)));
}
OptLBPFeature;

typedef struct __attribute__((aligned(4))) Stump
{
    float4 st __attribute__((aligned (4)));
}
Stump;

typedef struct __attribute__((aligned (4))) Stage
{
    int first __attribute__((aligned (4)));
    int ntrees __attribute__((aligned (4)));
    float threshold __attribute__((aligned (4)));
}
Stage;

typedef struct __attribute__((aligned (4))) ScaleData
{
    float scale __attribute__((aligned (4)));;
    int szi_width __attribute__((aligned (4)));
    int szi_height __attribute__((aligned (4)));
    int layer_ofs __attribute__((aligned (4)));
    int ystep __attribute__((aligned (4)));
}
ScaleData;

#define SURV_BUF_SIZE 512
//#define SPLIT_STAGE 3

__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE_X,LOCAL_SIZE_Y,1)))
void runHaarClassifierStump(
    int nscales, __global const ScaleData* scaleData,
    __global const int* sum,
    int _sumstep, int sumoffset,
    __global const OptHaarFeature* optfeatures,

    int nstages,
    __global const Stage* stages,
    __global const Stump* stumps,

    volatile __global int* facepos,
    int4 normrect, int sqofs, int2 windowsize, int maxFaces)
{
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int groupIdx = get_group_id(0);
    int i, ngroups = get_global_size(0)/LOCAL_SIZE_X;
    int scaleIdx, tileIdx, stageIdx;
    int sumstep = (int)(_sumstep/sizeof(int));
    int4 nofs0 = (int4)(mad24(normrect.y, sumstep, normrect.x),
                        mad24(normrect.y, sumstep, normrect.x + normrect.z),
                        mad24(normrect.y + normrect.w, sumstep, normrect.x),
                        mad24(normrect.y + normrect.w, sumstep, normrect.x + normrect.z));
    int normarea = normrect.z * normrect.w;
    float invarea = 1.f/normarea;

#ifdef SUM_BUF_SIZE
    __local int ibuf[SUM_BUF_SIZE];
    int lidx = ly*LOCAL_SIZE_X + lx;
    int4 nofs = (int4)(mad24(normrect.y, SUM_BUF_STEP, normrect.x),
                       mad24(normrect.y, SUM_BUF_STEP, normrect.x + normrect.z),
                       mad24(normrect.y + normrect.w, SUM_BUF_STEP, normrect.x),
                       mad24(normrect.y + normrect.w, SUM_BUF_STEP, normrect.x + normrect.z));
#else
    int4 nofs = nofs0;
#endif

    for( scaleIdx = nscales-1; scaleIdx >= 0; scaleIdx-- )
    {
        __global const ScaleData* s = scaleData + scaleIdx;
        int ystep = s->ystep;
        int2 worksize = (int2)(max(s->szi_width - windowsize.x, 0), max(s->szi_height - windowsize.y, 0));
        int2 ntiles = (int2)((worksize.x/ystep + LOCAL_SIZE_X-1)/LOCAL_SIZE_X,
                             (worksize.y/ystep + LOCAL_SIZE_Y-1)/LOCAL_SIZE_Y);
        int totalTiles = ntiles.x*ntiles.y;
#ifdef SUM_BUF_SIZE
        int2 bufsize = (int2)(LOCAL_SIZE_X*ystep + windowsize.x, LOCAL_SIZE_Y*ystep + windowsize.y);
        int buftotal = bufsize.x*bufsize.y;
#endif

        for( tileIdx = groupIdx; tileIdx < totalTiles; tileIdx += ngroups )
        {
            int ix0 = (tileIdx % ntiles.x)*LOCAL_SIZE_X*ystep;
            int iy0 = (tileIdx / ntiles.x)*LOCAL_SIZE_Y*ystep;
            int ix = lx*ystep, iy = ly*ystep;
            __global const int* psum0 = sum + mad24(iy0, sumstep, ix0) + s->layer_ofs;
            __global const int* psum1 = psum0 + mad24(iy, sumstep, ix);

        #ifdef SUM_BUF_SIZE
            if( ix0 >= worksize.x || iy0 >= worksize.y )
                continue;
            for( i = lidx*4; i < buftotal; i += LOCAL_SIZE_X*LOCAL_SIZE_Y*4 )
            {
                int dy = i/bufsize.x, dx = i - dy*bufsize.x;
                vstore4(vload4(0, psum0 + mad24(dy, sumstep, dx)), 0, ibuf+mad24(dy, SUM_BUF_STEP, dx));
            }
            /*for( i = lidx; i < buftotal; i += LOCAL_SIZE_X*LOCAL_SIZE_Y )
            {
                int dy = i/bufsize.x, dx = i - dy*bufsize.x;
                ibuf[mad24(dy, SUM_BUF_STEP, dx)] = psum0[mad24(dy, sumstep, dx)];
            }*/
            barrier(CLK_LOCAL_MEM_FENCE);
        #endif

            if( ix0 + ix < worksize.x && iy0 + iy < worksize.y )
            {
                __global const Stump* stump = stumps;
            #ifdef SUM_BUF_SIZE
                __local const int* psum = ibuf + mad24(iy, SUM_BUF_STEP, ix);
            #else
                __global const int* psum = psum1;
            #endif

                float sval = (psum[nofs.x] - psum[nofs.y] - psum[nofs.z] + psum[nofs.w])*invarea;
                float sqval = psum1[nofs0.x + sqofs]*invarea;
                float nf = (float)normarea * sqrt(max(sqval - sval * sval, 0.f));
                nf = nf > 0 ? nf : 1.f;

                for( stageIdx = 0; stageIdx < nstages; stageIdx++ )
                {
                    int ntrees = stages[stageIdx].ntrees;
                    float s = 0.f;
                    for( i = 0; i < ntrees; i++, stump++ )
                    {
                        float4 st = stump->st;
                        __global const OptHaarFeature* f = optfeatures + as_int(st.x);
                        float4 weight = f->weight;

                        int4 ofs = f->ofs[0];
                        sval = (psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w])*weight.x;
                        ofs = f->ofs[1];
                        sval += (psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w])*weight.y;
                        if( weight.z > 0 )
                        {
                            ofs = f->ofs[2];
                            sval += (psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w])*weight.z;
                        }

                        s += (sval < st.y*nf) ? st.z : st.w;
                    }

                    if( s < stages[stageIdx].threshold )
                        break;
                }

                if( stageIdx == nstages )
                {
                    int nfaces = atomic_inc(facepos);
                    if( nfaces < maxFaces )
                    {
                        volatile __global int* face = facepos + 1 + nfaces*3;
                        face[0] = scaleIdx;
                        face[1] = ix0 + ix;
                        face[2] = iy0 + iy;
                    }
                }
            }
        }
    }
}


#if LOCAL_SIZE_X == 4

#define loadop(ofs, addr) (ystep == 1 ? vload4(ofs, addr) : vload8(ofs, addr).s0246)
#define loadop_a(ofs, addr) (vload4(ofs, addr))
#define loadop_e(ofs, addr) (vload8(ofs, addr).s0246)
#define storeop(val, ofs, addr) vstore4(val, ofs, addr)
#define setall(tp, x) (tp)(x, x, x, x)
#define vectype_flt float4
#define vectype_int int4
#define convert_flt(x) convert_float4(x)

#elif LOCAL_SIZE_X == 8

#define loadop(ofs, addr) (ystep == 1 ? vload8(ofs, addr) : vload16(ofs, addr).s02468ace)
#define storeop(val, ofs, addr) vstore8(val, ofs, addr)
#define setall(tp, x) (tp)(x, x, x, x, x, x, x, x)
#define vectype_flt float8
#define vectype_int int8
#define convert_flt(x) convert_float8(x)

#endif

__kernel __attribute__((reqd_work_group_size(1,LOCAL_SIZE_Y,1)))
void runHaarClassifierStump2(
    int nscales, __global const ScaleData* scaleData,
    __global const int* sum,
    int _sumstep, int sumoffset,
    __global const OptHaarFeature* optfeatures,

    int nstages,
    __global const Stage* stages,
    __global const Stump* stumps,

    volatile __global int* facepos,
    int4 normrect, int sqofs, int2 windowsize, int maxFaces)
{
    int ly = get_local_id(1);
    int groupIdx = get_group_id(0);
    int i, ngroups = get_global_size(0)/LOCAL_SIZE_X;
    int scaleIdx, tileIdx, stageIdx;
    int sumstep = (int)(_sumstep/sizeof(int));
    int4 nofs = (int4)(mad24(normrect.y, sumstep, normrect.x),
                       mad24(normrect.y, sumstep, normrect.x + normrect.z),
                       mad24(normrect.y + normrect.w, sumstep, normrect.x),
                       mad24(normrect.y + normrect.w, sumstep, normrect.x + normrect.z));
    int normarea = normrect.z * normrect.w;
    float invarea = 1.f/normarea;
    vectype_flt z = setall(vectype_flt, 0.f);

    for( scaleIdx = nscales-1; scaleIdx >= 0; scaleIdx-- )
    {
        __global const ScaleData* s = scaleData + scaleIdx;
        int ystep = s->ystep;
        int2 worksize = (int2)(max(s->szi_width - windowsize.x, 0), max(s->szi_height - windowsize.y, 0));
        int2 ntiles = (int2)((worksize.x/ystep + LOCAL_SIZE_X-1)/LOCAL_SIZE_X,
                             (worksize.y/ystep + LOCAL_SIZE_Y-1)/LOCAL_SIZE_Y);
        int totalTiles = ntiles.x*ntiles.y;

        for( tileIdx = groupIdx; tileIdx < totalTiles; tileIdx += ngroups )
        {
            int ix = (tileIdx % ntiles.x)*LOCAL_SIZE_X*ystep;
            int iy = ((tileIdx / ntiles.x)*LOCAL_SIZE_Y + ly)*ystep;
            __global const int* psum = sum + mad24(iy, sumstep, ix) + s->layer_ofs;

            if( ix < worksize.x && iy < worksize.y )
            {
                __global const Stump* stump = stumps;
                vectype_int mask = setall(vectype_int, -1);
                vectype_flt sval, sqval, nf;

                sval = convert_flt(loadop(0, psum + nofs.x) - loadop(0, psum + nofs.y) -
                                   loadop(0, psum + nofs.z) + loadop(0, psum + nofs.w))*invarea;
                sqval = convert_flt(loadop(0, psum + nofs.x + sqofs))*invarea;
                nf = (float)normarea * sqrt(max(sqval - sval * sval, z));
                nf = nf > 0 ? nf : 1.f;

                for( stageIdx = 0; stageIdx < nstages; stageIdx++ )
                {
                    int ntrees = stages[stageIdx].ntrees;
                    vectype_flt s = z;
                    for( i = 0; i < ntrees; i++, stump++ )
                    {
                        float4 st = stump->st;
                        __global const OptHaarFeature* f = optfeatures + as_int(st.x);
                        float4 weight = f->weight;

                        int4 ofs0 = f->ofs[0], ofs1 = f->ofs[1];
                        vectype_int a0, b0, c0, d0, a1, b1, c1, d1;
                        if( ystep == 1 )
                        {
                            a0 = loadop_a(0, psum + ofs0.x);
                            b0 = loadop_a(0, psum + ofs0.y);
                            c0 = loadop_a(0, psum + ofs0.z);
                            d0 = loadop_a(0, psum + ofs0.w);

                            a1 = loadop_a(0, psum + ofs1.x);
                            b1 = loadop_a(0, psum + ofs1.y);
                            c1 = loadop_a(0, psum + ofs1.z);
                            d1 = loadop_a(0, psum + ofs1.w);
                        }
                        else
                        {
                            a0 = loadop_e(0, psum + ofs0.x);
                            b0 = loadop_e(0, psum + ofs0.y);
                            c0 = loadop_e(0, psum + ofs0.z);
                            d0 = loadop_e(0, psum + ofs0.w);

                            a1 = loadop_e(0, psum + ofs1.x);
                            b1 = loadop_e(0, psum + ofs1.y);
                            c1 = loadop_e(0, psum + ofs1.z);
                            d1 = loadop_e(0, psum + ofs1.w);
                        }
                        sval = convert_flt(a0 - b0 - c0 + d0)*weight.x +
                               convert_flt(a1 - b1 - c1 + d1)*weight.y;
                        if( weight.z > 0 )
                        {
                            ofs0 = f->ofs[2];
                            if( ystep == 1 )
                            {
                                a0 = loadop_a(0, psum + ofs0.x);
                                b0 = loadop_a(0, psum + ofs0.y);
                                c0 = loadop_a(0, psum + ofs0.z);
                                d0 = loadop_a(0, psum + ofs0.w);
                            }
                            else
                            {
                                a0 = loadop_e(0, psum + ofs0.x);
                                b0 = loadop_e(0, psum + ofs0.y);
                                c0 = loadop_e(0, psum + ofs0.z);
                                d0 = loadop_e(0, psum + ofs0.w);
                            }
                            sval += convert_flt(a0 - b0 - c0 + d0)*weight.z;
                        }

                        s += sval < st.y*nf ? setall(vectype_flt, st.z) : setall(vectype_flt, st.w);
                    }

                    mask &= s >= stages[stageIdx].threshold;
                    if( !any(mask) )
                        break;
                }

                if( stageIdx == nstages )
                {
                    int m[LOCAL_SIZE_X], n, nfaces;
                    storeop(mask, 0, m);

                    n = -(m[0] + m[1] + m[2] + m[3]
                    #if LOCAL_SIZE_X > 4
                    + m[4] + m[5] + m[6] + m[7]
                    #endif
                    );
                    nfaces = atomic_add(facepos, n);
                    if( nfaces <= maxFaces - n )
                    {
                        volatile __global int* face = facepos + 1 + nfaces*3;
                        for( i = 0; i < LOCAL_SIZE_X; i++ )
                        {
                            if(m[i] && ix + i < worksize.x )
                            {
                                face[0] = scaleIdx;
                                face[1] = ix + i;
                                face[2] = iy;
                                face += 3;
                            }
                        }
                    }
                }
            }
        }
    }
}


__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE_X,LOCAL_SIZE_Y,1)))
void runLBPClassifierStump(
    int nscales, __global const ScaleData* scaleData,
    __global const int* sum,
    int _sumstep, int sumoffset,
    __global const OptLBPFeature* optfeatures,

    int nstages,
    __global const Stage* stages,
    __global const Stump* stumps,
    __global const int* bitsets,
    int bitsetSize,

    volatile __global int* facepos,
    int2 windowsize, int maxFaces)
{
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int groupIdx = get_group_id(0);
    int ngroups = get_global_size(0)/LOCAL_SIZE_X;
    int scaleIdx, tileIdx, stageIdx;
    int startStage = 0, endStage = nstages;
    int sumstep = (int)(_sumstep/sizeof(int));

    for( scaleIdx = nscales-1; scaleIdx >= 0; scaleIdx-- )
    {
        __global const ScaleData* s = scaleData + scaleIdx;
        int ystep = s->ystep;
        int2 worksize = (int2)(max(s->szi_width - windowsize.x, 0), max(s->szi_height - windowsize.y, 0));
        int2 ntiles = (int2)((worksize.x/ystep + LOCAL_SIZE_X-1)/LOCAL_SIZE_X,
                             (worksize.y/ystep + LOCAL_SIZE_Y-1)/LOCAL_SIZE_Y);
        int totalTiles = ntiles.x*ntiles.y;

        for( tileIdx = groupIdx; tileIdx < totalTiles; tileIdx += ngroups )
        {
            int iy = ((tileIdx / ntiles.x)*LOCAL_SIZE_X + ly)*ystep;
            int ix = ((tileIdx % ntiles.x)*LOCAL_SIZE_Y + lx)*ystep;

            if( ix < worksize.x && iy < worksize.y )
            {
                __global const int* p = sum + mad24(iy, sumstep, ix) + s->layer_ofs;
                __global const Stump* stump = stumps;
                __global const int* bitset = bitsets;

                for( stageIdx = 0; stageIdx < endStage; stageIdx++ )
                {
                    int i, ntrees = stages[stageIdx].ntrees;
                    float s = 0.f;
                    for( i = 0; i < ntrees; i++, stump++, bitset += bitsetSize )
                    {
                        float4 st = stump->st;
                        __global const OptLBPFeature* f = optfeatures + as_int(st.x);
                        int16 ofs = f->ofs;

                        #define CALC_SUM_OFS_(p0, p1, p2, p3, ptr) \
                        ((ptr)[p0] - (ptr)[p1] - (ptr)[p2] + (ptr)[p3])

                        int cval = CALC_SUM_OFS_( ofs.s5, ofs.s6, ofs.s9, ofs.sa, p );

                        int mask, idx = (CALC_SUM_OFS_( ofs.s0, ofs.s1, ofs.s4, ofs.s5, p ) >= cval ? 4 : 0); // 0
                        idx |= (CALC_SUM_OFS_( ofs.s1, ofs.s2, ofs.s5, ofs.s6, p ) >= cval ? 2 : 0); // 1
                        idx |= (CALC_SUM_OFS_( ofs.s2, ofs.s3, ofs.s6, ofs.s7, p ) >= cval ? 1 : 0); // 2

                        mask = (CALC_SUM_OFS_( ofs.s6, ofs.s7, ofs.sa, ofs.sb, p ) >= cval ? 16 : 0); // 5
                        mask |= (CALC_SUM_OFS_( ofs.sa, ofs.sb, ofs.se, ofs.sf, p ) >= cval ? 8 : 0);  // 8
                        mask |= (CALC_SUM_OFS_( ofs.s9, ofs.sa, ofs.sd, ofs.se, p ) >= cval ? 4 : 0);  // 7
                        mask |= (CALC_SUM_OFS_( ofs.s8, ofs.s9, ofs.sc, ofs.sd, p ) >= cval ? 2 : 0);  // 6
                        mask |= (CALC_SUM_OFS_( ofs.s4, ofs.s5, ofs.s8, ofs.s9, p ) >= cval ? 1 : 0);  // 7

                        s += (bitset[idx] & (1 << mask)) ? st.z : st.w;
                    }

                    if( s < stages[stageIdx].threshold )
                        break;
                }

                if( stageIdx == nstages )
                {
                    int nfaces = atomic_inc(facepos);
                    if( nfaces < maxFaces )
                    {
                        volatile __global int* face = facepos + 1 + nfaces*3;
                        face[0] = scaleIdx;
                        face[1] = ix;
                        face[2] = iy;
                    }
                }
            }
        }
    }
}
