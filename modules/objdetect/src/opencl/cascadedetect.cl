///////////////////////////// OpenCL kernels for face detection //////////////////////////////
////////////////////////////// see the opencv/doc/license.txt ///////////////////////////////

//
// the code has been derived from the OpenCL Haar cascade kernel by
//
//    Niko Li, newlife20080214@gmail.com
//    Wang Weiyan, wangweiyanster@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Nathan, liujun@multicorewareinc.com
//    Peng Xiao, pengxiao@outlook.com
//    Erping Pang, erping@multicorewareinc.com
//

#ifdef HAAR
typedef struct __attribute__((aligned(4))) OptHaarFeature
{
    int4 ofs[3] __attribute__((aligned (4)));
    float4 weight __attribute__((aligned (4)));
}
OptHaarFeature;
#endif

#ifdef LBP
typedef struct __attribute__((aligned(4))) OptLBPFeature
{
    int16 ofs __attribute__((aligned (4)));
}
OptLBPFeature;
#endif

typedef struct __attribute__((aligned(4))) Stump
{
    float4 st __attribute__((aligned (4)));
}
Stump;

typedef struct __attribute__((aligned(4))) Node
{
    int4 n __attribute__((aligned (4)));
}
Node;

typedef struct __attribute__((aligned (4))) Stage
{
    int first __attribute__((aligned (4)));
    int ntrees __attribute__((aligned (4)));
    float threshold __attribute__((aligned (4)));
}
Stage;

typedef struct __attribute__((aligned (4))) ScaleData
{
    float scale __attribute__((aligned (4)));
    int szi_width __attribute__((aligned (4)));
    int szi_height __attribute__((aligned (4)));
    int layer_ofs __attribute__((aligned (4)));
    int ystep __attribute__((aligned (4)));
}
ScaleData;

#ifndef SUM_BUF_SIZE
#define SUM_BUF_SIZE 0
#endif

#ifndef NODE_COUNT
#define NODE_COUNT 1
#endif

#ifdef HAAR
__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE_X,LOCAL_SIZE_Y,1)))
void runHaarClassifier(
    int nscales, __global const ScaleData* scaleData,
    __global const int* sum,
    int _sumstep, int sumoffset,
    __global const OptHaarFeature* optfeatures,
    __global const Stage* stages,
    __global const Node* nodes,
    __global const float* leaves0,

    volatile __global int* facepos,
    int4 normrect, int sqofs, int2 windowsize)
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
    int lidx = ly*LOCAL_SIZE_X + lx;

    #if SUM_BUF_SIZE > 0
    int4 nofs = (int4)(mad24(normrect.y, SUM_BUF_STEP, normrect.x),
                       mad24(normrect.y, SUM_BUF_STEP, normrect.x + normrect.z),
                       mad24(normrect.y + normrect.w, SUM_BUF_STEP, normrect.x),
                       mad24(normrect.y + normrect.w, SUM_BUF_STEP, normrect.x + normrect.z));
    #else
    int4 nofs = nofs0;
    #endif
    #define LOCAL_SIZE (LOCAL_SIZE_X*LOCAL_SIZE_Y)
    __local int lstore[SUM_BUF_SIZE + LOCAL_SIZE*5/2+1];
    #if SUM_BUF_SIZE > 0
    __local int* ibuf = lstore;
    __local int* lcount = ibuf + SUM_BUF_SIZE;
    #else
    __local int* lcount = lstore;
    #endif
    __local float* lnf = (__local float*)(lcount + 1);
    __local float* lpartsum = lnf + LOCAL_SIZE;
    __local short* lbuf = (__local short*)(lpartsum + LOCAL_SIZE);

    for( scaleIdx = nscales-1; scaleIdx >= 0; scaleIdx-- )
    {
        __global const ScaleData* s = scaleData + scaleIdx;
        int ystep = s->ystep;
        int2 worksize = (int2)(max(s->szi_width - windowsize.x, 0), max(s->szi_height - windowsize.y, 0));
        int2 ntiles = (int2)((worksize.x + LOCAL_SIZE_X-1)/LOCAL_SIZE_X,
                             (worksize.y + LOCAL_SIZE_Y-1)/LOCAL_SIZE_Y);
        int totalTiles = ntiles.x*ntiles.y;

        for( tileIdx = groupIdx; tileIdx < totalTiles; tileIdx += ngroups )
        {
            int ix0 = (tileIdx % ntiles.x)*LOCAL_SIZE_X;
            int iy0 = (tileIdx / ntiles.x)*LOCAL_SIZE_Y;
            int ix = lx, iy = ly;
            __global const int* psum0 = sum + mad24(iy0, sumstep, ix0) + s->layer_ofs;
            __global const int* psum1 = psum0 + mad24(iy, sumstep, ix);

            if( ix0 >= worksize.x || iy0 >= worksize.y )
                continue;
            #if SUM_BUF_SIZE > 0
            for( i = lidx*4; i < SUM_BUF_SIZE; i += LOCAL_SIZE_X*LOCAL_SIZE_Y*4 )
            {
                int dy = i/SUM_BUF_STEP, dx = i - dy*SUM_BUF_STEP;
                vstore4(vload4(0, psum0 + mad24(dy, sumstep, dx)), 0, ibuf+i);
            }
            #endif

            if( lidx == 0 )
                lcount[0] = 0;
            barrier(CLK_LOCAL_MEM_FENCE);

            if( ix0 + ix < worksize.x && iy0 + iy < worksize.y )
            {
                #if NODE_COUNT==1
                __global const Stump* stump = (__global const Stump*)nodes;
                #else
                __global const Node* node = nodes;
                __global const float* leaves = leaves0;
                #endif
                #if SUM_BUF_SIZE > 0
                __local const int* psum = ibuf + mad24(iy, SUM_BUF_STEP, ix);
                #else
                __global const int* psum = psum1;
                #endif

                __global const int* psqsum = (__global const int*)(psum1 + sqofs);
                float sval = (psum[nofs.x] - psum[nofs.y] - psum[nofs.z] + psum[nofs.w])*invarea;
                float sqval = (psqsum[nofs0.x] - psqsum[nofs0.y] - psqsum[nofs0.z] + psqsum[nofs0.w])*invarea;
                float nf = (float)normarea * sqrt(max(sqval - sval * sval, 0.f));
                nf = nf > 0 ? nf : 1.f;

                for( stageIdx = 0; stageIdx < SPLIT_STAGE; stageIdx++ )
                {
                    int ntrees = stages[stageIdx].ntrees;
                    float s = 0.f;
                    #if NODE_COUNT==1
                    for( i = 0; i < ntrees; i++ )
                    {
                        float4 st = stump[i].st;
                        __global const OptHaarFeature* f = optfeatures + as_int(st.x);
                        float4 weight = f->weight;

                        int4 ofs = f->ofs[0];
                        sval = (psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w])*weight.x;
                        ofs = f->ofs[1];
                        sval = mad((float)(psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w]), weight.y, sval);
                        if( weight.z > 0 )
                        {
                            ofs = f->ofs[2];
                            sval = mad((float)(psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w]), weight.z, sval);
                        }

                        s += (sval < st.y*nf) ? st.z : st.w;
                    }
                    stump += ntrees;
                    #else
                    for( i = 0; i < ntrees; i++, node += NODE_COUNT, leaves += NODE_COUNT+1 )
                    {
                        int idx = 0;
                        do
                        {
                            int4 n = node[idx].n;
                            __global const OptHaarFeature* f = optfeatures + n.x;
                            float4 weight = f->weight;

                            int4 ofs = f->ofs[0];

                            sval = (psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w])*weight.x;
                            ofs = f->ofs[1];
                            sval = mad((float)(psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w]), weight.y, sval);
                            if( weight.z > 0 )
                            {
                                ofs = f->ofs[2];
                                sval = mad((float)(psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w]), weight.z, sval);
                            }

                            idx = (sval < as_float(n.y)*nf) ? n.z : n.w;
                        }
                        while(idx > 0);
                        s += leaves[-idx];
                    }
                    #endif

                    if( s < stages[stageIdx].threshold )
                        break;
                }

                if( stageIdx == SPLIT_STAGE && (ystep == 1 || ((ix | iy) & 1) == 0) )
                {
                    int count = atomic_inc(lcount);
                    lbuf[count] = (int)(ix | (iy << 8));
                    lnf[count] = nf;
                }
            }

            for( stageIdx = SPLIT_STAGE; stageIdx < N_STAGES; stageIdx++ )
            {
                barrier(CLK_LOCAL_MEM_FENCE);
                int nrects = lcount[0];

                if( nrects == 0 )
                    break;
                barrier(CLK_LOCAL_MEM_FENCE);
                if( lidx == 0 )
                    lcount[0] = 0;

                {
                    #if NODE_COUNT == 1
                    __global const Stump* stump = (__global const Stump*)nodes + stages[stageIdx].first;
                    #else
                    __global const Node* node = nodes + stages[stageIdx].first*NODE_COUNT;
                    __global const float* leaves = leaves0 + stages[stageIdx].first*(NODE_COUNT+1);
                    #endif
                    int nparts = LOCAL_SIZE / nrects;
                    int ntrees = stages[stageIdx].ntrees;
                    int ntrees_p = (ntrees + nparts - 1)/nparts;
                    int nr = lidx / nparts;
                    int partidx = -1, idxval = 0;
                    float partsum = 0.f, nf = 0.f;

                    if( nr < nrects )
                    {
                        partidx = lidx % nparts;
                        idxval = lbuf[nr];
                        nf = lnf[nr];

                        {
                        int ntrees0 = ntrees_p*partidx;
                        int ntrees1 = min(ntrees0 + ntrees_p, ntrees);
                        int ix1 = idxval & 255, iy1 = idxval >> 8;
                        #if SUM_BUF_SIZE > 0
                        __local const int* psum = ibuf + mad24(iy1, SUM_BUF_STEP, ix1);
                        #else
                        __global const int* psum = psum0 + mad24(iy1, sumstep, ix1);
                        #endif

                        #if NODE_COUNT == 1
                        for( i = ntrees0; i < ntrees1; i++ )
                        {
                            float4 st = stump[i].st;
                            __global const OptHaarFeature* f = optfeatures + as_int(st.x);
                            float4 weight = f->weight;

                            int4 ofs = f->ofs[0];
                            float sval = (psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w])*weight.x;
                            ofs = f->ofs[1];
                            sval = mad((float)(psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w]), weight.y, sval);
                            //if( weight.z > 0 )
                            if( fabs(weight.z) > 0 )
                            {
                                ofs = f->ofs[2];
                                sval = mad((float)(psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w]), weight.z, sval);
                            }

                            partsum += (sval < st.y*nf) ? st.z : st.w;
                        }
                        #else
                        for( i = ntrees0; i < ntrees1; i++ )
                        {
                            int idx = 0;
                            do
                            {
                                int4 n = node[i*2 + idx].n;
                                __global const OptHaarFeature* f = optfeatures + n.x;
                                float4 weight = f->weight;
                                int4 ofs = f->ofs[0];

                                float sval = (psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w])*weight.x;
                                ofs = f->ofs[1];
                                sval = mad((float)(psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w]), weight.y, sval);
                                if( weight.z > 0 )
                                {
                                    ofs = f->ofs[2];
                                    sval = mad((float)(psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w]), weight.z, sval);
                                }

                                idx = (sval < as_float(n.y)*nf) ? n.z : n.w;
                            }
                            while(idx > 0);
                            partsum += leaves[i*3-idx];
                        }
                        #endif
                        }
                    }
                    lpartsum[lidx] = partsum;
                    barrier(CLK_LOCAL_MEM_FENCE);

                    if( partidx == 0 )
                    {
                        float s = lpartsum[nr*nparts];
                        for( i = 1; i < nparts; i++ )
                            s += lpartsum[i + nr*nparts];
                        if( s >= stages[stageIdx].threshold )
                        {
                            int count = atomic_inc(lcount);
                            lbuf[count] = idxval;
                            lnf[count] = nf;
                        }
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            if( stageIdx == N_STAGES )
            {
                int nrects = lcount[0];
                if( lidx < nrects )
                {
                    int nfaces = atomic_inc(facepos);
                    if( nfaces < MAX_FACES )
                    {
                        volatile __global int* face = facepos + 1 + nfaces*3;
                        int val = lbuf[lidx];
                        face[0] = scaleIdx;
                        face[1] = ix0 + (val & 255);
                        face[2] = iy0 + (val >> 8);
                    }
                }
            }
        }
    }
}
#endif

#ifdef LBP
#undef CALC_SUM_OFS_
#define CALC_SUM_OFS_(p0, p1, p2, p3, ptr) \
    ((ptr)[p0] - (ptr)[p1] - (ptr)[p2] + (ptr)[p3])

__kernel void runLBPClassifierStumpSimple(
    int nscales, __global const ScaleData* scaleData,
    __global const int* sum,
    int _sumstep, int sumoffset,
    __global const OptLBPFeature* optfeatures,
    __global const Stage* stages,
    __global const Stump* stumps,
    __global const int* bitsets,
    int bitsetSize,

    volatile __global int* facepos,
    int2 windowsize)
{
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int local_size_x = get_local_size(0);
    int local_size_y = get_local_size(1);
    int groupIdx = get_group_id(1)*get_num_groups(0) + get_group_id(0);
    int ngroups = get_num_groups(0)*get_num_groups(1);
    int scaleIdx, tileIdx, stageIdx;
    int sumstep = (int)(_sumstep/sizeof(int));

    for( scaleIdx = nscales-1; scaleIdx >= 0; scaleIdx-- )
    {
        __global const ScaleData* s = scaleData + scaleIdx;
        int ystep = s->ystep;
        int2 worksize = (int2)(max(s->szi_width - windowsize.x, 0), max(s->szi_height - windowsize.y, 0));
        int2 ntiles = (int2)((worksize.x/ystep + local_size_x-1)/local_size_x,
                             (worksize.y/ystep + local_size_y-1)/local_size_y);
        int totalTiles = ntiles.x*ntiles.y;

        for( tileIdx = groupIdx; tileIdx < totalTiles; tileIdx += ngroups )
        {
            int iy = mad24((tileIdx / ntiles.x), local_size_y, ly) * ystep;
            int ix = mad24((tileIdx % ntiles.x), local_size_x, lx) * ystep;

            if( ix < worksize.x && iy < worksize.y )
            {
                __global const int* p = sum + mad24(iy, sumstep, ix) + s->layer_ofs;
                __global const Stump* stump = stumps;
                __global const int* bitset = bitsets;

                for( stageIdx = 0; stageIdx < N_STAGES; stageIdx++ )
                {
                    int i, ntrees = stages[stageIdx].ntrees;
                    float s = 0.f;
                    for( i = 0; i < ntrees; i++, stump++, bitset += bitsetSize )
                    {
                        float4 st = stump->st;
                        __global const OptLBPFeature* f = optfeatures + as_int(st.x);
                        int16 ofs = f->ofs;

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

                if( stageIdx == N_STAGES )
                {
                    int nfaces = atomic_inc(facepos);
                    if( nfaces < MAX_FACES )
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

__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE_X,LOCAL_SIZE_Y,1)))
void runLBPClassifierStump(
    int nscales, __global const ScaleData* scaleData,
    __global const int* sum,
    int _sumstep, int sumoffset,
    __global const OptLBPFeature* optfeatures,
    __global const Stage* stages,
    __global const Stump* stumps,
    __global const int* bitsets,
    int bitsetSize,

    volatile __global int* facepos,
    int2 windowsize)
{
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int groupIdx = get_group_id(0);
    int i, ngroups = get_global_size(0)/LOCAL_SIZE_X;
    int scaleIdx, tileIdx, stageIdx;
    int sumstep = (int)(_sumstep/sizeof(int));
    int lidx = ly*LOCAL_SIZE_X + lx;

    #define LOCAL_SIZE (LOCAL_SIZE_X*LOCAL_SIZE_Y)
    __local int lstore[SUM_BUF_SIZE + LOCAL_SIZE*3/2+1];
    #if SUM_BUF_SIZE > 0
    __local int* ibuf = lstore;
    __local int* lcount = ibuf + SUM_BUF_SIZE;
    #else
    __local int* lcount = lstore;
    #endif
    __local float* lpartsum = (__local float*)(lcount + 1);
    __local short* lbuf = (__local short*)(lpartsum + LOCAL_SIZE);

    for( scaleIdx = nscales-1; scaleIdx >= 0; scaleIdx-- )
    {
        __global const ScaleData* s = scaleData + scaleIdx;
        int ystep = s->ystep;
        int2 worksize = (int2)(max(s->szi_width - windowsize.x, 0), max(s->szi_height - windowsize.y, 0));
        int2 ntiles = (int2)((worksize.x + LOCAL_SIZE_X-1)/LOCAL_SIZE_X,
                             (worksize.y + LOCAL_SIZE_Y-1)/LOCAL_SIZE_Y);
        int totalTiles = ntiles.x*ntiles.y;

        for( tileIdx = groupIdx; tileIdx < totalTiles; tileIdx += ngroups )
        {
            int ix0 = (tileIdx % ntiles.x)*LOCAL_SIZE_X;
            int iy0 = (tileIdx / ntiles.x)*LOCAL_SIZE_Y;
            int ix = lx, iy = ly;
            __global const int* psum0 = sum + mad24(iy0, sumstep, ix0) + s->layer_ofs;

            if( ix0 >= worksize.x || iy0 >= worksize.y )
                continue;
            #if SUM_BUF_SIZE > 0
            for( i = lidx*4; i < SUM_BUF_SIZE; i += LOCAL_SIZE_X*LOCAL_SIZE_Y*4 )
            {
                int dy = i/SUM_BUF_STEP, dx = i - dy*SUM_BUF_STEP;
                vstore4(vload4(0, psum0 + mad24(dy, sumstep, dx)), 0, ibuf+i);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            #endif

            if( lidx == 0 )
                lcount[0] = 0;
            barrier(CLK_LOCAL_MEM_FENCE);

            if( ix0 + ix < worksize.x && iy0 + iy < worksize.y )
            {
                __global const Stump* stump = stumps;
                __global const int* bitset = bitsets;
                #if SUM_BUF_SIZE > 0
                __local const int* p = ibuf + mad24(iy, SUM_BUF_STEP, ix);
                #else
                __global const int* p = psum0 + mad24(iy, sumstep, ix);
                #endif

                for( stageIdx = 0; stageIdx < SPLIT_STAGE; stageIdx++ )
                {
                    int ntrees = stages[stageIdx].ntrees;
                    float s = 0.f;
                    for( i = 0; i < ntrees; i++, stump++, bitset += bitsetSize )
                    {
                        float4 st = stump->st;
                        __global const OptLBPFeature* f = optfeatures + as_int(st.x);
                        int16 ofs = f->ofs;

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

                if( stageIdx == SPLIT_STAGE && (ystep == 1 || ((ix | iy) & 1) == 0) )
                {
                    int count = atomic_inc(lcount);
                    lbuf[count] = (int)(ix | (iy << 8));
                }
            }

            for( stageIdx = SPLIT_STAGE; stageIdx < N_STAGES; stageIdx++ )
            {
                int nrects = lcount[0];

                barrier(CLK_LOCAL_MEM_FENCE);
                if( nrects == 0 )
                    break;
                if( lidx == 0 )
                    lcount[0] = 0;

                {
                    __global const Stump* stump = stumps + stages[stageIdx].first;
                    __global const int* bitset = bitsets + stages[stageIdx].first*bitsetSize;
                    int nparts = LOCAL_SIZE / nrects;
                    int ntrees = stages[stageIdx].ntrees;
                    int ntrees_p = (ntrees + nparts - 1)/nparts;
                    int nr = lidx / nparts;
                    int partidx = -1, idxval = 0;
                    float partsum = 0.f, nf = 0.f;

                    if( nr < nrects )
                    {
                        partidx = lidx % nparts;
                        idxval = lbuf[nr];

                        {
                            int ntrees0 = ntrees_p*partidx;
                            int ntrees1 = min(ntrees0 + ntrees_p, ntrees);
                            int ix1 = idxval & 255, iy1 = idxval >> 8;
                            #if SUM_BUF_SIZE > 0
                            __local const int* p = ibuf + mad24(iy1, SUM_BUF_STEP, ix1);
                            #else
                            __global const int* p = psum0 + mad24(iy1, sumstep, ix1);
                            #endif

                            for( i = ntrees0; i < ntrees1; i++ )
                            {
                                float4 st = stump[i].st;
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

                                partsum += (bitset[i*bitsetSize + idx] & (1 << mask)) ? st.z : st.w;
                            }
                        }
                    }
                    lpartsum[lidx] = partsum;
                    barrier(CLK_LOCAL_MEM_FENCE);

                    if( partidx == 0 )
                    {
                        float s = lpartsum[nr*nparts];
                        for( i = 1; i < nparts; i++ )
                            s += lpartsum[i + nr*nparts];
                        if( s >= stages[stageIdx].threshold )
                        {
                            int count = atomic_inc(lcount);
                            lbuf[count] = idxval;
                        }
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            if( stageIdx == N_STAGES )
            {
                int nrects = lcount[0];
                if( lidx < nrects )
                {
                    int nfaces = atomic_inc(facepos);
                    if( nfaces < MAX_FACES )
                    {
                        volatile __global int* face = facepos + 1 + nfaces*3;
                        int val = lbuf[lidx];
                        face[0] = scaleIdx;
                        face[1] = ix0 + (val & 255);
                        face[2] = iy0 + (val >> 8);
                    }
                }
            }
        }
    }
}
#endif
