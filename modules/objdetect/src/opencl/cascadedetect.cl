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


__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE_X,LOCAL_SIZE_Y,1)))
void runHaarClassifierStumpSplit(
    int nscales, __global const ScaleData* scaleData,
    __global const int* sum,
    int _sumstep, int sumoffset,
    __global const OptHaarFeature* optfeatures,

    int splitstage, int nstages,
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
    int lidx = ly*LOCAL_SIZE_X + lx;

    #ifdef SUM_BUF_SIZE
    __local int ibuf[SUM_BUF_SIZE];
    int4 nofs = (int4)(mad24(normrect.y, SUM_BUF_STEP, normrect.x),
                       mad24(normrect.y, SUM_BUF_STEP, normrect.x + normrect.z),
                       mad24(normrect.y + normrect.w, SUM_BUF_STEP, normrect.x),
                       mad24(normrect.y + normrect.w, SUM_BUF_STEP, normrect.x + normrect.z));
    #else
    int4 nofs = nofs0;
    #endif
    #define LOCAL_SIZE (LOCAL_SIZE_X*LOCAL_SIZE_Y)
    __local short lbuf[2][LOCAL_SIZE];
    __local float lnf[2][LOCAL_SIZE], lsum[LOCAL_SIZE], lpartsum[LOCAL_SIZE];
    __local int lcount[2];

    if(lidx == 0)
        lcount[0] = lcount[1] = 0;

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
            barrier(CLK_LOCAL_MEM_FENCE);
            #endif

            if( lidx == 0 )
                lcount[0] = 0;
            barrier(CLK_LOCAL_MEM_FENCE);

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

                for( stageIdx = 0; stageIdx < splitstage; stageIdx++ )
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

                if( stageIdx == splitstage )
                {
                    int count = atomic_inc(&lcount[0]);
                    lbuf[0][count] = (int)(ix | (iy << 8));
                    lnf[0][count] = nf;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for( stageIdx = splitstage; stageIdx < nstages; stageIdx++ )
            {
                int t = (stageIdx - splitstage) & 1, new_t = t ^ 1;
                int nr, nrects = lcount[t];
                int ntrees = stages[stageIdx].ntrees;
                __global const Stump* stump = stumps + stages[stageIdx].first;
                int ntrees_l = (ntrees + LOCAL_SIZE - 1)/LOCAL_SIZE;

                if( lidx == 0 )
                    lcount[new_t] = 0;
                if( nrects == 0 )
                    break;
                barrier(CLK_LOCAL_MEM_FENCE);

                for( nr = 0; nr < nrects; nr++ )
                {
                    float partsum = 0;
                    int ntrees0 = ntrees_l*lidx;
                    int ntrees1 = min(ntrees0 + ntrees_l, ntrees);
                    float nf = lnf[t][nr];
                    int n = LOCAL_SIZE/2, idxval = lbuf[t][nr], ix1 = idxval & 255, iy1 = idxval >> 8;
                    #ifdef SUM_BUF_SIZE
                    __local const int* psum = ibuf + mad24(iy1, SUM_BUF_STEP, ix1);
                    #else
                    __global const int* psum = psum0 + mad24(iy1, sumstep, ix1);
                    #endif
                    for( i = ntrees0; i < ntrees1; i++ )
                    {
                        float4 st = stump[i].st;
                        __global const OptHaarFeature* f = optfeatures + as_int(st.x);
                        float4 weight = f->weight;

                        int4 ofs = f->ofs[0];
                        float sval = (psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w])*weight.x;
                        ofs = f->ofs[1];
                        sval += (psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w])*weight.y;
                        //if( weight.z > 0 )
                        {
                            ofs = f->ofs[2];
                            sval += (psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w])*weight.z;
                        }

                        partsum += (sval < st.y*nf) ? st.z : st.w;
                    }
                    lpartsum[lidx] = partsum;
                    for( ; n > 1; n >>= 1 )
                    {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        if( lidx < n )
                            lpartsum[lidx] += lpartsum[lidx + n];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if( lidx < 1 )
                        lsum[nr] = lpartsum[0] + lpartsum[1];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                if( lidx < nrects )
                {
                    if( lsum[lidx] >= stages[stageIdx].threshold )
                    {
                        int count = atomic_inc(&lcount[new_t]);
                        lbuf[new_t][count] = lbuf[t][lidx];
                        lnf[new_t][count] = lnf[t][lidx];
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            if( stageIdx == nstages )
            {
                int t = (nstages - splitstage) & 1;
                int nrects = lcount[t];
                if( lidx < nrects )
                {
                    int nfaces = atomic_inc(facepos);
                    if( nfaces < maxFaces )
                    {
                        volatile __global int* face = facepos + 1 + nfaces*3;
                        int val = lbuf[t][lidx];
                        face[0] = scaleIdx;
                        face[1] = ix0 + (val & 255);
                        face[2] = iy0 + (val >> 8);
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
