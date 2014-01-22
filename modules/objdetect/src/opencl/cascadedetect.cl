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

#define LOCAL_SIZE 8
#define SURV_BUF_SIZE 512
//#define SPLIT_STAGE 3

__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE,LOCAL_SIZE,1)))
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
    int ngroups = get_global_size(0)/LOCAL_SIZE;
    int scaleIdx, tileIdx, stageIdx;
    int startStage = 0, endStage = nstages;
    int sumstep = (int)(_sumstep/sizeof(int));
    int4 nofs = (int4)(mad24(normrect.y, sumstep, normrect.x),
                       mad24(normrect.y, sumstep, normrect.x + normrect.z),
                       mad24(normrect.y + normrect.w, sumstep, normrect.x),
                       mad24(normrect.y + normrect.w, sumstep, normrect.x + normrect.z));
    int normarea = normrect.z * normrect.w;
    float invarea = 1.f/normarea;
#ifdef SPLIT_STAGE
    int j, lidx = ly*LOCAL_SIZE + lx;
    __local int4 survived[SURV_BUF_SIZE];
    volatile __local int nsurvived_local;
    int nsurvived;

    if( lidx == 0 )
        nsurvived_local = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    endStage = min(nstages, SPLIT_STAGE);
#endif

    for( scaleIdx = nscales-1; scaleIdx >= 0; scaleIdx-- )
    {
        __global const ScaleData* s = scaleData + scaleIdx;
        int ystep = s->ystep;
        int2 worksize = (int2)(max(s->szi_width - windowsize.x, 0), max(s->szi_height - windowsize.y, 0));
        int2 ntiles = (int2)((worksize.x/ystep + LOCAL_SIZE-1)/LOCAL_SIZE,
                             (worksize.y/ystep + LOCAL_SIZE-1)/LOCAL_SIZE);
        int totalTiles = ntiles.x*ntiles.y;

        for( tileIdx = groupIdx; tileIdx < totalTiles; tileIdx += ngroups )
        {
            int iy = ((tileIdx / ntiles.x)*LOCAL_SIZE + ly)*ystep;
            int ix = ((tileIdx % ntiles.x)*LOCAL_SIZE + lx)*ystep;

            if( ix < worksize.x && iy < worksize.y )
            {
                __global const int* psum = sum + mad24(iy, sumstep, ix) + s->layer_ofs;
                __global const Stump* stump = stumps;

                float sval = (psum[nofs.x] - psum[nofs.y] - psum[nofs.z] + psum[nofs.w])*invarea;
                float sqval = psum[nofs.x + sqofs]*invarea;
                float nf = (float)normarea * sqrt(max(sqval - sval * sval, 0.f));
                nf = nf > 0 ? nf : 1.f;

                for( stageIdx = 0; stageIdx < endStage; stageIdx++ )
                {
                    int i, ntrees = stages[stageIdx].ntrees;
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

                if( stageIdx == endStage )
                {
                #ifdef SPLIT_STAGE
                    int nfaces = atomic_inc(&nsurvived_local);
                    if( nfaces < SURV_BUF_SIZE )
                        survived[nfaces] = (int4)(ix, iy, scaleIdx, as_int(nf));
                #else
                    int nfaces = atomic_inc(facepos);
                    if( nfaces < maxFaces )
                    {
                        volatile __global int* face = facepos + 1 + nfaces*4;
                        float factor = s->scale;
                        face[0] = convert_int_rte(ix*factor);
                        face[1] = convert_int_rte(iy*factor);
                        face[2] = convert_int_rte(windowsize.x*factor);
                        face[3] = convert_int_rte(windowsize.y*factor);
                    }
                #endif
                }
            }
        }
    }

#ifdef SPLIT_STAGE
    barrier(CLK_LOCAL_MEM_FENCE);
    nsurvived = nsurvived_local;
    endStage = nstages;

    for( j = lidx; j < nsurvived; j += LOCAL_SIZE*LOCAL_SIZE )
    {
        int4 si = survived[j];
        int ix = si.x, iy = si.y;
        float nf = as_float(si.w);
        scaleIdx = si.z;

        __global const ScaleData* s = scaleData + scaleIdx;
        __global const int* psum = sum + mad24(iy, sumstep, ix) + s->layer_ofs;
        __global const Stump* stump = stumps;

        for( stageIdx = 0; stageIdx < startStage; stageIdx++ )
            stump += stages[stageIdx].ntrees;

        for( ; stageIdx < endStage; stageIdx++ )
        {
            int i, ntrees = stages[stageIdx].ntrees;
            float s = 0.f;
            for( i = 0; i < ntrees; i++, stump++ )
            {
                float4 st = stump->st;
                __global const OptHaarFeature* f = optfeatures + as_int(st.x);
                float4 weight = f->weight;

                int4 ofs = f->ofs[0];
                float sval = (psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w])*weight.x;
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
                volatile __global int* face = facepos + 1 + nfaces*4;
                float factor = s->scale;
                face[0] = convert_int_rte(ix*factor);
                face[1] = convert_int_rte(iy*factor);
                face[2] = convert_int_rte(windowsize.x*factor);
                face[3] = convert_int_rte(windowsize.y*factor);
            }
        }
    }
#endif
}


__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE,LOCAL_SIZE,1)))
void runLBPClassifierStump(
    int nscales, __global const ScaleData* scaleData,
    __global const int* sum,
    int _sumstep, int sumoffset,
    __global const OptHaarFeature* optfeatures,

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
    int ngroups = get_global_size(0)/LOCAL_SIZE;
    int scaleIdx, tileIdx, stageIdx;
    int startStage = 0, endStage = nstages;
    int sumstep = (int)(_sumstep/sizeof(int));

    for( scaleIdx = nscales-1; scaleIdx >= 0; scaleIdx-- )
    {
        __global const ScaleData* s = scaleData + scaleIdx;
        int ystep = s->ystep;
        int2 worksize = (int2)(max(s->szi_width - windowsize.x, 0), max(s->szi_height - windowsize.y, 0));
        int2 ntiles = (int2)((worksize.x/ystep + LOCAL_SIZE-1)/LOCAL_SIZE,
                             (worksize.y/ystep + LOCAL_SIZE-1)/LOCAL_SIZE);
        int totalTiles = ntiles.x*ntiles.y;

        for( tileIdx = groupIdx; tileIdx < totalTiles; tileIdx += ngroups )
        {
            int iy = ((tileIdx / ntiles.x)*LOCAL_SIZE + ly)*ystep;
            int ix = ((tileIdx % ntiles.x)*LOCAL_SIZE + lx)*ystep;

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
                        volatile __global int* face = facepos + 1 + nfaces*4;
                        float factor = s->scale;
                        face[0] = convert_int_rte(ix*factor);
                        face[1] = convert_int_rte(iy*factor);
                        face[2] = convert_int_rte(windowsize.x*factor);
                        face[3] = convert_int_rte(windowsize.y*factor);
                    }
                }
            }
        }
    }
}
