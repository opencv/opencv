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

__kernel void runHaarClassifierStump(
    __global const int* sum,
    int sumstep, int sumoffset,
    __global const int* sqsum,
    int sqsumstep, int sqsumoffset,
    __global const OptHaarFeature* optfeatures,

    int nstages,
    __global const Stage* stages,
    __global const Stump* stumps,

    volatile __global int* facepos,
    int2 imgsize, int xyscale, float factor,
    int4 normrect, int2 windowsize, int maxFaces)
{
    int ix = get_global_id(0)*xyscale;
    int iy = get_global_id(1)*xyscale;
    sumstep /= sizeof(int);
    sqsumstep /= sizeof(int);

    if( ix < imgsize.x && iy < imgsize.y )
    {
        int stageIdx;
        __global const Stump* stump = stumps;

        __global const int* psum = sum + mad24(iy, sumstep, ix);
        __global const int* pnsum = psum + mad24(normrect.y, sumstep, normrect.x);
        int normarea = normrect.z * normrect.w;
        float invarea = 1.f/normarea;
        float sval = (pnsum[0] - pnsum[normrect.z] - pnsum[mul24(normrect.w, sumstep)] +
                      pnsum[mad24(normrect.w, sumstep, normrect.z)])*invarea;
        float sqval = (sqsum[mad24(iy + normrect.y, sqsumstep, ix + normrect.x)])*invarea;
        float nf = (float)normarea * sqrt(max(sqval - sval * sval, 0.f));
        nf = nf > 0 ? nf : 1.f;

        for( stageIdx = 0; stageIdx < nstages; stageIdx++ )
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

        if( stageIdx == nstages )
        {
            int nfaces = atomic_inc(facepos);
            if( nfaces < maxFaces )
            {
                volatile __global int* face = facepos + 1 + nfaces*4;
                face[0] = convert_int_rte(ix*factor);
                face[1] = convert_int_rte(iy*factor);
                face[2] = convert_int_rte(windowsize.x*factor);
                face[3] = convert_int_rte(windowsize.y*factor);
            }
        }
    }
}


__kernel void runLBPClassifierStump(
    __global const int* sum,
    int sumstep, int sumoffset,
    __global const OptLBPFeature* optfeatures,

    int nstages,
    __global const Stage* stages,
    __global const Stump* stumps,
    __global const int* bitsets,
    int bitsetSize,

    volatile __global int* facepos,
    int2 imgsize, int xyscale, float factor,
    int2 windowsize, int maxFaces)
{
    int ix = get_global_id(0)*xyscale;
    int iy = get_global_id(1)*xyscale;
    sumstep /= sizeof(int);

    if( ix < imgsize.x && iy < imgsize.y )
    {
        int stageIdx;
        __global const Stump* stump = stumps;
        __global const int* p = sum + mad24(iy, sumstep, ix);

        for( stageIdx = 0; stageIdx < nstages; stageIdx++ )
        {
            int i, ntrees = stages[stageIdx].ntrees;
            float s = 0.f;
            for( i = 0; i < ntrees; i++, stump++, bitsets += bitsetSize )
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

                s += (bitsets[idx] & (1 << mask)) ? st.z : st.w;
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
                face[0] = convert_int_rte(ix*factor);
                face[1] = convert_int_rte(iy*factor);
                face[2] = convert_int_rte(windowsize.x*factor);
                face[3] = convert_int_rte(windowsize.y*factor);
            }
        }
    }
}
