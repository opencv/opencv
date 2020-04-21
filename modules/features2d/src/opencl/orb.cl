// OpenCL port of the ORB feature detector and descriptor extractor
// Copyright (C) 2014, Itseez Inc. See the license at http://opencv.org
//
// The original code has been contributed by Peter Andreas Entschev, peter@entschev.com

#define LAYERINFO_SIZE 1
#define LAYERINFO_OFS 0
#define KEYPOINT_SIZE 3
#define ORIENTED_KEYPOINT_SIZE 4
#define KEYPOINT_X 0
#define KEYPOINT_Y 1
#define KEYPOINT_Z 2
#define KEYPOINT_ANGLE 3

/////////////////////////////////////////////////////////////

#ifdef ORB_RESPONSES

__kernel void
ORB_HarrisResponses(__global const uchar* imgbuf, int imgstep, int imgoffset0,
                    __global const int* layerinfo, __global const int* keypoints,
                    __global float* responses, int nkeypoints )
{
    int idx = get_global_id(0);
    if( idx < nkeypoints )
    {
        __global const int* kpt = keypoints + idx*KEYPOINT_SIZE;
        __global const int* layer = layerinfo + kpt[KEYPOINT_Z]*LAYERINFO_SIZE;
        __global const uchar* img = imgbuf + imgoffset0 + layer[LAYERINFO_OFS] +
            (kpt[KEYPOINT_Y] - blockSize/2)*imgstep + (kpt[KEYPOINT_X] - blockSize/2);

        int i, j;
        int a = 0, b = 0, c = 0;
        for( i = 0; i < blockSize; i++, img += imgstep-blockSize )
        {
            for( j = 0; j < blockSize; j++, img++ )
            {
                int Ix = (img[1] - img[-1])*2 + img[-imgstep+1] - img[-imgstep-1] + img[imgstep+1] - img[imgstep-1];
                int Iy = (img[imgstep] - img[-imgstep])*2 + img[imgstep-1] - img[-imgstep-1] + img[imgstep+1] - img[-imgstep+1];
                a += Ix*Ix;
                b += Iy*Iy;
                c += Ix*Iy;
            }
        }
        responses[idx] = ((float)a * b - (float)c * c - HARRIS_K * (float)(a + b) * (a + b))*scale_sq_sq;
    }
}

#endif

/////////////////////////////////////////////////////////////

#ifdef ORB_ANGLES

#define _DBL_EPSILON 2.2204460492503131e-16f
#define atan2_p1 (0.9997878412794807f*57.29577951308232f)
#define atan2_p3 (-0.3258083974640975f*57.29577951308232f)
#define atan2_p5 (0.1555786518463281f*57.29577951308232f)
#define atan2_p7 (-0.04432655554792128f*57.29577951308232f)

inline float fastAtan2( float y, float x )
{
    float ax = fabs(x), ay = fabs(y);
    float a, c, c2;
    if( ax >= ay )
    {
        c = ay/(ax + _DBL_EPSILON);
        c2 = c*c;
        a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
    }
    else
    {
        c = ax/(ay + _DBL_EPSILON);
        c2 = c*c;
        a = 90.f - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
    }
    if( x < 0 )
        a = 180.f - a;
    if( y < 0 )
        a = 360.f - a;
    return a;
}


__kernel void
ORB_ICAngle(__global const uchar* imgbuf, int imgstep, int imgoffset0,
            __global const int* layerinfo, __global const int* keypoints,
            __global float* responses, const __global int* u_max,
            int nkeypoints, int half_k )
{
    int idx = get_global_id(0);
    if( idx < nkeypoints )
    {
        __global const int* kpt = keypoints + idx*KEYPOINT_SIZE;

        __global const int* layer = layerinfo + kpt[KEYPOINT_Z]*LAYERINFO_SIZE;
        __global const uchar* center = imgbuf + imgoffset0 + layer[LAYERINFO_OFS] +
            kpt[KEYPOINT_Y]*imgstep + kpt[KEYPOINT_X];

        int u, v, m_01 = 0, m_10 = 0;

        // Treat the center line differently, v=0
        for( u = -half_k; u <= half_k; u++ )
            m_10 += u * center[u];

        // Go line by line in the circular patch
        for( v = 1; v <= half_k; v++ )
        {
            // Proceed over the two lines
            int v_sum = 0;
            int d = u_max[v];
            for( u = -d; u <= d; u++ )
            {
                int val_plus = center[u + v*imgstep], val_minus = center[u - v*imgstep];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        // we do not use OpenCL's atan2 intrinsic,
        // because we want to get _exactly_ the same results as the CPU version
        responses[idx] = fastAtan2((float)m_01, (float)m_10);
    }
}

#endif

/////////////////////////////////////////////////////////////

#ifdef ORB_DESCRIPTORS

__kernel void
ORB_computeDescriptor(__global const uchar* imgbuf, int imgstep, int imgoffset0,
                      __global const int* layerinfo, __global const int* keypoints,
                      __global uchar* _desc, const __global int* pattern,
                      int nkeypoints, int dsize )
{
    int idx = get_global_id(0);
    if( idx < nkeypoints )
    {
        int i;
        __global const int* kpt = keypoints + idx*ORIENTED_KEYPOINT_SIZE;

        __global const int* layer = layerinfo + kpt[KEYPOINT_Z]*LAYERINFO_SIZE;
        __global const uchar* center = imgbuf + imgoffset0 + layer[LAYERINFO_OFS] +
                                kpt[KEYPOINT_Y]*imgstep + kpt[KEYPOINT_X];
        float angle = as_float(kpt[KEYPOINT_ANGLE]);
        angle *= 0.01745329251994329547f;

        float cosa;
        float sina = sincos(angle, &cosa);

        __global uchar* desc = _desc + idx*dsize;

        #define GET_VALUE(idx) \
            center[mad24(convert_int_rte(pattern[(idx)*2] * sina + pattern[(idx)*2+1] * cosa), imgstep, \
                        convert_int_rte(pattern[(idx)*2] * cosa - pattern[(idx)*2+1] * sina))]

        for( i = 0; i < dsize; i++ )
        {
            int val;
        #if WTA_K == 2
            int t0, t1;

            t0 = GET_VALUE(0); t1 = GET_VALUE(1);
            val = t0 < t1;

            t0 = GET_VALUE(2); t1 = GET_VALUE(3);
            val |= (t0 < t1) << 1;

            t0 = GET_VALUE(4); t1 = GET_VALUE(5);
            val |= (t0 < t1) << 2;

            t0 = GET_VALUE(6); t1 = GET_VALUE(7);
            val |= (t0 < t1) << 3;

            t0 = GET_VALUE(8); t1 = GET_VALUE(9);
            val |= (t0 < t1) << 4;

            t0 = GET_VALUE(10); t1 = GET_VALUE(11);
            val |= (t0 < t1) << 5;

            t0 = GET_VALUE(12); t1 = GET_VALUE(13);
            val |= (t0 < t1) << 6;

            t0 = GET_VALUE(14); t1 = GET_VALUE(15);
            val |= (t0 < t1) << 7;

            pattern += 16*2;

        #elif WTA_K == 3
            int t0, t1, t2;

            t0 = GET_VALUE(0); t1 = GET_VALUE(1); t2 = GET_VALUE(2);
            val = t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0);

            t0 = GET_VALUE(3); t1 = GET_VALUE(4); t2 = GET_VALUE(5);
            val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 2;

            t0 = GET_VALUE(6); t1 = GET_VALUE(7); t2 = GET_VALUE(8);
            val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 4;

            t0 = GET_VALUE(9); t1 = GET_VALUE(10); t2 = GET_VALUE(11);
            val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 6;

            pattern += 12*2;

        #elif WTA_K == 4
            int t0, t1, t2, t3, k;
            int a, b;

            t0 = GET_VALUE(0); t1 = GET_VALUE(1);
            t2 = GET_VALUE(2); t3 = GET_VALUE(3);
            a = 0, b = 2;
            if( t1 > t0 ) t0 = t1, a = 1;
            if( t3 > t2 ) t2 = t3, b = 3;
            k = t0 > t2 ? a : b;
            val = k;

            t0 = GET_VALUE(4); t1 = GET_VALUE(5);
            t2 = GET_VALUE(6); t3 = GET_VALUE(7);
            a = 0, b = 2;
            if( t1 > t0 ) t0 = t1, a = 1;
            if( t3 > t2 ) t2 = t3, b = 3;
            k = t0 > t2 ? a : b;
            val |= k << 2;

            t0 = GET_VALUE(8); t1 = GET_VALUE(9);
            t2 = GET_VALUE(10); t3 = GET_VALUE(11);
            a = 0, b = 2;
            if( t1 > t0 ) t0 = t1, a = 1;
            if( t3 > t2 ) t2 = t3, b = 3;
            k = t0 > t2 ? a : b;
            val |= k << 4;

            t0 = GET_VALUE(12); t1 = GET_VALUE(13);
            t2 = GET_VALUE(14); t3 = GET_VALUE(15);
            a = 0, b = 2;
            if( t1 > t0 ) t0 = t1, a = 1;
            if( t3 > t2 ) t2 = t3, b = 3;
            k = t0 > t2 ? a : b;
            val |= k << 6;

            pattern += 16*2;
        #else
            #error "unknown/undefined WTA_K value; should be 2, 3 or 4"
        #endif
            desc[i] = (uchar)val;
        }
    }
}

#endif
