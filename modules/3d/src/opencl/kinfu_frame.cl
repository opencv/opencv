// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

inline float3 reproject(float3 p, float2 fxyinv, float2 cxy)
{
    float2 pp = p.z*(p.xy - cxy)*fxyinv;
    return (float3)(pp, p.z);
}

typedef float4 ptype;

__kernel void computePointsNormals(__global char * pointsptr,
                                   int points_step, int points_offset,
                                   __global char * normalsptr,
                                   int normals_step, int normals_offset,
                                   __global const char * depthptr,
                                   int depth_step, int depth_offset,
                                   int depth_rows, int depth_cols,
                                   const float2 fxyinv,
                                   const float2 cxy,
                                   const float dfac
                                    )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= depth_cols || y >= depth_rows)
        return;

    __global const float* row0 = (__global const float*)(depthptr + depth_offset +
                                                         (y+0)*depth_step);
    __global const float* row1 = (__global const float*)(depthptr + depth_offset +
                                                         (y+1)*depth_step);

    float d00 = row0[x];
    float z00 = d00*dfac;
    float3 p00 = (float3)(convert_float2((int2)(x, y)), z00);
    float3 v00 = reproject(p00, fxyinv, cxy);

    float3 p = nan((uint)0), n = nan((uint)0);

    if(x < depth_cols - 1 && y < depth_rows - 1)
    {
        float d01 = row0[x+1];
        float d10 = row1[x];

        float z01 = d01*dfac;
        float z10 = d10*dfac;

        if(z00 != 0 && z01 != 0 && z10 != 0)
        {
            float3 p01 = (float3)(convert_float2((int2)(x+1, y+0)), z01);
            float3 p10 = (float3)(convert_float2((int2)(x+0, y+1)), z10);
            float3 v01 = reproject(p01, fxyinv, cxy);
            float3 v10 = reproject(p10, fxyinv, cxy);

            float3 vec = cross(v01 - v00, v10 - v00);
            n = - normalize(vec);
            p = v00;
        }
    }

    __global float* pts = (__global float*)(pointsptr  +  points_offset + y*points_step  + x*sizeof(ptype));
    __global float* nrm = (__global float*)(normalsptr + normals_offset + y*normals_step + x*sizeof(ptype));
    vstore4((ptype)(p, 0), 0, pts);
    vstore4((ptype)(n, 0), 0, nrm);
}

__kernel void pyrDownBilateral(__global const char * depthptr,
                               int depth_step, int depth_offset,
                               int depth_rows, int depth_cols,
                               __global char * depthDownptr,
                               int depthDown_step, int depthDown_offset,
                               int depthDown_rows, int depthDown_cols,
                               const float sigma
                               )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= depthDown_cols || y >= depthDown_rows)
        return;

    const float sigma3 = sigma*3;
    const int D = 5;

    __global const float* srcCenterRow = (__global const float*)(depthptr + depth_offset +
                                                                 (2*y)*depth_step);

    float center = srcCenterRow[2*x];

    int sx = max(0, 2*x - D/2), ex = min(2*x - D/2 + D, depth_cols-1);
    int sy = max(0, 2*y - D/2), ey = min(2*y - D/2 + D, depth_rows-1);

    float sum = 0;
    int count = 0;

    for(int iy = sy; iy < ey; iy++)
    {
        __global const float* srcRow = (__global const float*)(depthptr + depth_offset +
                                                               (iy)*depth_step);
        for(int ix = sx; ix < ex; ix++)
        {
            float val = srcRow[ix];
            if(fabs(val - center) < sigma3)
            {
                sum += val; count++;
            }
        }
    }

    __global float* downRow = (__global float*)(depthDownptr + depthDown_offset +
                                                y*depthDown_step + x*sizeof(float));

    *downRow = (count == 0) ? 0 : sum/convert_float(count);
}

//TODO: remove bilateral when OpenCV performs 32f bilat with OpenCL

__kernel void customBilateral(__global const char * srcptr,
                              int src_step, int src_offset,
                              __global char * dstptr,
                              int dst_step, int dst_offset,
                              const int2 frameSize,
                              const int kernelSize,
                              const float sigma_spatial2_inv_half,
                              const float sigma_depth2_inv_half
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= frameSize.x || y >= frameSize.y)
        return;

    __global const float* srcCenterRow = (__global const float*)(srcptr + src_offset +
                                                                 y*src_step);
    float value = srcCenterRow[x];

    int tx = min (x - kernelSize / 2 + kernelSize, frameSize.x - 1);
    int ty = min (y - kernelSize / 2 + kernelSize, frameSize.y - 1);

    float sum1 = 0;
    float sum2 = 0;

    for (int cy = max (y - kernelSize / 2, 0); cy < ty; ++cy)
    {
        __global const float* srcRow = (__global const float*)(srcptr + src_offset +
                                                               cy*src_step);
        for (int cx = max (x - kernelSize / 2, 0); cx < tx; ++cx)
        {
            float depth = srcRow[cx];

            float space2 = convert_float((x - cx) * (x - cx) + (y - cy) * (y - cy));
            float color2 = (value - depth) * (value - depth);

            float weight = native_exp (-(space2 * sigma_spatial2_inv_half +
                                         color2 * sigma_depth2_inv_half));

            sum1 += depth * weight;
            sum2 += weight;
        }
    }

    __global float* dst = (__global float*)(dstptr + dst_offset +
                                            y*dst_step + x*sizeof(float));
    *dst = sum1/sum2;
}

__kernel void pyrDownPointsNormals(__global const char * pptr,
                                   int p_step, int p_offset,
                                   __global const char * nptr,
                                   int n_step, int n_offset,
                                   __global char * pdownptr,
                                   int pdown_step, int pdown_offset,
                                   __global char * ndownptr,
                                   int ndown_step, int ndown_offset,
                                   const int2 downSize
                                   )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= downSize.x || y >= downSize.y)
        return;

    float3 point = nan((uint)0), normal = nan((uint)0);

    __global const ptype* pUpRow0 = (__global const ptype*)(pptr + p_offset + (2*y  )*p_step);
    __global const ptype* pUpRow1 = (__global const ptype*)(pptr + p_offset + (2*y+1)*p_step);

    float3 d00 = pUpRow0[2*x  ].xyz;
    float3 d01 = pUpRow0[2*x+1].xyz;
    float3 d10 = pUpRow1[2*x  ].xyz;
    float3 d11 = pUpRow1[2*x+1].xyz;

    if(!(any(isnan(d00)) || any(isnan(d01)) ||
         any(isnan(d10)) || any(isnan(d11))))
    {
        point = (d00 + d01 + d10 + d11)*0.25f;

        __global const ptype* nUpRow0 = (__global const ptype*)(nptr + n_offset + (2*y  )*n_step);
        __global const ptype* nUpRow1 = (__global const ptype*)(nptr + n_offset + (2*y+1)*n_step);

        float3 n00 = nUpRow0[2*x  ].xyz;
        float3 n01 = nUpRow0[2*x+1].xyz;
        float3 n10 = nUpRow1[2*x  ].xyz;
        float3 n11 = nUpRow1[2*x+1].xyz;

        normal = (n00 + n01 + n10 + n11)*0.25f;
    }

    __global ptype* pts = (__global ptype*)(pdownptr + pdown_offset + y*pdown_step);
    __global ptype* nrm = (__global ptype*)(ndownptr + ndown_offset + y*ndown_step);
    pts[x] = (ptype)(point,  0);
    nrm[x] = (ptype)(normal, 0);
}

typedef char4 pixelType;

// 20 is fixed power
float specPow20(float x)
{
    float x2 = x*x;
    float x5 = x2*x2*x;
    float x10 = x5*x5;
    float x20 = x10*x10;
    return x20;
}

__kernel void render(__global const char * pointsptr,
                     int points_step, int points_offset,
                     __global const char * normalsptr,
                     int normals_step, int normals_offset,
                     __global char * imgptr,
                     int img_step, int img_offset,
                     const int2 frameSize,
                     const float4 lightPt
                    )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= frameSize.x || y >= frameSize.y)
        return;

    __global const ptype* ptsRow = (__global const ptype*)(pointsptr  + points_offset  + y*points_step  + x*sizeof(ptype));
    __global const ptype* nrmRow = (__global const ptype*)(normalsptr + normals_offset + y*normals_step + x*sizeof(ptype));

    float3 p = (*ptsRow).xyz;
    float3 n = (*nrmRow).xyz;

    pixelType color;

    if(any(isnan(p)))
    {
        color = (pixelType)(0, 32, 0, 0);
    }
    else
    {
        const float Ka = 0.3f;  //ambient coeff
        const float Kd = 0.5f;  //diffuse coeff
        const float Ks = 0.2f;  //specular coeff
        //const int   sp = 20;  //specular power, fixed in specPow20()

        const float Ax = 1.f;   //ambient color,  can be RGB
        const float Dx = 1.f;   //diffuse color,  can be RGB
        const float Sx = 1.f;   //specular color, can be RGB
        const float Lx = 1.f;   //light color

        float3 l = normalize(lightPt.xyz - p);
        float3 v = normalize(-p);
        float3 r = normalize(2.f*n*dot(n, l) - l);

        float val = (Ax*Ka*Dx + Lx*Kd*Dx*max(0.f, dot(n, l)) +
                     Lx*Ks*Sx*specPow20(max(0.f, dot(r, v))));

        uchar ix = convert_uchar(val*255.f);
        color = (pixelType)(ix, ix, ix, 0);
    }

    __global char* imgRow = (__global char*)(imgptr + img_offset + y*img_step + x*sizeof(pixelType));
    vstore4(color, 0, imgRow);
}
