#define CV_2PI 6.283185307179586476925286766559
#ifdef ForwardMap
__kernel void computeAngleRadius(__global float2* cp_sp, __global float* r, float maxRadius_width,  float PI2_height,  unsigned width, unsigned height)
{
    unsigned gid = get_global_id(0);
    if (gid < height)
    {
        float angle = gid * PI2_height;
        float2 angle_tri=(float2)(cos(angle), sin(angle));
        cp_sp[gid] = angle_tri;
    }
    if (gid < width)
    {
        r[gid] = maxRadius_width*gid;
    }
}
__kernel void linearPolar(__global float* mx, __global float* my, __global float2* cp_sp,  __global float* r, float cx, float cy, unsigned width, unsigned height)
{
    __local float l_r[MEM_SIZE];
    __local float2 l_double[MEM_SIZE];
    unsigned rho = get_global_id(0);

    unsigned phi = get_global_id(1);
    unsigned local_0 = get_local_id(0);
    unsigned local_1 = get_local_id(1);
    if (local_1 == 0)
    {
        unsigned temp_phi=phi + local_0;
        if (temp_phi < height)
        {
            l_double[local_0] = cp_sp[temp_phi];
        }
    }
    if (local_1 == 1 )
    {
        if (rho < width)
        {
            l_r[local_0 ] = r[rho];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (rho<width && phi<height)
    {
        unsigned g_id = rho + phi*width;
        float radius = l_r[local_0];
        float2 tri= l_double[local_1];
        mx[g_id] = fma(radius, tri.x , cx);
        my[g_id] = fma(radius, tri.y , cy);
    }
}
#elif defined (InverseMap)
__kernel void linearPolar(__global float* mx, __global float* my, float ascale, float pscale, float cx, float cy, int angle_border, unsigned width, unsigned height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < width && y < height)
    {
        unsigned g_id = x + y*width;
        float dx = (float)x - cx;
        float dy = (float)y - cy;
        float mag = sqrt(dx*dx + dy*dy);
        float angle = atan2(dy, dx);
        if (angle < 0)
            angle = angle + CV_2PI;
        mx[g_id] = mag*pscale;
        my[g_id] = (angle*ascale) + angle_border;
    }
}
#endif