// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

//#define CV_USE_SUBGROUPS

#define EPS 0.001f
#define INF 1E+10F

//#define DIS_BORDER_SIZE xxx
//#define DIS_PATCH_SIZE xxx
//#define DIS_PATCH_STRIDE xxx

#define DIS_PATCH_SIZE_HALF (DIS_PATCH_SIZE / 2)

#ifndef DIS_BORDER_SIZE

__kernel void dis_precomputeStructureTensor_hor(__global const short *I0x,
                                                __global const short *I0y,
                                                int w, int h, int ws,
                                                __global float *I0xx_aux_ptr,
                                                __global float *I0yy_aux_ptr,
                                                __global float *I0xy_aux_ptr,
                                                __global float *I0x_aux_ptr,
                                                __global float *I0y_aux_ptr)
{

    int i = get_global_id(0);

    if (i >= h) return;

    const __global short *x_row = I0x + i * w;
    const __global short *y_row = I0y + i * w;

    float sum_xx = 0.0f, sum_yy = 0.0f, sum_xy = 0.0f, sum_x = 0.0f, sum_y = 0.0f;
    float8 x_vec = convert_float8(vload8(0, x_row));
    float8 y_vec = convert_float8(vload8(0, y_row));
    sum_xx = dot(x_vec.lo, x_vec.lo) + dot(x_vec.hi, x_vec.hi);
    sum_yy = dot(y_vec.lo, y_vec.lo) + dot(y_vec.hi, y_vec.hi);
    sum_xy = dot(x_vec.lo, y_vec.lo) + dot(x_vec.hi, y_vec.hi);
    sum_x = dot(x_vec.lo, 1.0f) + dot(x_vec.hi, 1.0f);
    sum_y = dot(y_vec.lo, 1.0f) + dot(y_vec.hi, 1.0f);

    I0xx_aux_ptr[i * ws] = sum_xx;
    I0yy_aux_ptr[i * ws] = sum_yy;
    I0xy_aux_ptr[i * ws] = sum_xy;
    I0x_aux_ptr[i * ws] = sum_x;
    I0y_aux_ptr[i * ws] = sum_y;

    int js = 1;
    for (int j = DIS_PATCH_SIZE; j < w; j++)
    {
        short x_val1 = x_row[j];
        short x_val2 = x_row[j - DIS_PATCH_SIZE];
        short y_val1 = y_row[j];
        short y_val2 = y_row[j - DIS_PATCH_SIZE];
        sum_xx += (x_val1 * x_val1 - x_val2 * x_val2);
        sum_yy += (y_val1 * y_val1 - y_val2 * y_val2);
        sum_xy += (x_val1 * y_val1 - x_val2 * y_val2);
        sum_x += (x_val1 - x_val2);
        sum_y += (y_val1 - y_val2);
        if ((j - DIS_PATCH_SIZE + 1) % DIS_PATCH_STRIDE == 0)
        {
            int index = i * ws + js;
            I0xx_aux_ptr[index] = sum_xx;
            I0yy_aux_ptr[index] = sum_yy;
            I0xy_aux_ptr[index] = sum_xy;
            I0x_aux_ptr[index] = sum_x;
            I0y_aux_ptr[index] = sum_y;
            js++;
        }
    }
}

__kernel void dis_precomputeStructureTensor_ver(__global const float *I0xx_aux_ptr,
                                                __global const float *I0yy_aux_ptr,
                                                __global const float *I0xy_aux_ptr,
                                                __global const float *I0x_aux_ptr,
                                                __global const float *I0y_aux_ptr,
                                                int w, int h, int ws,
                                                __global float *I0xx_ptr,
                                                __global float *I0yy_ptr,
                                                __global float *I0xy_ptr,
                                                __global float *I0x_ptr,
                                                __global float *I0y_ptr)
{
    int j = get_global_id(0);

    if (j >= ws) return;

    float sum_xx, sum_yy, sum_xy, sum_x, sum_y;
    sum_xx = sum_yy = sum_xy = sum_x = sum_y = 0.0f;

    for (int i = 0; i < DIS_PATCH_SIZE; i++)
    {
        sum_xx += I0xx_aux_ptr[i * ws + j];
        sum_yy += I0yy_aux_ptr[i * ws + j];
        sum_xy += I0xy_aux_ptr[i * ws + j];
        sum_x  += I0x_aux_ptr[i * ws + j];
        sum_y  += I0y_aux_ptr[i * ws + j];
    }
    I0xx_ptr[j] = sum_xx;
    I0yy_ptr[j] = sum_yy;
    I0xy_ptr[j] = sum_xy;
    I0x_ptr[j] = sum_x;
    I0y_ptr[j] = sum_y;

    int is = 1;
    for (int i = DIS_PATCH_SIZE; i < h; i++)
    {
        sum_xx += (I0xx_aux_ptr[i * ws + j] - I0xx_aux_ptr[(i - DIS_PATCH_SIZE) * ws + j]);
        sum_yy += (I0yy_aux_ptr[i * ws + j] - I0yy_aux_ptr[(i - DIS_PATCH_SIZE) * ws + j]);
        sum_xy += (I0xy_aux_ptr[i * ws + j] - I0xy_aux_ptr[(i - DIS_PATCH_SIZE) * ws + j]);
        sum_x  += (I0x_aux_ptr[i * ws + j] - I0x_aux_ptr[(i - DIS_PATCH_SIZE) * ws + j]);
        sum_y  += (I0y_aux_ptr[i * ws + j] - I0y_aux_ptr[(i - DIS_PATCH_SIZE) * ws + j]);

        if ((i - DIS_PATCH_SIZE + 1) % DIS_PATCH_STRIDE == 0)
        {
            I0xx_ptr[is * ws + j] = sum_xx;
            I0yy_ptr[is * ws + j] = sum_yy;
            I0xy_ptr[is * ws + j] = sum_xy;
            I0x_ptr[is * ws + j] = sum_x;
            I0y_ptr[is * ws + j] = sum_y;
            is++;
        }
    }
}

__kernel void dis_densification(__global const float2 *S_ptr,
                                __global const uchar *i0, __global const uchar *i1,
                                int w, int h, int ws,
                                __global float2 *U_ptr)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int i, j;

    if (x >= w || y >= h) return;

    int start_is, end_is;
    int start_js, end_js;

    end_is = min(y / DIS_PATCH_STRIDE, (h - DIS_PATCH_SIZE) / DIS_PATCH_STRIDE);
    start_is = max(0, y - DIS_PATCH_SIZE + DIS_PATCH_STRIDE) / DIS_PATCH_STRIDE;
    start_is = min(start_is, end_is);

    end_js = min(x / DIS_PATCH_STRIDE, (w - DIS_PATCH_SIZE) / DIS_PATCH_STRIDE);
    start_js = max(0, x - DIS_PATCH_SIZE + DIS_PATCH_STRIDE) / DIS_PATCH_STRIDE;
    start_js = min(start_js, end_js);

    float sum_coef = 0.0f;
    float2 sum_U = (float2)(0.0f, 0.0f);

    int i_l, i_u;
    int j_l, j_u;
    float i_m, j_m, diff;

    i = y;
    j = x;

    /* Iterate through all the patches that overlap the current location (i,j) */
    for (int is = start_is; is <= end_is; is++)
        for (int js = start_js; js <= end_js; js++)
        {
            float2 s_val = S_ptr[is * ws + js];
            uchar2 i1_vec1, i1_vec2;

            j_m = min(max(j + s_val.x, 0.0f), w - 1.0f - EPS);
            i_m = min(max(i + s_val.y, 0.0f), h - 1.0f - EPS);
            j_l = (int)j_m;
            j_u = j_l + 1;
            i_l = (int)i_m;
            i_u = i_l + 1;
            i1_vec1 = vload2(0, i1 + i_u * w + j_l);
            i1_vec2 = vload2(0, i1 + i_l * w + j_l);
            diff = (j_m - j_l) * (i_m - i_l) * i1_vec1.y +
                   (j_u - j_m) * (i_m - i_l) * i1_vec1.x +
                   (j_m - j_l) * (i_u - i_m) * i1_vec2.y +
                   (j_u - j_m) * (i_u - i_m) * i1_vec2.x - i0[i * w + j];
            float coef = 1.0f / max(1.0f, fabs(diff));
            sum_U += coef * s_val;
            sum_coef += coef;
        }

    float inv_sum_coef = 1.0 / sum_coef;
    U_ptr[i * w + j] = sum_U * inv_sum_coef;
}

#else // DIS_BORDER_SIZE

#define INIT_BILINEAR_WEIGHTS(Ux, Uy) \
    i_I1 = clamp(i + Uy + DIS_BORDER_SIZE, i_lower_limit, i_upper_limit); \
    j_I1 = clamp(j + Ux + DIS_BORDER_SIZE, j_lower_limit, j_upper_limit); \
    { \
        float di = i_I1 - floor(i_I1); \
        float dj = j_I1 - floor(j_I1); \
        w11 = di       * dj; \
        w10 = di       * (1 - dj); \
        w01 = (1 - di) * dj; \
        w00 = (1 - di) * (1 - dj); \
    }

float computeSSDMeanNorm(const __global uchar *I0_ptr, const __global uchar *I1_ptr,
                         int I0_stride, int I1_stride,
                         float w00, float w01, float w10, float w11, int i
#ifndef CV_USE_SUBGROUPS
                         , __local float2 *smem /*[8]*/
#endif
)
{
    float sum_diff = 0.0f, sum_diff_sq = 0.0f;
    int n = DIS_PATCH_SIZE * DIS_PATCH_SIZE;

    uchar8 I1_vec1, I1_vec2, I0_vec;
    uchar I1_val1, I1_val2;

    I0_vec  = vload8(0, I0_ptr + i * I0_stride);
    I1_vec1 = vload8(0, I1_ptr + i * I1_stride);
    I1_vec2 = vload8(0, I1_ptr + (i + 1) * I1_stride);
    I1_val1 = I1_ptr[i * I1_stride + 8];
    I1_val2 = I1_ptr[(i + 1) * I1_stride + 8];

    float8 vec = w00 * convert_float8(I1_vec1) + w01 * convert_float8((uchar8)(I1_vec1.s123, I1_vec1.s4567, I1_val1)) +
                 w10 * convert_float8(I1_vec2) + w11 * convert_float8((uchar8)(I1_vec2.s123, I1_vec2.s4567, I1_val2)) -
                 convert_float8(I0_vec);

    sum_diff = (dot(vec.lo, 1.0) + dot(vec.hi, 1.0));
    sum_diff_sq = (dot(vec.lo, vec.lo) + dot(vec.hi, vec.hi));

#ifdef CV_USE_SUBGROUPS
    sum_diff = sub_group_reduce_add(sum_diff);
    sum_diff_sq = sub_group_reduce_add(sum_diff_sq);
#else
    barrier(CLK_LOCAL_MEM_FENCE);
    smem[i] = (float2)(sum_diff, sum_diff_sq);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i < 4)
        smem[i] += smem[i + 4];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i < 2)
        smem[i] += smem[i + 2];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i == 0)
        smem[0] += smem[1];
    barrier(CLK_LOCAL_MEM_FENCE);
    float2 reduce_add_result = smem[0];
    sum_diff = reduce_add_result.x;
    sum_diff_sq = reduce_add_result.y;
#endif

    return sum_diff_sq - sum_diff * sum_diff / n;
}

__attribute__((reqd_work_group_size(8, 1, 1)))
__kernel void dis_patch_inverse_search_fwd_1(__global const float2 *U_ptr,
                                             __global const uchar *I0_ptr, __global const uchar *I1_ptr,
                                             int w, int h, int ws, int hs,
                                             __global float2 *S_ptr)
{
    int id = get_global_id(0);
    int is = get_group_id(0);

    int i = is * DIS_PATCH_STRIDE;
    int j = 0;
    int w_ext = w + 2 * DIS_BORDER_SIZE;

    float i_lower_limit = DIS_BORDER_SIZE - DIS_PATCH_SIZE + 1.0f;
    float i_upper_limit = DIS_BORDER_SIZE + h - 1.0f;
    float j_lower_limit = DIS_BORDER_SIZE - DIS_PATCH_SIZE + 1.0f;
    float j_upper_limit = DIS_BORDER_SIZE + w - 1.0f;

    float2 prev_U = U_ptr[(i + DIS_PATCH_SIZE_HALF) * w + j + DIS_PATCH_SIZE_HALF];
    S_ptr[is * ws] = prev_U;
    j += DIS_PATCH_STRIDE;

#ifdef CV_USE_SUBGROUPS
    int sid = get_sub_group_local_id();
#define EXTRA_ARGS_computeSSDMeanNorm sid
#else
    __local float2 smem[8];
    int sid = get_local_id(0);
#define EXTRA_ARGS_computeSSDMeanNorm sid, smem
#endif
    for (int js = 1; js < ws; js++, j += DIS_PATCH_STRIDE)
    {
        float2 U = U_ptr[(i + DIS_PATCH_SIZE_HALF) * w + j + DIS_PATCH_SIZE_HALF];

        float i_I1, j_I1, w00, w01, w10, w11;

        INIT_BILINEAR_WEIGHTS(U.x, U.y);
        float min_SSD = computeSSDMeanNorm(
                I0_ptr + i * w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                w, w_ext, w00, w01, w10, w11, EXTRA_ARGS_computeSSDMeanNorm);

        INIT_BILINEAR_WEIGHTS(prev_U.x, prev_U.y);
        float cur_SSD = computeSSDMeanNorm(
                I0_ptr + i * w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                w, w_ext, w00, w01, w10, w11, EXTRA_ARGS_computeSSDMeanNorm);

        prev_U = (cur_SSD < min_SSD) ? prev_U : U;
        S_ptr[is * ws + js] = prev_U;
    }

#undef EXTRA_ARGS_computeSSDMeanNorm
}
#endif // DIS_BORDER_SIZE

float4 processPatchMeanNorm(const __global uchar *I0_ptr, const __global uchar *I1_ptr,
                            const __global short *I0x_ptr, const __global short *I0y_ptr,
                            int I0_stride, int I1_stride, float w00, float w01, float w10,
                            float w11, float x_grad_sum, float y_grad_sum)
{
    const float inv_n = 1.0f / (float)(DIS_PATCH_SIZE * DIS_PATCH_SIZE);

    float sum_diff = 0.0, sum_diff_sq = 0.0;
    float sum_I0x_mul = 0.0, sum_I0y_mul = 0.0;

    uchar8 I1_vec1;
    uchar8 I1_vec2 = vload8(0, I1_ptr);
    uchar I1_val1;
    uchar I1_val2 = I1_ptr[DIS_PATCH_SIZE];

    for (int i = 0; i < 8; i++)
    {
        uchar8 I0_vec = vload8(0, I0_ptr + i * I0_stride);

        I1_vec1 = I1_vec2;
        I1_vec2 = vload8(0, I1_ptr + (i + 1) * I1_stride);
        I1_val1 = I1_val2;
        I1_val2 = I1_ptr[(i + 1) * I1_stride + DIS_PATCH_SIZE];

        float8 vec = w00 * convert_float8(I1_vec1) + w01 * convert_float8((uchar8)(I1_vec1.s123, I1_vec1.s4567, I1_val1)) +
                     w10 * convert_float8(I1_vec2) + w11 * convert_float8((uchar8)(I1_vec2.s123, I1_vec2.s4567, I1_val2)) -
                     convert_float8(I0_vec);

        sum_diff += (dot(vec.lo, 1.0) + dot(vec.hi, 1.0));
        sum_diff_sq += (dot(vec.lo, vec.lo) + dot(vec.hi, vec.hi));

        short8 I0x_vec = vload8(0, I0x_ptr + i * I0_stride);
        short8 I0y_vec = vload8(0, I0y_ptr + i * I0_stride);

        sum_I0x_mul += dot(vec.lo, convert_float4(I0x_vec.lo));
        sum_I0x_mul += dot(vec.hi, convert_float4(I0x_vec.hi));
        sum_I0y_mul += dot(vec.lo, convert_float4(I0y_vec.lo));
        sum_I0y_mul += dot(vec.hi, convert_float4(I0y_vec.hi));
    }

    float dst_dUx = sum_I0x_mul - sum_diff * x_grad_sum * inv_n;
    float dst_dUy = sum_I0y_mul - sum_diff * y_grad_sum * inv_n;
    float SSD = sum_diff_sq - sum_diff * sum_diff * inv_n;

    return (float4)(SSD, dst_dUx, dst_dUy, 0);
}

#ifdef DIS_BORDER_SIZE
__kernel void dis_patch_inverse_search_fwd_2(__global const float2 *U_ptr,
                                             __global const uchar *I0_ptr, __global const uchar *I1_ptr,
                                             __global const short *I0x_ptr, __global const short *I0y_ptr,
                                             __global const float *xx_ptr, __global const float *yy_ptr,
                                             __global const float *xy_ptr,
                                             __global const float *x_ptr, __global const float *y_ptr,
                                             int w, int h, int ws, int hs, int num_inner_iter,
                                             __global float2 *S_ptr)
{
    int js = get_global_id(0);
    int is = get_global_id(1);
    int i = is * DIS_PATCH_STRIDE;
    int j = js * DIS_PATCH_STRIDE;
    const int psz = DIS_PATCH_SIZE;
    int w_ext = w + 2 * DIS_BORDER_SIZE;
    int index = is * ws + js;

    if (js >= ws || is >= hs) return;

    float2 U0 = S_ptr[index];
    float2 cur_U = U0;
    float cur_xx = xx_ptr[index];
    float cur_yy = yy_ptr[index];
    float cur_xy = xy_ptr[index];
    float detH = cur_xx * cur_yy - cur_xy * cur_xy;

    float inv_detH = (fabs(detH) < EPS) ? 1.0 / EPS : 1.0 / detH;
    float invH11 = cur_yy * inv_detH;
    float invH12 = -cur_xy * inv_detH;
    float invH22 = cur_xx * inv_detH;

    float prev_SSD = INF;
    float x_grad_sum = x_ptr[index];
    float y_grad_sum = y_ptr[index];

    const float i_lower_limit = DIS_BORDER_SIZE - DIS_PATCH_SIZE + 1.0f;
    const float i_upper_limit = DIS_BORDER_SIZE + h - 1.0f;
    const float j_lower_limit = DIS_BORDER_SIZE - DIS_PATCH_SIZE + 1.0f;
    const float j_upper_limit = DIS_BORDER_SIZE + w - 1.0f;

    for (int t = 0; t < num_inner_iter; t++)
    {
        float i_I1, j_I1, w00, w01, w10, w11;
        INIT_BILINEAR_WEIGHTS(cur_U.x, cur_U.y);
        float4 res = processPatchMeanNorm(
                I0_ptr  + i * w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                I0x_ptr + i * w + j, I0y_ptr + i * w + j,
                w, w_ext, w00, w01, w10, w11,
                x_grad_sum, y_grad_sum);

        float SSD = res.x;
        float dUx = res.y;
        float dUy = res.z;
        float dx = invH11 * dUx + invH12 * dUy;
        float dy = invH12 * dUx + invH22 * dUy;

        cur_U -= (float2)(dx, dy);

        if (SSD >= prev_SSD)
            break;
        prev_SSD = SSD;
    }

    float2 vec = cur_U - U0;
    S_ptr[index] = (dot(vec, vec) <= (float)(DIS_PATCH_SIZE * DIS_PATCH_SIZE)) ? cur_U : U0;
}

__attribute__((reqd_work_group_size(8, 1, 1)))
__kernel void dis_patch_inverse_search_bwd_1(__global const uchar *I0_ptr, __global const uchar *I1_ptr,
                                             int w, int h, int ws, int hs,
                                             __global float2 *S_ptr)
{
    int id = get_global_id(0);
    int is = get_group_id(0);

    is = (hs - 1 - is);
    int i = is * DIS_PATCH_STRIDE;
    int j = (ws - 2) * DIS_PATCH_STRIDE;
    const int w_ext = w + 2 * DIS_BORDER_SIZE;

    const float i_lower_limit = DIS_BORDER_SIZE - DIS_PATCH_SIZE + 1.0f;
    const float i_upper_limit = DIS_BORDER_SIZE + h - 1.0f;
    const float j_lower_limit = DIS_BORDER_SIZE - DIS_PATCH_SIZE + 1.0f;
    const float j_upper_limit = DIS_BORDER_SIZE + w - 1.0f;

#ifdef CV_USE_SUBGROUPS
    int sid = get_sub_group_local_id();
#define EXTRA_ARGS_computeSSDMeanNorm sid
#else
    __local float2 smem[8];
    int sid = get_local_id(0);
#define EXTRA_ARGS_computeSSDMeanNorm sid, smem
#endif

    for (int js = (ws - 2); js > -1; js--, j -= DIS_PATCH_STRIDE)
    {
        float2 U0 = S_ptr[is * ws + js];
        float2 U1 = S_ptr[is * ws + js + 1];

        float i_I1, j_I1, w00, w01, w10, w11;

        INIT_BILINEAR_WEIGHTS(U0.x, U0.y);
        float min_SSD = computeSSDMeanNorm(
                I0_ptr + i * w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                w, w_ext, w00, w01, w10, w11, EXTRA_ARGS_computeSSDMeanNorm);

        INIT_BILINEAR_WEIGHTS(U1.x, U1.y);
        float cur_SSD = computeSSDMeanNorm(
                I0_ptr + i * w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                w, w_ext, w00, w01, w10, w11, EXTRA_ARGS_computeSSDMeanNorm);

        S_ptr[is * ws + js] = (cur_SSD < min_SSD) ? U1 : U0;
    }

#undef EXTRA_ARGS_computeSSDMeanNorm
}

__kernel void dis_patch_inverse_search_bwd_2(__global const uchar *I0_ptr, __global const uchar *I1_ptr,
                                             __global const short *I0x_ptr, __global const short *I0y_ptr,
                                             __global const float *xx_ptr, __global const float *yy_ptr,
                                             __global const float *xy_ptr,
                                             __global const float *x_ptr, __global const float *y_ptr,
                                             int w, int h, int ws, int hs, int num_inner_iter,
                                             __global float2 *S_ptr)
{
    int js = get_global_id(0);
    int is = get_global_id(1);
    if (js >= ws || is >= hs) return;

    js = (ws - 1 - js);
    is = (hs - 1 - is);

    int j = js * DIS_PATCH_STRIDE;
    int i = is * DIS_PATCH_STRIDE;
    int w_ext = w + 2 * DIS_BORDER_SIZE;
    int index = is * ws + js;

    float2 U0 = S_ptr[index];
    float2 cur_U = U0;
    float cur_xx = xx_ptr[index];
    float cur_yy = yy_ptr[index];
    float cur_xy = xy_ptr[index];
    float detH = cur_xx * cur_yy - cur_xy * cur_xy;

    float inv_detH = (fabs(detH) < EPS) ? 1.0 / EPS : 1.0 / detH;
    float invH11 = cur_yy * inv_detH;
    float invH12 = -cur_xy * inv_detH;
    float invH22 = cur_xx * inv_detH;

    float prev_SSD = INF;
    float x_grad_sum = x_ptr[index];
    float y_grad_sum = y_ptr[index];

    const float i_lower_limit = DIS_BORDER_SIZE - DIS_PATCH_SIZE + 1.0f;
    const float i_upper_limit = DIS_BORDER_SIZE + h - 1.0f;
    const float j_lower_limit = DIS_BORDER_SIZE - DIS_PATCH_SIZE + 1.0f;
    const float j_upper_limit = DIS_BORDER_SIZE + w - 1.0f;

    for (int t = 0; t < num_inner_iter; t++)
    {
        float i_I1, j_I1, w00, w01, w10, w11;
        INIT_BILINEAR_WEIGHTS(cur_U.x, cur_U.y);
        float4 res = processPatchMeanNorm(
                I0_ptr  + i * w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                I0x_ptr + i * w + j, I0y_ptr + i * w + j,
                w, w_ext, w00, w01, w10, w11,
                x_grad_sum, y_grad_sum);

        float SSD = res.x;
        float dUx = res.y;
        float dUy = res.z;
        float dx = invH11 * dUx + invH12 * dUy;
        float dy = invH12 * dUx + invH22 * dUy;

        cur_U -= (float2)(dx, dy);

        if (SSD >= prev_SSD)
            break;
        prev_SSD = SSD;
    }

    float2 vec = cur_U - U0;
    S_ptr[index] = ((dot(vec, vec)) <= (float)(DIS_PATCH_SIZE * DIS_PATCH_SIZE)) ? cur_U : U0;
}
#endif // DIS_BORDER_SIZE
