// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#define EPS 0.001f
#define INF 1E+10F

__kernel void dis_precomputeStructureTensor_hor(__global const short *I0x,
                                                __global const short *I0y,
                                                int patch_size, int patch_stride,
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
    for (int j = patch_size; j < w; j++)
    {
        short x_val1 = x_row[j];
        short x_val2 = x_row[j - patch_size];
        short y_val1 = y_row[j];
        short y_val2 = y_row[j - patch_size];
        sum_xx += (x_val1 * x_val1 - x_val2 * x_val2);
        sum_yy += (y_val1 * y_val1 - y_val2 * y_val2);
        sum_xy += (x_val1 * y_val1 - x_val2 * y_val2);
        sum_x += (x_val1 - x_val2);
        sum_y += (y_val1 - y_val2);
        if ((j - patch_size + 1) % patch_stride == 0)
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
                                                int patch_size, int patch_stride,
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

    for (int i = 0; i < patch_size; i++)
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
    for (int i = patch_size; i < h; i++)
    {
        sum_xx += (I0xx_aux_ptr[i * ws + j] - I0xx_aux_ptr[(i - patch_size) * ws + j]);
        sum_yy += (I0yy_aux_ptr[i * ws + j] - I0yy_aux_ptr[(i - patch_size) * ws + j]);
        sum_xy += (I0xy_aux_ptr[i * ws + j] - I0xy_aux_ptr[(i - patch_size) * ws + j]);
        sum_x  += (I0x_aux_ptr[i * ws + j] - I0x_aux_ptr[(i - patch_size) * ws + j]);
        sum_y  += (I0y_aux_ptr[i * ws + j] - I0y_aux_ptr[(i - patch_size) * ws + j]);

        if ((i - patch_size + 1) % patch_stride == 0)
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

__kernel void dis_densification(__global const float *sx, __global const float *sy,
                                __global const uchar *i0, __global const uchar *i1,
                                int psz, int pstr,
                                int w, int h, int ws,
                                __global float *ux, __global float *uy)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int i, j;

    if (x >= w || y >= h) return;

    int start_is, end_is;
    int start_js, end_js;

    end_is = min(y / pstr, (h - psz) / pstr);
    start_is = max(0, y - psz + pstr) / pstr;
    start_is = min(start_is, end_is);

    end_js = min(x / pstr, (w - psz) / pstr);
    start_js = max(0, x - psz + pstr) / pstr;
    start_js = min(start_js, end_js);

    float coef, sum_coef = 0.0f;
    float sum_Ux = 0.0f;
    float sum_Uy = 0.0f;

    int i_l, i_u;
    int j_l, j_u;
    float i_m, j_m, diff;

    i = y;
    j = x;

    /* Iterate through all the patches that overlap the current location (i,j) */
    for (int is = start_is; is <= end_is; is++)
        for (int js = start_js; js <= end_js; js++)
        {
            float sx_val = sx[is * ws + js];
            float sy_val = sy[is * ws + js];
            uchar2 i1_vec1, i1_vec2;

            j_m = min(max(j + sx_val, 0.0f), w - 1.0f - EPS);
            i_m = min(max(i + sy_val, 0.0f), h - 1.0f - EPS);
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
            coef = 1 / max(1.0f, fabs(diff));
            sum_Ux += coef * sx_val;
            sum_Uy += coef * sy_val;
            sum_coef += coef;
        }

    ux[i * w + j] = sum_Ux / sum_coef;
    uy[i * w + j] = sum_Uy / sum_coef;
}

#define INIT_BILINEAR_WEIGHTS(Ux, Uy)                                                                                  \
    i_I1 = min(max(i + Uy + bsz, i_lower_limit), i_upper_limit);                                                       \
    j_I1 = min(max(j + Ux + bsz, j_lower_limit), j_upper_limit);                                                       \
                                                                                                                       \
    w11 = (i_I1 - floor(i_I1)) * (j_I1 - floor(j_I1));                                                                 \
    w10 = (i_I1 - floor(i_I1)) * (floor(j_I1) + 1 - j_I1);                                                             \
    w01 = (floor(i_I1) + 1 - i_I1) * (j_I1 - floor(j_I1));                                                             \
    w00 = (floor(i_I1) + 1 - i_I1) * (floor(j_I1) + 1 - j_I1);

float computeSSDMeanNorm(const __global uchar *I0_ptr, const __global uchar *I1_ptr,
                         int I0_stride, int I1_stride,
                         float w00, float w01, float w10, float w11, int patch_sz, int i)
{
    float sum_diff = 0.0f, sum_diff_sq = 0.0f;
    int n = patch_sz * patch_sz;

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

    sum_diff = sub_group_reduce_add(sum_diff);
    sum_diff_sq = sub_group_reduce_add(sum_diff_sq);

    return sum_diff_sq - sum_diff * sum_diff / n;
}

__kernel void dis_patch_inverse_search_fwd_1(__global const float *Ux_ptr, __global const float *Uy_ptr,
                                             __global const uchar *I0_ptr, __global const uchar *I1_ptr,
                                             int border_size, int patch_size, int patch_stride,
                                             int w, int h, int ws, int hs, int pyr_level,
                                             __global float *Sx_ptr, __global float *Sy_ptr)
{
    int id = get_global_id(0);
    int is = id / 8;
    if (id >= (hs * 8)) return;

    int i = is * patch_stride;
    int j = 0;
    int psz = patch_size;
    int psz2 = psz / 2;
    int w_ext = w + 2 * border_size;
    int bsz = border_size;

    float i_lower_limit = bsz - psz + 1.0f;
    float i_upper_limit = bsz + h - 1.0f;
    float j_lower_limit = bsz - psz + 1.0f;
    float j_upper_limit = bsz + w - 1.0f;
    float i_I1, j_I1, w00, w01, w10, w11;

    float prev_Ux = Ux_ptr[(i + psz2) * w + j + psz2];
    float prev_Uy = Uy_ptr[(i + psz2) * w + j + psz2];
    Sx_ptr[is * ws] = prev_Ux;
    Sy_ptr[is * ws] = prev_Uy;
    j += patch_stride;

    int sid = get_sub_group_local_id();
    for (int js = 1; js < ws; js++, j += patch_stride)
    {
        float min_SSD, cur_SSD;
        float Ux = Ux_ptr[(i + psz2) * w + j + psz2];
        float Uy = Uy_ptr[(i + psz2) * w + j + psz2];

        INIT_BILINEAR_WEIGHTS(Ux, Uy);
        min_SSD = computeSSDMeanNorm(I0_ptr + i * w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                                     w, w_ext, w00, w01, w10, w11, psz, sid);

        INIT_BILINEAR_WEIGHTS(prev_Ux, prev_Uy);
        cur_SSD = computeSSDMeanNorm(I0_ptr + i * w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                                     w, w_ext, w00, w01, w10, w11, psz, sid);
        if (cur_SSD < min_SSD)
        {
            Ux = prev_Ux;
            Uy = prev_Uy;
        }

        prev_Ux = Ux;
        prev_Uy = Uy;
        Sx_ptr[is * ws + js] = Ux;
        Sy_ptr[is * ws + js] = Uy;
    }
}

float3 processPatchMeanNorm(const __global uchar *I0_ptr, const __global uchar *I1_ptr,
                            const __global short *I0x_ptr, const __global short *I0y_ptr,
                            int I0_stride, int I1_stride, float w00, float w01, float w10,
                            float w11, int patch_sz, float x_grad_sum, float y_grad_sum)
{
    float sum_diff = 0.0, sum_diff_sq = 0.0;
    float sum_I0x_mul = 0.0, sum_I0y_mul = 0.0;
    int n = patch_sz * patch_sz;
    uchar8 I1_vec1, I1_vec2;
    uchar I1_val1, I1_val2;

    for (int i = 0; i < 8; i++)
    {
        uchar8 I0_vec = vload8(0, I0_ptr + i * I0_stride);

        I1_vec1 = (i == 0) ? vload8(0, I1_ptr + i * I1_stride) : I1_vec2;
        I1_vec2 = vload8(0, I1_ptr + (i + 1) * I1_stride);
        I1_val1 = (i == 0) ? I1_ptr[i * I1_stride + patch_sz] : I1_val2;
        I1_val2 = I1_ptr[(i + 1) * I1_stride + patch_sz];

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

    float dst_dUx = sum_I0x_mul - sum_diff * x_grad_sum / n;
    float dst_dUy = sum_I0y_mul - sum_diff * y_grad_sum / n;
    float SSD = sum_diff_sq - sum_diff * sum_diff / n;

    return (float3)(SSD, dst_dUx, dst_dUy);
}

__kernel void dis_patch_inverse_search_fwd_2(__global const float *Ux_ptr, __global const float *Uy_ptr,
                                             __global const uchar *I0_ptr, __global const uchar *I1_ptr,
                                             __global const short *I0x_ptr, __global const short *I0y_ptr,
                                             __global const float *xx_ptr, __global const float *yy_ptr,
                                             __global const float *xy_ptr,
                                             __global const float *x_ptr, __global const float *y_ptr,
                                             int border_size, int patch_size, int patch_stride,
                                             int w, int h, int ws, int hs, int num_inner_iter, int pyr_level,
                                             __global float *Sx_ptr, __global float *Sy_ptr)
{
    int js = get_global_id(0);
    int is = get_global_id(1);
    int i = is * patch_stride;
    int j = js * patch_stride;
    int psz = patch_size;
    int psz2 = psz / 2;
    int w_ext = w + 2 * border_size;
    int bsz = border_size;
    int index = is * ws + js;

    if (js >= ws || is >= hs) return;

    float Ux = Sx_ptr[index];
    float Uy = Sy_ptr[index];
    float cur_Ux = Ux;
    float cur_Uy = Uy;
    float cur_xx = xx_ptr[index];
    float cur_yy = yy_ptr[index];
    float cur_xy = xy_ptr[index];
    float detH = cur_xx * cur_yy - cur_xy * cur_xy;

    if (fabs(detH) < EPS) detH = EPS;

    float invH11 = cur_yy / detH;
    float invH12 = -cur_xy / detH;
    float invH22 = cur_xx / detH;
    float prev_SSD = INF, SSD;
    float x_grad_sum = x_ptr[index];
    float y_grad_sum = y_ptr[index];

    float i_lower_limit = bsz - psz + 1.0f;
    float i_upper_limit = bsz + h - 1.0f;
    float j_lower_limit = bsz - psz + 1.0f;
    float j_upper_limit = bsz + w - 1.0f;
    float dUx, dUy, i_I1, j_I1, w00, w01, w10, w11, dx, dy;
    float3 res;

    for (int t = 0; t < num_inner_iter; t++)
    {
        INIT_BILINEAR_WEIGHTS(cur_Ux, cur_Uy);
        res = processPatchMeanNorm(I0_ptr + i * w + j,
                                   I1_ptr + (int)i_I1 * w_ext + (int)j_I1, I0x_ptr + i * w + j,
                                   I0y_ptr + i * w + j, w, w_ext, w00, w01, w10, w11, psz,
                                   x_grad_sum, y_grad_sum);

        SSD = res.x;
        dUx = res.y;
        dUy = res.z;
        dx = invH11 * dUx + invH12 * dUy;
        dy = invH12 * dUx + invH22 * dUy;

        cur_Ux -= dx;
        cur_Uy -= dy;

        if (SSD >= prev_SSD)
            break;
        prev_SSD = SSD;
    }

    float2 vec = (float2)(cur_Ux - Ux, cur_Uy - Uy);
    if (dot(vec, vec) <= (float)(psz * psz))
    {
        Sx_ptr[index] = cur_Ux;
        Sy_ptr[index] = cur_Uy;
    }
}

__kernel void dis_patch_inverse_search_bwd_1(__global const uchar *I0_ptr, __global const uchar *I1_ptr,
                                             int border_size, int patch_size, int patch_stride,
                                             int w, int h, int ws, int hs, int pyr_level,
                                             __global float *Sx_ptr, __global float *Sy_ptr)
{
    int id = get_global_id(0);
    int is = id / 8;
    if (id >= (hs * 8)) return;

    is = (hs - 1 - is);
    int i = is * patch_stride;
    int j = (ws - 2) * patch_stride;
    int psz = patch_size;
    int psz2 = psz / 2;
    int w_ext = w + 2 * border_size;
    int bsz = border_size;

    float i_lower_limit = bsz - psz + 1.0f;
    float i_upper_limit = bsz + h - 1.0f;
    float j_lower_limit = bsz - psz + 1.0f;
    float j_upper_limit = bsz + w - 1.0f;
    float i_I1, j_I1, w00, w01, w10, w11;

    int sid = get_sub_group_local_id();
    for (int js = (ws - 2); js > -1; js--, j -= patch_stride)
    {
        float min_SSD, cur_SSD;
        float2 Ux = vload2(0, Sx_ptr + is * ws + js);
        float2 Uy = vload2(0, Sy_ptr + is * ws + js);

        INIT_BILINEAR_WEIGHTS(Ux.x, Uy.x);
        min_SSD = computeSSDMeanNorm(I0_ptr + i * w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                                     w, w_ext, w00, w01, w10, w11, psz, sid);

        INIT_BILINEAR_WEIGHTS(Ux.y, Uy.y);
        cur_SSD = computeSSDMeanNorm(I0_ptr + i * w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                                     w, w_ext, w00, w01, w10, w11, psz, sid);
        if (cur_SSD < min_SSD)
        {
            Sx_ptr[is * ws + js] = Ux.y;
            Sy_ptr[is * ws + js] = Uy.y;
        }
    }
}

__kernel void dis_patch_inverse_search_bwd_2(__global const uchar *I0_ptr, __global const uchar *I1_ptr,
                                             __global const short *I0x_ptr, __global const short *I0y_ptr,
                                             __global const float *xx_ptr, __global const float *yy_ptr,
                                             __global const float *xy_ptr,
                                             __global const float *x_ptr, __global const float *y_ptr,
                                             int border_size, int patch_size, int patch_stride,
                                             int w, int h, int ws, int hs, int num_inner_iter,
                                             __global float *Sx_ptr, __global float *Sy_ptr)
{
    int js = get_global_id(0);
    int is = get_global_id(1);
    if (js >= ws || is >= hs) return;

    js = (ws - 1 - js);
    is = (hs - 1 - is);

    int j = js * patch_stride;
    int i = is * patch_stride;
    int psz = patch_size;
    int psz2 = psz / 2;
    int w_ext = w + 2 * border_size;
    int bsz = border_size;
    int index = is * ws + js;

    float Ux = Sx_ptr[index];
    float Uy = Sy_ptr[index];
    float cur_Ux = Ux;
    float cur_Uy = Uy;
    float cur_xx = xx_ptr[index];
    float cur_yy = yy_ptr[index];
    float cur_xy = xy_ptr[index];
    float detH = cur_xx * cur_yy - cur_xy * cur_xy;

    if (fabs(detH) < EPS) detH = EPS;

    float invH11 = cur_yy / detH;
    float invH12 = -cur_xy / detH;
    float invH22 = cur_xx / detH;
    float prev_SSD = INF, SSD;
    float x_grad_sum = x_ptr[index];
    float y_grad_sum = y_ptr[index];

    float i_lower_limit = bsz - psz + 1.0f;
    float i_upper_limit = bsz + h - 1.0f;
    float j_lower_limit = bsz - psz + 1.0f;
    float j_upper_limit = bsz + w - 1.0f;
    float dUx, dUy, i_I1, j_I1, w00, w01, w10, w11, dx, dy;
    float3 res;

    for (int t = 0; t < num_inner_iter; t++)
    {
        INIT_BILINEAR_WEIGHTS(cur_Ux, cur_Uy);
        res = processPatchMeanNorm(I0_ptr + i * w + j,
                                   I1_ptr + (int)i_I1 * w_ext + (int)j_I1, I0x_ptr + i * w + j,
                                   I0y_ptr + i * w + j, w, w_ext, w00, w01, w10, w11, psz,
                                   x_grad_sum, y_grad_sum);

        SSD = res.x;
        dUx = res.y;
        dUy = res.z;
        dx = invH11 * dUx + invH12 * dUy;
        dy = invH12 * dUx + invH22 * dUy;

        cur_Ux -= dx;
        cur_Uy -= dy;

        if (SSD >= prev_SSD)
            break;
        prev_SSD = SSD;
    }

    float2 vec = (float2)(cur_Ux - Ux, cur_Uy - Uy);
    if ((dot(vec, vec)) <= (float)(psz * psz))
    {
        Sx_ptr[index] = cur_Ux;
        Sy_ptr[index] = cur_Uy;
    }
}
