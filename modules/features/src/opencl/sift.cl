// OpenCL port of the SIFT detector and descriptor extractor
// See sift.simd.hpp for the CPU reference.
// Patent US6711293 expired in March 2020.

#define SIFT_DESCR_WIDTH 4
#define SIFT_DESCR_HIST_BINS 8
#define SIFT_DESCR_SCL_FCTR 3.0f
#define SIFT_DESCR_MAG_THR 0.2f
#define SIFT_INT_DESCR_FCTR 512.0f

#define SIFT_D SIFT_DESCR_WIDTH
#define SIFT_N SIFT_DESCR_HIST_BINS
#define SIFT_DH (SIFT_D + 2)
#define SIFT_NH (SIFT_N + 2)
#define SIFT_HIST_LEN (SIFT_DH * SIFT_DH * SIFT_NH)
#define SIFT_DESCR_LEN (SIFT_D * SIFT_D * SIFT_N)

#define SIFT_PI 3.14159265358979323846f
#define SIFT_RAD2DEG (180.0f / SIFT_PI)
#define SIFT_FLT_EPS 1.1920928955078125e-7f

#define SIFT_IMG_BORDER 5
#define SIFT_MAX_INTERP_STEPS 5
#define SIFT_ORI_HIST_BINS 36
#define SIFT_ORI_SIG_FCTR 1.5f
#define SIFT_ORI_RADIUS 4.5f
#define SIFT_ORI_PEAK_RATIO 0.8f

#define SIFT_BLUR_WG_X 16
#define SIFT_BLUR_WG_Y 16
#define SIFT_BLUR_MAX_RADIUS 32

#define SIFT_DESC_KP_PER_WG 4
#define SIFT_DESC_THREADS_PER_KP 16
#define SIFT_DESC_WG_SIZE (SIFT_DESC_KP_PER_WG * SIFT_DESC_THREADS_PER_KP)
#define SIFT_DESC_BINS (SIFT_N + 2)

#define READ_F32(buf, step, ofs, x, y) \
    (*(__global const float*)((buf) + (ofs) + (y) * (step) + (x) * 4))

#define READ_DOG(gp, gs, go, x, y, lc, rows) \
    (READ_F32(gp, gs, go, x, (y) + (lc+1)*(rows)) - \
     READ_F32(gp, gs, go, x, (y) + (lc)*(rows)))

__kernel void
SIFT_gaussian_blur_h(
    __global const uchar* src, int src_step, int src_ofs,
    __global uchar* dst, int dst_step, int dst_ofs,
    int cols, int rows,
    __global const float* coeffs, int radius)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int gx = get_group_id(0) * SIFT_BLUR_WG_X;
    int gy = get_group_id(1) * SIFT_BLUR_WG_Y;

    __local float lbuf[SIFT_BLUR_WG_Y][SIFT_BLUR_WG_X + 2 * SIFT_BLUR_MAX_RADIUS];

    int tile_w = SIFT_BLUR_WG_X + 2 * radius;
    int src_y = gy + ly;
    if (src_y < rows)
    {
        for (int col = lx; col < tile_w; col += SIFT_BLUR_WG_X)
        {
            int xx = gx - radius + col;
            if (xx < 0) xx = -xx;
            if (xx >= cols) xx = 2 * cols - xx - 2;
            lbuf[ly][col] = READ_F32(src, src_step, src_ofs, xx, src_y);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (x >= cols || y >= rows)
        return;

    float sum = 0.0f;
    for (int k = -radius; k <= radius; k++)
    {
        sum += lbuf[ly][lx + radius + k] * coeffs[k + radius];
    }
    *((__global float*)(dst + dst_ofs + y * dst_step + x * 4)) = sum;
}

__kernel void
SIFT_gaussian_blur_v(
    __global const uchar* src, int src_step, int src_ofs,
    __global uchar* dst, int dst_step, int dst_ofs,
    int cols, int rows,
    __global const float* coeffs, int radius)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int gx = get_group_id(0) * SIFT_BLUR_WG_X;
    int gy = get_group_id(1) * SIFT_BLUR_WG_Y;

    __local float lbuf[SIFT_BLUR_WG_Y + 2 * SIFT_BLUR_MAX_RADIUS][SIFT_BLUR_WG_X];

    int tile_h = SIFT_BLUR_WG_Y + 2 * radius;
    for (int row = ly; row < tile_h; row += SIFT_BLUR_WG_Y)
    {
        int xx = gx + lx;
        int yy = gy - radius + row;
        if (xx >= cols) xx = cols - 1;
        if (yy < 0) yy = -yy;
        if (yy >= rows) yy = 2 * rows - yy - 2;
        lbuf[row][lx] = READ_F32(src, src_step, src_ofs, xx, yy);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (x >= cols || y >= rows)
        return;

    float sum = 0.0f;
    for (int k = -radius; k <= radius; k++)
    {
        sum += lbuf[ly + radius + k][lx] * coeffs[k + radius];
    }
    *((__global float*)(dst + dst_ofs + y * dst_step + x * 4)) = sum;
}

#define SIFT_SCATTER(cr, cc, w0, w1) \
    if ((cr) >= 0 && (cr) < SIFT_D && (cc) >= 0 && (cc) < SIFT_D) { \
        int dri = (cr) - ti, dci = (cc) - tj; \
        if (dri == 0 && dci == 0) { \
            my_bins[o0]     += (w0); \
            my_bins[o0 + 1] += (w1); \
        } else if (dri == 1 && dci == 0) { \
            border_down[kid_in_wg][tid][o0]     += (w0); \
            border_down[kid_in_wg][tid][o0 + 1] += (w1); \
        } else if (dri == 0 && dci == 1) { \
            border_right[kid_in_wg][tid][o0]     += (w0); \
            border_right[kid_in_wg][tid][o0 + 1] += (w1); \
        } else { \
            border_down_right[kid_in_wg][tid][o0]     += (w0); \
            border_down_right[kid_in_wg][tid][o0 + 1] += (w1); \
        } \
    }

__kernel void
SIFT_compute_descriptor(
    __global const uchar* img, int img_step, int img_offset, int img_cols, int img_rows,
    float diag,
    __global const uchar* kpts, int kpts_step, int kpts_offset,
    __global const int* out_rows, int row_start,
    int nkeypoints,
    __global uchar* desc, int desc_step, int desc_offset,
    int descriptor_type)
{
    int lid = get_local_id(0);
    int kid_in_wg = lid >> 4;
    int tid = lid & 15;
    int ti = tid >> 2;
    int tj = tid & 3;

    int kid = get_group_id(0) * SIFT_DESC_KP_PER_WG + kid_in_wg;
    bool valid = (kid < nkeypoints);

    __local float border_down[SIFT_DESC_KP_PER_WG][16][SIFT_DESC_BINS];
    __local float border_right[SIFT_DESC_KP_PER_WG][16][SIFT_DESC_BINS];
    __local float border_down_right[SIFT_DESC_KP_PER_WG][16][SIFT_DESC_BINS];
    __local float nrm2_buf[SIFT_DESC_KP_PER_WG][16];

    for (int k = 0; k < SIFT_DESC_BINS; k++)
    {
        border_down[kid_in_wg][tid][k] = 0.0f;
        border_right[kid_in_wg][tid][k] = 0.0f;
        border_down_right[kid_in_wg][tid][k] = 0.0f;
    }

    float my_bins[SIFT_DESC_BINS];
    for (int k = 0; k < SIFT_DESC_BINS; k++)
        my_bins[k] = 0.0f;

    float cos_t = 0.0f, sin_t = 0.0f;
    float bins_per_rad = 0.0f, exp_scale = 0.0f;
    int radius = 0, pt_x = 0, pt_y = 0;
    float ori = 0.0f;
    __global const uchar* img_base = 0;
    int row = 0;

    if (valid)
    {
        __global const float* kpt = (__global const float*)(kpts + kpts_offset + kid * kpts_step);
        float ptx = kpt[0];
        float pty = kpt[1];
        ori = kpt[2];
        float scl = kpt[3];
        int level_idx = (int)kpt[4];

        pt_x = convert_int_rte(ptx);
        pt_y = convert_int_rte(pty);

        cos_t = cos(ori * SIFT_PI / 180.0f);
        sin_t = sin(ori * SIFT_PI / 180.0f);
        bins_per_rad = (float)SIFT_N / 360.0f;
        exp_scale = -1.0f / ((float)SIFT_D * (float)SIFT_D * 0.5f);
        float hist_width = SIFT_DESCR_SCL_FCTR * scl;
        radius = convert_int_rte(hist_width * 1.4142135623730951f * (float)(SIFT_D + 1) * 0.5f);
        if ((float)radius > diag)
            radius = convert_int_rte(diag);
        float inv_hist_width = native_recip(hist_width);
        cos_t *= inv_hist_width;
        sin_t *= inv_hist_width;

        img_base = img + img_offset + (long)level_idx * img_rows * img_step;
        row = out_rows[kid + row_start];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (valid)
    {
        float r_min = (ti == 0) ? -1.0f : (float)ti;
        float r_max = (ti == 0) ? 1.0f : (float)(ti + 1);
        float c_min = (tj == 0) ? -1.0f : (float)tj;
        float c_max = (tj == 0) ? 1.0f : (float)(tj + 1);

        float rr0 = r_min - 1.5f, rr1 = r_max - 1.5f;
        float cc0 = c_min - 1.5f, cc1 = c_max - 1.5f;

        float det = cos_t * cos_t + sin_t * sin_t;
        float inv_det = native_recip(det);

        float i1 = (-cc0 * sin_t + rr0 * cos_t) * inv_det;
        float i2 = (-cc0 * sin_t + rr1 * cos_t) * inv_det;
        float i3 = (-cc1 * sin_t + rr0 * cos_t) * inv_det;
        float i4 = (-cc1 * sin_t + rr1 * cos_t) * inv_det;
        float j1 = (cc0 * cos_t + rr0 * sin_t) * inv_det;
        float j2 = (cc0 * cos_t + rr1 * sin_t) * inv_det;
        float j3 = (cc1 * cos_t + rr0 * sin_t) * inv_det;
        float j4 = (cc1 * cos_t + rr1 * sin_t) * inv_det;

        int i_lo = max(-radius, (int)floor(fmin(fmin(i1, i2), fmin(i3, i4))) - 1);
        int i_hi = min(radius, (int)ceil(fmax(fmax(i1, i2), fmax(i3, i4))) + 1);
        int j_lo = max(-radius, (int)floor(fmin(fmin(j1, j2), fmin(j3, j4))) - 1);
        int j_hi = min(radius, (int)ceil(fmax(fmax(j1, j2), fmax(j3, j4))) + 1);

        for (int i = i_lo; i <= i_hi; i++)
        {
            for (int j = j_lo; j <= j_hi; j++)
            {
                float c_rot = (float)j * cos_t - (float)i * sin_t;
                float r_rot = (float)j * sin_t + (float)i * cos_t;
                float rbin = r_rot + (float)(SIFT_D / 2) - 0.5f;
                float cbin = c_rot + (float)(SIFT_D / 2) - 0.5f;
                int r = pt_y + i;
                int c = pt_x + j;

                if (rbin > -1.0f && rbin < (float)SIFT_D &&
                    cbin > -1.0f && cbin < (float)SIFT_D &&
                    r > 0 && r < img_rows - 1 && c > 0 && c < img_cols - 1)
                {
                    int r0 = (int)floor(rbin);
                    int c0 = (int)floor(cbin);

                    int my_r0 = (r0 > 0) ? r0 : 0;
                    int my_c0 = (c0 > 0) ? c0 : 0;
                    if (my_r0 != ti || my_c0 != tj)
                        continue;

                    __global const uchar* row_base = img_base + r * img_step;
                    int c4 = c * 4;
                    float dx = *(__global const float*)(row_base + c4 + 4) -
                               *(__global const float*)(row_base + c4 - 4);
                    float dy = *(__global const float*)(row_base - img_step + c4) -
                               *(__global const float*)(row_base + img_step + c4);
                    float w = (c_rot * c_rot + r_rot * r_rot) * exp_scale;
                    float ang = atan2(dy, dx) * SIFT_RAD2DEG;
                    if (ang < 0.0f)
                        ang += 360.0f;
                    float mag = native_sqrt(dx * dx + dy * dy);
                    float W = native_exp(w);

                    float obin = (ang - ori) * bins_per_rad;
                    float magW = mag * W;

                    rbin -= (float)r0;
                    cbin -= (float)c0;
                    int o0 = (int)floor(obin);
                    obin -= (float)o0;

                    if (o0 < 0)
                        o0 += SIFT_N;
                    if (o0 >= SIFT_N)
                        o0 -= SIFT_N;

                    float v_r1 = magW * rbin, v_r0 = magW - v_r1;
                    float v_rc11 = v_r1 * cbin, v_rc10 = v_r1 - v_rc11;
                    float v_rc01 = v_r0 * cbin, v_rc00 = v_r0 - v_rc01;
                    float v_rco111 = v_rc11 * obin, v_rco110 = v_rc11 - v_rco111;
                    float v_rco101 = v_rc10 * obin, v_rco100 = v_rc10 - v_rco101;
                    float v_rco011 = v_rc01 * obin, v_rco010 = v_rc01 - v_rco011;
                    float v_rco001 = v_rc00 * obin, v_rco000 = v_rc00 - v_rco001;

                    SIFT_SCATTER(r0,     c0,     v_rco000, v_rco001);
                    SIFT_SCATTER(r0,     c0 + 1, v_rco010, v_rco011);
                    SIFT_SCATTER(r0 + 1, c0,     v_rco100, v_rco101);
                    SIFT_SCATTER(r0 + 1, c0 + 1, v_rco110, v_rco111);
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (valid)
    {
        if (ti > 0)
        {
            int above_tid = ((ti - 1) << 2) | tj;
            for (int k = 0; k < SIFT_DESC_BINS; k++)
                my_bins[k] += border_down[kid_in_wg][above_tid][k];
        }
        if (tj > 0)
        {
            int left_tid = (ti << 2) | (tj - 1);
            for (int k = 0; k < SIFT_DESC_BINS; k++)
                my_bins[k] += border_right[kid_in_wg][left_tid][k];
        }
        if (ti > 0 && tj > 0)
        {
            int ul_tid = ((ti - 1) << 2) | (tj - 1);
            for (int k = 0; k < SIFT_DESC_BINS; k++)
                my_bins[k] += border_down_right[kid_in_wg][ul_tid][k];
        }

        my_bins[0] += my_bins[SIFT_N];
        my_bins[1] += my_bins[SIFT_N + 1];

        float my_nrm2 = 0.0f;
        for (int k = 0; k < SIFT_N; k++)
            my_nrm2 += my_bins[k] * my_bins[k];
        nrm2_buf[kid_in_wg][tid] = my_nrm2;
    }
    else
    {
        nrm2_buf[kid_in_wg][tid] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (valid)
    {
        float total_nrm2 = 0.0f;
        for (int k = 0; k < 16; k++)
            total_nrm2 += nrm2_buf[kid_in_wg][k];

        float thr = native_sqrt(total_nrm2) * SIFT_DESCR_MAG_THR;

        float my_nrm2 = 0.0f;
        for (int k = 0; k < SIFT_N; k++)
        {
            float val = my_bins[k];
            if (val > thr)
                val = thr;
            my_bins[k] = val;
            my_nrm2 += val * val;
        }
        nrm2_buf[kid_in_wg][tid] = my_nrm2;
    }
    else
    {
        nrm2_buf[kid_in_wg][tid] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (valid)
    {
        float total_nrm2 = 0.0f;
        for (int k = 0; k < 16; k++)
            total_nrm2 += nrm2_buf[kid_in_wg][k];

        float inv = SIFT_INT_DESCR_FCTR / fmax(native_sqrt(total_nrm2), SIFT_FLT_EPS);

        int didx = (ti * SIFT_D + tj) * SIFT_N;
        if (descriptor_type == 0)
        {
            __global uchar* dst = desc + desc_offset + row * desc_step;
            for (int k = 0; k < SIFT_N; k++)
                dst[didx + k] = convert_uchar_sat_rte(my_bins[k] * inv);
        }
        else
        {
            __global float* dst = (__global float*)(desc + desc_offset + row * desc_step);
            for (int k = 0; k < SIFT_N; k++)
            {
                float v = round(my_bins[k] * inv);
                dst[didx + k] = clamp(v, 0.0f, 255.0f);
            }
        }
    }
}

#undef SIFT_SCATTER

__kernel void
SIFT_detect_and_orient(
    __global const uchar* gauss_pack, int gauss_step, int gauss_offset,
    int dog_cols, int dog_rows_per_layer,
    int threshold,
    float contrastThreshold, float edgeThreshold, float sigma,
    int nOctaveLayers, int octave, int layer,
    __global int* output_count,
    __global float* output_kpts,
    int max_output)
{
    int c = get_global_id(0);
    int r = get_global_id(1);

    if (c < SIFT_IMG_BORDER || c >= dog_cols - SIFT_IMG_BORDER ||
        r < SIFT_IMG_BORDER || r >= dog_rows_per_layer - SIFT_IMG_BORDER)
        return;

    int lc = layer;

    float val = READ_DOG(gauss_pack, gauss_step, gauss_offset, c, r, lc, dog_rows_per_layer);
    if (fabs(val) <= (float)threshold)
        return;

    float cv00 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c-1, r-1, lc, dog_rows_per_layer);
    float cv01 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c,   r-1, lc, dog_rows_per_layer);
    float cv02 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c+1, r-1, lc, dog_rows_per_layer);
    float cv10 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c-1, r,   lc, dog_rows_per_layer);
    float cv12 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c+1, r,   lc, dog_rows_per_layer);
    float cv20 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c-1, r+1, lc, dog_rows_per_layer);
    float cv21 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c,   r+1, lc, dog_rows_per_layer);
    float cv22 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c+1, r+1, lc, dog_rows_per_layer);

    float pv00 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c-1, r-1, lc-1, dog_rows_per_layer);
    float pv01 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c,   r-1, lc-1, dog_rows_per_layer);
    float pv02 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c+1, r-1, lc-1, dog_rows_per_layer);
    float pv10 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c-1, r,   lc-1, dog_rows_per_layer);
    float pv11 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c,   r,   lc-1, dog_rows_per_layer);
    float pv12 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c+1, r,   lc-1, dog_rows_per_layer);
    float pv20 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c-1, r+1, lc-1, dog_rows_per_layer);
    float pv21 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c,   r+1, lc-1, dog_rows_per_layer);
    float pv22 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c+1, r+1, lc-1, dog_rows_per_layer);

    float nv00 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c-1, r-1, lc+1, dog_rows_per_layer);
    float nv01 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c,   r-1, lc+1, dog_rows_per_layer);
    float nv02 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c+1, r-1, lc+1, dog_rows_per_layer);
    float nv10 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c-1, r,   lc+1, dog_rows_per_layer);
    float nv11 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c,   r,   lc+1, dog_rows_per_layer);
    float nv12 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c+1, r,   lc+1, dog_rows_per_layer);
    float nv20 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c-1, r+1, lc+1, dog_rows_per_layer);
    float nv21 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c,   r+1, lc+1, dog_rows_per_layer);
    float nv22 = READ_DOG(gauss_pack, gauss_step, gauss_offset, c+1, r+1, lc+1, dog_rows_per_layer);

    bool is_extremum = false;
    if (val > 0.0f)
    {
        float vmax = fmax(fmax(fmax(cv00, cv01), fmax(cv02, cv10)), fmax(fmax(cv12, cv20), fmax(cv21, cv22)));
        vmax = fmax(vmax, fmax(fmax(fmax(pv00, pv01), fmax(pv02, pv10)), fmax(fmax(pv12, pv20), fmax(pv21, pv22))));
        vmax = fmax(vmax, fmax(fmax(fmax(nv00, nv01), fmax(nv02, nv10)), fmax(fmax(nv12, nv20), fmax(nv21, nv22))));
        vmax = fmax(vmax, fmax(pv11, nv11));
        is_extremum = (val >= vmax);
    }
    else
    {
        float vmin = fmin(fmin(fmin(cv00, cv01), fmin(cv02, cv10)), fmin(fmin(cv12, cv20), fmin(cv21, cv22)));
        vmin = fmin(vmin, fmin(fmin(fmin(pv00, pv01), fmin(pv02, pv10)), fmin(fmin(pv12, pv20), fmin(pv21, pv22))));
        vmin = fmin(vmin, fmin(fmin(fmin(nv00, nv01), fmin(nv02, nv10)), fmin(fmin(nv12, nv20), fmin(nv21, nv22))));
        vmin = fmin(vmin, fmin(pv11, nv11));
        is_extremum = (val <= vmin);
    }

    if (!is_extremum)
        return;

    const float img_scale = 1.0f / 255.0f;
    const float deriv_scale = img_scale * 0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale * 0.25f;

    int rc = r, cc = c;
    float xc = 0.0f, xr = 0.0f, xi = 0.0f, contr = 0.0f;
    int iter = 0;
    bool converged = false;

    for (; iter < SIFT_MAX_INTERP_STEPS; iter++)
    {
        float cur_v = READ_DOG(gauss_pack, gauss_step, gauss_offset, cc, rc, lc, dog_rows_per_layer);
        float prv_c = READ_DOG(gauss_pack, gauss_step, gauss_offset, cc, rc, lc-1, dog_rows_per_layer);
        float nxt_c = READ_DOG(gauss_pack, gauss_step, gauss_offset, cc, rc, lc+1, dog_rows_per_layer);

        float dD0 = (READ_DOG(gauss_pack, gauss_step, gauss_offset, cc+1, rc, lc, dog_rows_per_layer) -
                     READ_DOG(gauss_pack, gauss_step, gauss_offset, cc-1, rc, lc, dog_rows_per_layer)) * deriv_scale;
        float dD1 = (READ_DOG(gauss_pack, gauss_step, gauss_offset, cc, rc+1, lc, dog_rows_per_layer) -
                     READ_DOG(gauss_pack, gauss_step, gauss_offset, cc, rc-1, lc, dog_rows_per_layer)) * deriv_scale;
        float dD2 = (nxt_c - prv_c) * deriv_scale;

        float v2 = cur_v * 2.0f;
        float dxx = (READ_DOG(gauss_pack, gauss_step, gauss_offset, cc+1, rc, lc, dog_rows_per_layer) +
                     READ_DOG(gauss_pack, gauss_step, gauss_offset, cc-1, rc, lc, dog_rows_per_layer) - v2) * second_deriv_scale;
        float dyy = (READ_DOG(gauss_pack, gauss_step, gauss_offset, cc, rc+1, lc, dog_rows_per_layer) +
                     READ_DOG(gauss_pack, gauss_step, gauss_offset, cc, rc-1, lc, dog_rows_per_layer) - v2) * second_deriv_scale;
        float dss = (nxt_c + prv_c - v2) * second_deriv_scale;
        float dxy = (READ_DOG(gauss_pack, gauss_step, gauss_offset, cc+1, rc+1, lc, dog_rows_per_layer) -
                     READ_DOG(gauss_pack, gauss_step, gauss_offset, cc-1, rc+1, lc, dog_rows_per_layer) -
                     READ_DOG(gauss_pack, gauss_step, gauss_offset, cc+1, rc-1, lc, dog_rows_per_layer) +
                     READ_DOG(gauss_pack, gauss_step, gauss_offset, cc-1, rc-1, lc, dog_rows_per_layer)) * cross_deriv_scale;
        float dxs = (READ_DOG(gauss_pack, gauss_step, gauss_offset, cc+1, rc, lc+1, dog_rows_per_layer) -
                     READ_DOG(gauss_pack, gauss_step, gauss_offset, cc-1, rc, lc+1, dog_rows_per_layer) -
                     READ_DOG(gauss_pack, gauss_step, gauss_offset, cc+1, rc, lc-1, dog_rows_per_layer) +
                     READ_DOG(gauss_pack, gauss_step, gauss_offset, cc-1, rc, lc-1, dog_rows_per_layer)) * cross_deriv_scale;
        float dys = (READ_DOG(gauss_pack, gauss_step, gauss_offset, cc, rc+1, lc+1, dog_rows_per_layer) -
                     READ_DOG(gauss_pack, gauss_step, gauss_offset, cc, rc-1, lc+1, dog_rows_per_layer) -
                     READ_DOG(gauss_pack, gauss_step, gauss_offset, cc, rc+1, lc-1, dog_rows_per_layer) +
                     READ_DOG(gauss_pack, gauss_step, gauss_offset, cc, rc-1, lc-1, dog_rows_per_layer)) * cross_deriv_scale;

        float detH = dxx*(dyy*dss - dys*dys) - dxy*(dxy*dss - dys*dxs) + dxs*(dxy*dys - dyy*dxs);
        if (fabs(detH) < SIFT_FLT_EPS)
        {
            xc = 0.0f; xr = 0.0f; xi = 0.0f;
            converged = true;
            contr = cur_v * img_scale;
            break;
        }

        float inv_det = 1.0f / detH;
        xc = -(dD0*(dyy*dss - dys*dys) - dxy*(dD1*dss - dys*dD2) + dxs*(dD1*dys - dyy*dD2)) * inv_det;
        xr = -(dxx*(dD1*dss - dys*dD2) - dD0*(dxy*dss - dys*dxs) + dxs*(dxy*dD2 - dD1*dxs)) * inv_det;
        xi = -(dxx*(dyy*dD2 - dD1*dys) - dxy*(dxy*dD2 - dD1*dxs) + dD0*(dxy*dys - dyy*dxs)) * inv_det;

        if (fabs(xi) < 0.5f && fabs(xr) < 0.5f && fabs(xc) < 0.5f)
        {
            converged = true;
            contr = cur_v * img_scale + (dD0*xc + dD1*xr + dD2*xi) * 0.5f;
            break;
        }

        if (fabs(xc) > (float)(INT_MAX/3) || fabs(xr) > (float)(INT_MAX/3) || fabs(xi) > (float)(INT_MAX/3))
            return;

        cc += convert_int_rte(xc);
        rc += convert_int_rte(xr);
        lc += convert_int_rte(xi);

        if (lc < 1 || lc > nOctaveLayers ||
            cc < SIFT_IMG_BORDER || cc >= dog_cols - SIFT_IMG_BORDER ||
            rc < SIFT_IMG_BORDER || rc >= dog_rows_per_layer - SIFT_IMG_BORDER)
            return;
    }

    if (!converged)
        return;

    if (fabs(contr) * (float)nOctaveLayers < contrastThreshold)
        return;

    float cur_v = READ_DOG(gauss_pack, gauss_step, gauss_offset, cc, rc, lc, dog_rows_per_layer);
    float v2 = cur_v * 2.0f;
    float dxx = (READ_DOG(gauss_pack, gauss_step, gauss_offset, cc+1, rc, lc, dog_rows_per_layer) +
                 READ_DOG(gauss_pack, gauss_step, gauss_offset, cc-1, rc, lc, dog_rows_per_layer) - v2) * second_deriv_scale;
    float dyy = (READ_DOG(gauss_pack, gauss_step, gauss_offset, cc, rc+1, lc, dog_rows_per_layer) +
                 READ_DOG(gauss_pack, gauss_step, gauss_offset, cc, rc-1, lc, dog_rows_per_layer) - v2) * second_deriv_scale;
    float dxy = (READ_DOG(gauss_pack, gauss_step, gauss_offset, cc+1, rc+1, lc, dog_rows_per_layer) -
                 READ_DOG(gauss_pack, gauss_step, gauss_offset, cc-1, rc+1, lc, dog_rows_per_layer) -
                 READ_DOG(gauss_pack, gauss_step, gauss_offset, cc+1, rc-1, lc, dog_rows_per_layer) +
                 READ_DOG(gauss_pack, gauss_step, gauss_offset, cc-1, rc-1, lc, dog_rows_per_layer)) * cross_deriv_scale;
    float tr = dxx + dyy;
    float det = dxx * dyy - dxy * dxy;
    if (det <= 0.0f || tr*tr*edgeThreshold >= (edgeThreshold + 1.0f)*(edgeThreshold + 1.0f)*det)
        return;

    float scl_octv = sigma * exp2(((float)lc + xi) / (float)nOctaveLayers);
    int radius = convert_int_rte(SIFT_ORI_RADIUS * scl_octv);
    float ori_sig = SIFT_ORI_SIG_FCTR * scl_octv;
    float expf_scale = -1.0f / (2.0f * ori_sig * ori_sig);

    float hist[SIFT_ORI_HIST_BINS];
    for (int b = 0; b < SIFT_ORI_HIST_BINS; b++)
        hist[b] = 0.0f;

    int gauss_row_base = lc * dog_rows_per_layer;

    for (int i = -radius; i <= radius; i++)
    {
        int gy = rc + i;
        if (gy <= 0 || gy >= dog_rows_per_layer - 1)
            continue;
        int grow = gauss_row_base + gy;
        for (int j = -radius; j <= radius; j++)
        {
            int gx = cc + j;
            if (gx <= 0 || gx >= dog_cols - 1)
                continue;
            float dx = READ_F32(gauss_pack, gauss_step, gauss_offset, gx+1, grow) -
                       READ_F32(gauss_pack, gauss_step, gauss_offset, gx-1, grow);
            float dy = READ_F32(gauss_pack, gauss_step, gauss_offset, gx, grow-1) -
                       READ_F32(gauss_pack, gauss_step, gauss_offset, gx, grow+1);
            float ang = atan2(dy, dx) * SIFT_RAD2DEG;
            if (ang < 0.0f) ang += 360.0f;
            float mag = native_sqrt(dx*dx + dy*dy);
            float w = native_exp((float)(i*i + j*j) * expf_scale);
            int bin = convert_int_rte((float)SIFT_ORI_HIST_BINS / 360.0f * ang);
            if (bin >= SIFT_ORI_HIST_BINS) bin -= SIFT_ORI_HIST_BINS;
            if (bin < 0) bin += SIFT_ORI_HIST_BINS;
            hist[bin] += w * mag;
        }
    }

    float sm[SIFT_ORI_HIST_BINS];
    for (int b = 0; b < SIFT_ORI_HIST_BINS; b++)
    {
        int bm2 = (b + SIFT_ORI_HIST_BINS - 2) % SIFT_ORI_HIST_BINS;
        int bm1 = (b + SIFT_ORI_HIST_BINS - 1) % SIFT_ORI_HIST_BINS;
        int bp1 = (b + 1) % SIFT_ORI_HIST_BINS;
        int bp2 = (b + 2) % SIFT_ORI_HIST_BINS;
        sm[b] = (hist[bm2] + hist[bp2]) * (1.0f/16.0f) +
                (hist[bm1] + hist[bp1]) * (4.0f/16.0f) +
                hist[b] * (6.0f/16.0f);
    }

    float maxval = sm[0];
    for (int b = 1; b < SIFT_ORI_HIST_BINS; b++)
        maxval = fmax(maxval, sm[b]);
    float mag_thr = maxval * SIFT_ORI_PEAK_RATIO;

    float kpt_x = (cc + xc) * (float)(1 << octave);
    float kpt_y = (rc + xr) * (float)(1 << octave);
    float kpt_size = scl_octv * (float)(1 << octave) * 2.0f;
    float kpt_resp = fabs(contr);
    int kpt_oct = octave + (lc << 8) + (convert_int_rte((xi + 0.5f) * 255.0f) << 16);

    for (int b = 0; b < SIFT_ORI_HIST_BINS; b++)
    {
        int bm1 = (b + SIFT_ORI_HIST_BINS - 1) % SIFT_ORI_HIST_BINS;
        int bp1 = (b + 1) % SIFT_ORI_HIST_BINS;
        if (sm[b] > sm[bm1] && sm[b] > sm[bp1] && sm[b] >= mag_thr)
        {
            float denom = sm[bm1] - 2.0f*sm[b] + sm[bp1];
            if (fabs(denom) < SIFT_FLT_EPS)
                continue;
            float bin = (float)b + 0.5f * (sm[bm1] - sm[bp1]) / denom;
            if (bin < 0.0f) bin += (float)SIFT_ORI_HIST_BINS;
            if (bin >= (float)SIFT_ORI_HIST_BINS) bin -= (float)SIFT_ORI_HIST_BINS;
            float angle = 360.0f - (360.0f / (float)SIFT_ORI_HIST_BINS) * bin;
            if (fabs(angle - 360.0f) < SIFT_FLT_EPS)
                angle = 0.0f;

            int out_idx = atomic_inc(output_count);
            if (out_idx >= max_output)
                return;
            int base = out_idx * 6;
            output_kpts[base + 0] = kpt_x;
            output_kpts[base + 1] = kpt_y;
            output_kpts[base + 2] = angle;
            output_kpts[base + 3] = kpt_size;
            output_kpts[base + 4] = kpt_resp;
            output_kpts[base + 5] = as_float(kpt_oct);
        }
    }
}
