// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

inline float gaussian(float x, float y, float sigma)
{
    return native_exp(-(x*x + y*y) / (2.0f*sigma*sigma));
}

// Pre-computed gaussian weights for descriptor 9×9 sample grid (81 values, row-major).
// gauss_s1(ri,ci) = exp(-((ri-5)^2 + (ci-5)^2) / 12.5)
__constant float gauss_s1_weights[81] = {
    0.01831564f, 0.03762826f, 0.06587475f, 0.09827359f, 0.12493021f, 0.13533528f, 0.12493021f, 0.09827359f, 0.06587475f,
    0.03762826f, 0.07730474f, 0.13533528f, 0.20189652f, 0.25666078f, 0.27803730f, 0.25666078f, 0.20189652f, 0.13533528f,
    0.06587475f, 0.13533528f, 0.23692776f, 0.35345468f, 0.44932896f, 0.48675226f, 0.44932896f, 0.35345468f, 0.23692776f,
    0.09827359f, 0.20189652f, 0.35345468f, 0.52729242f, 0.67032005f, 0.72614904f, 0.67032005f, 0.52729242f, 0.35345468f,
    0.12493021f, 0.25666078f, 0.44932896f, 0.67032005f, 0.85214379f, 0.92311635f, 0.85214379f, 0.67032005f, 0.44932896f,
    0.13533528f, 0.27803730f, 0.48675226f, 0.72614904f, 0.92311635f, 1.00000000f, 0.92311635f, 0.72614904f, 0.48675226f,
    0.12493021f, 0.25666078f, 0.44932896f, 0.67032005f, 0.85214379f, 0.92311635f, 0.85214379f, 0.67032005f, 0.44932896f,
    0.09827359f, 0.20189652f, 0.35345468f, 0.52729242f, 0.67032005f, 0.72614904f, 0.67032005f, 0.52729242f, 0.35345468f,
    0.06587475f, 0.13533528f, 0.23692776f, 0.35345468f, 0.44932896f, 0.48675226f, 0.44932896f, 0.35345468f, 0.23692776f
};

// Pre-computed gaussian weights for 4×4 subregion outer weighting (16 values, row-major).
// cx=sr+0.5, cy=sc+0.5 for sr,sc in {0,1,2,3}
// gauss_s2(sr,sc) = exp(-((sr-1.5)^2 + (sc-1.5)^2) / 4.5)
__constant float gauss_s2_weights[16] = {
    0.36787944f, 0.57375342f, 0.57375342f, 0.36787944f,
    0.57375342f, 0.89483932f, 0.89483932f, 0.57375342f,
    0.57375342f, 0.89483932f, 0.89483932f, 0.57375342f,
    0.36787944f, 0.57375342f, 0.57375342f, 0.36787944f
};

/**
 * @brief Compute KAZE upright 64-dimensional descriptor (optimized)
 * @details Matches CPU Get_KAZE_Upright_Descriptor_64 exactly:
 *          - integer scale via round(kpt_size/2)
 *          - bilinear with y1=(int)(y-0.5), y2=(int)(y+0.5) convention
 *          - clamp-on-border (not skip)
 *
 * Optimizations:
 * - Pre-computed gaussian weights (__constant, no exp() calls in inner loop)
 * - Vectorized vload2 for bilinear pixel pairs
 * - Pre-computed row addresses outside ci loop
 * - Simplified loop structure
 *
 * @param Lx     First-order x-derivative (sigma_size-scaled), row-major float array
 * @param Ly     First-order y-derivative (sigma_size-scaled), row-major float array
 * @param lx_step    Row stride of Lx/Ly in float elements (= cols for continuous mat)
 * @param lx_rows    Image height
 * @param lx_cols    Image width
 * @param keypoints_x  Keypoint x coordinates
 * @param keypoints_y  Keypoint y coordinates
 * @param keypoints_size  Keypoint sizes (diameter = 2*sigma)
 * @param descriptors  Output descriptor matrix, shape [nkeypoints, 64]
 * @param nkeypoints   Number of keypoints
 */
__kernel void
KAZE_compute_upright_descriptor_64(
    __global const float* Lx,
    __global const float* Ly,
    int lx_step, int lx_rows, int lx_cols,
    __global const float* keypoints_x,
    __global const float* keypoints_y,
    __global const float* keypoints_size,
    __global float* descriptors,
    int nkeypoints)
{
    int idx = get_global_id(0);
    if (idx >= nkeypoints)
        return;

    float kpt_x = keypoints_x[idx];
    float kpt_y = keypoints_y[idx];
    float kpt_sz = keypoints_size[idx];

    int scale = (int)(kpt_sz / 2.0f + 0.5f);
    float xf = kpt_x;
    float yf = kpt_y;

    const int dsize = 64;
    float len = 0.0f;
    int dcount = 0;

    for (int sr = 0; sr < 4; sr++)
    {
        int base_k = sr * 5 - 12;
        for (int sc = 0; sc < 4; sc++)
        {
            int base_l = sc * 5 - 12;

            float dx = 0.0f, dy = 0.0f, mdx = 0.0f, mdy = 0.0f;

            for (int ri = 0; ri < 9; ri++)
            {
                int k = base_k + ri;
                float sample_y = (float)k * scale + yf;
                int y1 = (int)(sample_y - 0.5f);
                int y2 = (int)(sample_y + 0.5f);
                y1 = clamp(y1, 0, lx_rows - 1);
                y2 = clamp(y2, 0, lx_rows - 1);

                int row0 = y1 * lx_step;
                int row1 = y2 * lx_step;
                float fy = sample_y - (float)y1;
                float inv_fy = 1.0f - fy;

                for (int ci = 0; ci < 9; ci++)
                {
                    int l = base_l + ci;
                    float sample_x = (float)l * scale + xf;

                    int x1_raw = (int)(sample_x - 0.5f);
                    int x2_raw = (int)(sample_x + 0.5f);
                    int x1 = clamp(x1_raw, 0, lx_cols - 1);
                    int x2 = clamp(x2_raw, 0, lx_cols - 1);

                    float fx = sample_x - (float)x1;
                    float inv_fx = 1.0f - fx;

                    float gs1 = gauss_s1_weights[ri * 9 + ci];

                    bool use_v2 = (x1_raw >= 0 && x2_raw < lx_cols);

                    float lx00, lx01, ly00, ly01;
                    float lx10, lx11, ly10, ly11;

                    if (use_v2)
                    {
                        float2 vx = vload2(0, Lx + row0 + x1);
                        float2 vy = vload2(0, Ly + row0 + x1);
                        lx00 = vx.s0; lx01 = vx.s1;
                        ly00 = vy.s0; ly01 = vy.s1;
                    }
                    else
                    {
                        lx00 = Lx[row0 + x1];
                        lx01 = Lx[row0 + x2];
                        ly00 = Ly[row0 + x1];
                        ly01 = Ly[row0 + x2];
                    }

                    if (y1 != y2)
                    {
                        if (use_v2)
                        {
                            float2 vx = vload2(0, Lx + row1 + x1);
                            float2 vy = vload2(0, Ly + row1 + x1);
                            lx10 = vx.s0; lx11 = vx.s1;
                            ly10 = vy.s0; ly11 = vy.s1;
                        }
                        else
                        {
                            lx10 = Lx[row1 + x1];
                            lx11 = Lx[row1 + x2];
                            ly10 = Ly[row1 + x1];
                            ly11 = Ly[row1 + x2];
                        }
                    }
                    else
                    {
                        lx10 = lx00; lx11 = lx01;
                        ly10 = ly00; ly11 = ly01;
                    }

                    float rx = inv_fx * inv_fy * lx00 + fx * inv_fy * lx01
                             + inv_fx * fy * lx10 + fx * fy * lx11;
                    float ry = inv_fx * inv_fy * ly00 + fx * inv_fy * ly01
                             + inv_fx * fy * ly10 + fx * fy * ly11;

                    rx = gs1 * rx;
                    ry = gs1 * ry;

                    dx  += rx;
                    dy  += ry;
                    mdx += fabs(rx);
                    mdy += fabs(ry);
                }
            }

            float gs2 = gauss_s2_weights[sr * 4 + sc];

            descriptors[idx * dsize + dcount++] = dx  * gs2;
            descriptors[idx * dsize + dcount++] = dy  * gs2;
            descriptors[idx * dsize + dcount++] = mdx * gs2;
            descriptors[idx * dsize + dcount++] = mdy * gs2;

            len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy) * gs2 * gs2;
        }
    }

    len = sqrt(len);
    if (len > 1e-10f)
    {
        float len_inv = 1.0f / len;
        for (int i = 0; i < dsize; i++)
            descriptors[idx * dsize + i] *= len_inv;
    }
}

/**
 * @brief Compute KAZE upright 128-dimensional descriptor (optimized)
 * @details Matches CPU Get_KAZE_Upright_Descriptor_128: splits dx/dy by sign of the
 *          cross-component for 8 values per subregion.
 *
 * Optimizations:
 * - Pre-computed gaussian weights (__constant, no exp() calls in inner loop)
 * - Vectorized vload2 for bilinear pixel pairs
 * - Pre-computed row addresses outside ci loop
 */
__kernel void
KAZE_compute_upright_descriptor_128(
    __global const float* Lx,
    __global const float* Ly,
    int lx_step, int lx_rows, int lx_cols,
    __global const float* keypoints_x,
    __global const float* keypoints_y,
    __global const float* keypoints_size,
    __global float* descriptors,
    int nkeypoints)
{
    int idx = get_global_id(0);
    if (idx >= nkeypoints)
        return;

    float kpt_x = keypoints_x[idx];
    float kpt_y = keypoints_y[idx];
    float kpt_sz = keypoints_size[idx];

    int scale = (int)(kpt_sz / 2.0f + 0.5f);
    float xf = kpt_x;
    float yf = kpt_y;

    const int dsize = 128;
    float len = 0.0f;
    int dcount = 0;

    for (int sr = 0; sr < 4; sr++)
    {
        int base_k = sr * 5 - 12;
        for (int sc = 0; sc < 4; sc++)
        {
            int base_l = sc * 5 - 12;

            float dxp = 0.0f, dxn = 0.0f, mdxp = 0.0f, mdxn = 0.0f;
            float dyp = 0.0f, dyn = 0.0f, mdyp = 0.0f, mdyn = 0.0f;

            for (int ri = 0; ri < 9; ri++)
            {
                int k = base_k + ri;
                float sample_y = (float)k * scale + yf;
                int y1 = (int)(sample_y - 0.5f);
                int y2 = (int)(sample_y + 0.5f);
                y1 = clamp(y1, 0, lx_rows - 1);
                y2 = clamp(y2, 0, lx_rows - 1);

                int row0 = y1 * lx_step;
                int row1 = y2 * lx_step;
                float fy = sample_y - (float)y1;
                float inv_fy = 1.0f - fy;

                for (int ci = 0; ci < 9; ci++)
                {
                    int l = base_l + ci;
                    float sample_x = (float)l * scale + xf;

                    int x1_raw = (int)(sample_x - 0.5f);
                    int x2_raw = (int)(sample_x + 0.5f);
                    int x1 = clamp(x1_raw, 0, lx_cols - 1);
                    int x2 = clamp(x2_raw, 0, lx_cols - 1);

                    float fx = sample_x - (float)x1;
                    float inv_fx = 1.0f - fx;

                    float gs1 = gauss_s1_weights[ri * 9 + ci];

                    bool use_v2 = (x1_raw >= 0 && x2_raw < lx_cols);

                    float lx00, lx01, ly00, ly01;
                    float lx10, lx11, ly10, ly11;

                    if (use_v2)
                    {
                        float2 vx = vload2(0, Lx + row0 + x1);
                        float2 vy = vload2(0, Ly + row0 + x1);
                        lx00 = vx.s0; lx01 = vx.s1;
                        ly00 = vy.s0; ly01 = vy.s1;
                    }
                    else
                    {
                        lx00 = Lx[row0 + x1];
                        lx01 = Lx[row0 + x2];
                        ly00 = Ly[row0 + x1];
                        ly01 = Ly[row0 + x2];
                    }

                    if (y1 != y2)
                    {
                        if (use_v2)
                        {
                            float2 vx = vload2(0, Lx + row1 + x1);
                            float2 vy = vload2(0, Ly + row1 + x1);
                            lx10 = vx.s0; lx11 = vx.s1;
                            ly10 = vy.s0; ly11 = vy.s1;
                        }
                        else
                        {
                            lx10 = Lx[row1 + x1];
                            lx11 = Lx[row1 + x2];
                            ly10 = Ly[row1 + x1];
                            ly11 = Ly[row1 + x2];
                        }
                    }
                    else
                    {
                        lx10 = lx00; lx11 = lx01;
                        ly10 = ly00; ly11 = ly01;
                    }

                    float rx = inv_fx * inv_fy * lx00 + fx * inv_fy * lx01
                             + inv_fx * fy * lx10 + fx * fy * lx11;
                    float ry = inv_fx * inv_fy * ly00 + fx * inv_fy * ly01
                             + inv_fx * fy * ly10 + fx * fy * ly11;

                    rx = gs1 * rx;
                    ry = gs1 * ry;

                    if (ry >= 0.0f) { dxp += rx;  mdxp += fabs(rx); }
                    else            { dxn += rx;  mdxn += fabs(rx); }
                    if (rx >= 0.0f) { dyp += ry;  mdyp += fabs(ry); }
                    else            { dyn += ry;  mdyn += fabs(ry); }
                }
            }

            float gs2 = gauss_s2_weights[sr * 4 + sc];

            descriptors[idx * dsize + dcount++] = dxp  * gs2;
            descriptors[idx * dsize + dcount++] = dxn  * gs2;
            descriptors[idx * dsize + dcount++] = mdxp * gs2;
            descriptors[idx * dsize + dcount++] = mdxn * gs2;
            descriptors[idx * dsize + dcount++] = dyp  * gs2;
            descriptors[idx * dsize + dcount++] = dyn  * gs2;
            descriptors[idx * dsize + dcount++] = mdyp * gs2;
            descriptors[idx * dsize + dcount++] = mdyn * gs2;

            len += (dxp*dxp + dxn*dxn + mdxp*mdxp + mdxn*mdxn
                  + dyp*dyp + dyn*dyn + mdyp*mdyp + mdyn*mdyn) * gs2 * gs2;
        }
    }

    len = sqrt(len);
    if (len > 1e-10f)
    {
        float len_inv = 1.0f / len;
        for (int i = 0; i < dsize; i++)
            descriptors[idx * dsize + i] *= len_inv;
    }
}

/**
 * @brief Compute KAZE rotation-invariant 64-dimensional descriptor (optimized)
 * @details Matches CPU Get_KAZE_Descriptor_64 exactly:
 *          - Uses keypoint angle to rotate sampling coordinates and derivatives
 *          - integer scale via round(kpt_size/2)
 *          - bilinear with cvFloor convention (matches CPU)
 *          - clamp-on-border (not skip)
 *
 * Optimizations:
 * - Pre-computed gaussian weights (__constant, no exp() calls in inner loop)
 * - Vectorized vload2 for bilinear pixel pairs
 * - Pre-computed row addresses outside ci loop
 *
 * @param Lx     First-order x-derivative (sigma_size-scaled), row-major float array
 * @param Ly     First-order y-derivative (sigma_size-scaled), row-major float array
 * @param lx_step    Row stride of Lx/Ly in float elements (= cols for continuous mat)
 * @param lx_rows    Image height
 * @param lx_cols    Image width
 * @param keypoints_x  Keypoint x coordinates
 * @param keypoints_y  Keypoint y coordinates
 * @param keypoints_size  Keypoint sizes (diameter = 2*sigma)
 * @param keypoints_angle Keypoint orientations in degrees
 * @param descriptors  Output descriptor matrix, shape [nkeypoints, 64]
 * @param nkeypoints   Number of keypoints
 */
__kernel void
KAZE_compute_descriptor_64(
    __global const float* Lx,
    __global const float* Ly,
    int lx_step, int lx_rows, int lx_cols,
    __global const float* keypoints_x,
    __global const float* keypoints_y,
    __global const float* keypoints_size,
    __global const float* keypoints_angle,
    __global float* descriptors,
    int nkeypoints)
{
    int idx = get_global_id(0);
    if (idx >= nkeypoints)
        return;

    float kpt_x = keypoints_x[idx];
    float kpt_y = keypoints_y[idx];
    float kpt_sz = keypoints_size[idx];
    float kpt_angle_deg = keypoints_angle[idx];

    int scale = (int)(kpt_sz / 2.0f + 0.5f);
    float xf = kpt_x;
    float yf = kpt_y;
    float angle = kpt_angle_deg * (float)(M_PI_F / 180.0f);
    float co = native_cos(angle);
    float si = native_sin(angle);

    const int dsize = 64;
    float len = 0.0f;
    int dcount = 0;

    for (int sr = 0; sr < 4; sr++)
    {
        int base_k = sr * 5 - 12;
        for (int sc = 0; sc < 4; sc++)
        {
            int base_l = sc * 5 - 12;

            float dx = 0.0f, dy = 0.0f, mdx = 0.0f, mdy = 0.0f;

            for (int ri = 0; ri < 9; ri++)
            {
                int k = base_k + ri;
                for (int ci = 0; ci < 9; ci++)
                {
                    int l = base_l + ci;

                    float sample_x = xf + (-l * scale * si + k * scale * co);
                    float sample_y = yf + (l * scale * co + k * scale * si);

                    int y1 = (int)floor(sample_y);
                    int x1 = (int)floor(sample_x);
                    y1 = clamp(y1, 0, lx_rows - 1);
                    x1 = clamp(x1, 0, lx_cols - 1);

                    int y2 = y1 + 1;
                    int x2 = x1 + 1;
                    y2 = clamp(y2, 0, lx_rows - 1);
                    x2 = clamp(x2, 0, lx_cols - 1);

                    float fx = sample_x - (float)x1;
                    float fy = sample_y - (float)y1;
                    float inv_fx = 1.0f - fx;
                    float inv_fy = 1.0f - fy;

                    float gs1 = gauss_s1_weights[ri * 9 + ci];

                    int row0 = y1 * lx_step;
                    int row1 = y2 * lx_step;

                    float lx00, lx01, lx10, lx11;
                    if (x1 < lx_cols - 1)
                    {
                        float2 v = vload2(0, Lx + row0 + x1);
                        lx00 = v.s0; lx01 = v.s1;
                    }
                    else
                    {
                        lx00 = Lx[row0 + x1];
                        lx01 = lx00;
                    }

                    if (y1 != y2)
                    {
                        if (x1 < lx_cols - 1)
                        {
                            float2 v = vload2(0, Lx + row1 + x1);
                            lx10 = v.s0; lx11 = v.s1;
                        }
                        else
                        {
                            lx10 = Lx[row1 + x1];
                            lx11 = lx10;
                        }
                    }
                    else
                    {
                        lx10 = lx00;
                        lx11 = lx01;
                    }

                    float ly00, ly01, ly10, ly11;
                    if (x1 < lx_cols - 1)
                    {
                        float2 v = vload2(0, Ly + row0 + x1);
                        ly00 = v.s0; ly01 = v.s1;
                    }
                    else
                    {
                        ly00 = Ly[row0 + x1];
                        ly01 = ly00;
                    }

                    if (y1 != y2)
                    {
                        if (x1 < lx_cols - 1)
                        {
                            float2 v = vload2(0, Ly + row1 + x1);
                            ly10 = v.s0; ly11 = v.s1;
                        }
                        else
                        {
                            ly10 = Ly[row1 + x1];
                            ly11 = ly10;
                        }
                    }
                    else
                    {
                        ly10 = ly00;
                        ly11 = ly01;
                    }

                    float rx = inv_fx * inv_fy * lx00 + fx * inv_fy * lx01
                             + inv_fx * fy * lx10 + fx * fy * lx11;
                    float ry = inv_fx * inv_fy * ly00 + fx * inv_fy * ly01
                             + inv_fx * fy * ly10 + fx * fy * ly11;

                    float rry = gs1 * (rx * co + ry * si);
                    float rrx = gs1 * (-rx * si + ry * co);

                    dx += rrx;
                    dy += rry;
                    mdx += fabs(rrx);
                    mdy += fabs(rry);
                }
            }

            float gs2 = gauss_s2_weights[sr * 4 + sc];

            descriptors[idx * dsize + dcount++] = dx * gs2;
            descriptors[idx * dsize + dcount++] = dy * gs2;
            descriptors[idx * dsize + dcount++] = mdx * gs2;
            descriptors[idx * dsize + dcount++] = mdy * gs2;

            len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy) * gs2 * gs2;
        }
    }

    len = sqrt(len);
    if (len > 1e-10f)
    {
        float len_inv = 1.0f / len;
        for (int i = 0; i < dsize; i++)
            descriptors[idx * dsize + i] *= len_inv;
    }
}

/**
 * @brief Compute KAZE rotation-invariant 128-dimensional descriptor (optimized)
 * @details Matches CPU Get_KAZE_Descriptor_128: splits derivatives by sign of cross-component
 *          for 8 values per subregion, with rotation applied
 *
 * Optimizations:
 * - Pre-computed gaussian weights (__constant, no exp() calls in inner loop)
 * - Vectorized vload2 for bilinear pixel pairs
 */
__kernel void
KAZE_compute_descriptor_128(
    __global const float* Lx,
    __global const float* Ly,
    int lx_step, int lx_rows, int lx_cols,
    __global const float* keypoints_x,
    __global const float* keypoints_y,
    __global const float* keypoints_size,
    __global const float* keypoints_angle,
    __global float* descriptors,
    int nkeypoints)
{
    int idx = get_global_id(0);
    if (idx >= nkeypoints)
        return;

    float kpt_x = keypoints_x[idx];
    float kpt_y = keypoints_y[idx];
    float kpt_sz = keypoints_size[idx];
    float kpt_angle_deg = keypoints_angle[idx];

    int scale = (int)(kpt_sz / 2.0f + 0.5f);
    float xf = kpt_x;
    float yf = kpt_y;
    float angle = kpt_angle_deg * (float)(M_PI_F / 180.0f);
    float co = native_cos(angle);
    float si = native_sin(angle);

    const int dsize = 128;
    float len = 0.0f;
    int dcount = 0;

    for (int sr = 0; sr < 4; sr++)
    {
        int base_k = sr * 5 - 12;
        for (int sc = 0; sc < 4; sc++)
        {
            int base_l = sc * 5 - 12;

            float dxp = 0.0f, dxn = 0.0f, mdxp = 0.0f, mdxn = 0.0f;
            float dyp = 0.0f, dyn = 0.0f, mdyp = 0.0f, mdyn = 0.0f;

            for (int ri = 0; ri < 9; ri++)
            {
                int k = base_k + ri;
                for (int ci = 0; ci < 9; ci++)
                {
                    int l = base_l + ci;

                    float sample_x = xf + (-l * scale * si + k * scale * co);
                    float sample_y = yf + (l * scale * co + k * scale * si);

                    int y1 = (int)floor(sample_y);
                    int x1 = (int)floor(sample_x);
                    y1 = clamp(y1, 0, lx_rows - 1);
                    x1 = clamp(x1, 0, lx_cols - 1);

                    int y2 = y1 + 1;
                    int x2 = x1 + 1;
                    y2 = clamp(y2, 0, lx_rows - 1);
                    x2 = clamp(x2, 0, lx_cols - 1);

                    float fx = sample_x - (float)x1;
                    float fy = sample_y - (float)y1;
                    float inv_fx = 1.0f - fx;
                    float inv_fy = 1.0f - fy;

                    float gs1 = gauss_s1_weights[ri * 9 + ci];

                    int row0 = y1 * lx_step;
                    int row1 = y2 * lx_step;

                    float lx00, lx01, lx10, lx11;
                    if (x1 < lx_cols - 1)
                    {
                        float2 v = vload2(0, Lx + row0 + x1);
                        lx00 = v.s0; lx01 = v.s1;
                    }
                    else
                    {
                        lx00 = Lx[row0 + x1];
                        lx01 = lx00;
                    }

                    if (y1 != y2)
                    {
                        if (x1 < lx_cols - 1)
                        {
                            float2 v = vload2(0, Lx + row1 + x1);
                            lx10 = v.s0; lx11 = v.s1;
                        }
                        else
                        {
                            lx10 = Lx[row1 + x1];
                            lx11 = lx10;
                        }
                    }
                    else
                    {
                        lx10 = lx00;
                        lx11 = lx01;
                    }

                    float ly00, ly01, ly10, ly11;
                    if (x1 < lx_cols - 1)
                    {
                        float2 v = vload2(0, Ly + row0 + x1);
                        ly00 = v.s0; ly01 = v.s1;
                    }
                    else
                    {
                        ly00 = Ly[row0 + x1];
                        ly01 = ly00;
                    }

                    if (y1 != y2)
                    {
                        if (x1 < lx_cols - 1)
                        {
                            float2 v = vload2(0, Ly + row1 + x1);
                            ly10 = v.s0; ly11 = v.s1;
                        }
                        else
                        {
                            ly10 = Ly[row1 + x1];
                            ly11 = ly10;
                        }
                    }
                    else
                    {
                        ly10 = ly00;
                        ly11 = ly01;
                    }

                    float rx = inv_fx * inv_fy * lx00 + fx * inv_fy * lx01
                             + inv_fx * fy * lx10 + fx * fy * lx11;
                    float ry = inv_fx * inv_fy * ly00 + fx * inv_fy * ly01
                             + inv_fx * fy * ly10 + fx * fy * ly11;

                    float rry = gs1 * (rx * co + ry * si);
                    float rrx = gs1 * (-rx * si + ry * co);

                    if (rry >= 0.0f) {
                        dxp += rrx;
                        mdxp += fabs(rrx);
                    }
                    else {
                        dxn += rrx;
                        mdxn += fabs(rrx);
                    }

                    if (rrx >= 0.0f) {
                        dyp += rry;
                        mdyp += fabs(rry);
                    }
                    else {
                        dyn += rry;
                        mdyn += fabs(rry);
                    }
                }
            }

            float gs2 = gauss_s2_weights[sr * 4 + sc];

            descriptors[idx * dsize + dcount++] = dxp * gs2;
            descriptors[idx * dsize + dcount++] = dxn * gs2;
            descriptors[idx * dsize + dcount++] = mdxp * gs2;
            descriptors[idx * dsize + dcount++] = mdxn * gs2;
            descriptors[idx * dsize + dcount++] = dyp * gs2;
            descriptors[idx * dsize + dcount++] = dyn * gs2;
            descriptors[idx * dsize + dcount++] = mdyp * gs2;
            descriptors[idx * dsize + dcount++] = mdyn * gs2;

            len += (dxp*dxp + dxn*dxn + mdxp*mdxp + mdxn*mdxn
                  + dyp*dyp + dyn*dyn + mdyp*mdyp + mdyn*mdyn) * gs2 * gs2;
        }
    }

    len = sqrt(len);
    if (len > 1e-10f)
    {
        float len_inv = 1.0f / len;
        for (int i = 0; i < dsize; i++)
            descriptors[idx * dsize + i] *= len_inv;
    }
}

/**
 * @brief Fused kernel: squared gradient magnitude from Lx/Ly derivatives.
 * @details Replaces multiply(Lx^2) + multiply(Ly^2) + add(mag2) with a single pass.
 *          Lx and Ly are already computed by Scharr. Writes mag2 for minMaxLoc.
 *
 * @param Lx Input x-derivative
 * @param Ly Input y-derivative
 * @param step Row stride of Lx/Ly in float elements
 * @param cols Image width
 * @param rows Image height
 * @param mag2_out Output squared magnitude (Lx^2 + Ly^2)
 * @param mag2_step Row stride of mag2_out in float elements
 */
__kernel void
KAZE_compute_mag2(
    __global const float* Lx,
    __global const float* Ly,
    int step, int cols, int rows,
    __global float* mag2_out, int mag2_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= cols || y >= rows)
        return;

    int idx = y * step + x;
    float lx = Lx[idx];
    float ly = Ly[idx];

    mag2_out[y * mag2_step + x] = lx * lx + ly * ly;
}

/**
 * @brief Compute histogram of gradient magnitudes for kcontrast estimation.
 * @details Each workgroup builds a local histogram using local atomics, then
 *          merges to global via atomic_add. Skips border pixels (1px border)
 *          to match CPU compute_k_percentile.
 *
 * @param Lx     Scharr x-derivative of Lsmooth (row-major float)
 * @param Ly     Scharr y-derivative of Lsmooth (row-major float)
 * @param lx_step    Row stride of Lx/Ly in float elements
 * @param cols   Image width
 * @param rows   Image height
 * @param hmax   sqrt(max(Lx^2 + Ly^2)) gradient magnitude max
 * @param nbins  Number of histogram bins (e.g. 300)
 * @param histogram  Output histogram (single global histogram, size=nbins)
 * @param local_hist Local memory buffer, size=nbins ints
 */
__kernel void
KAZE_compute_kcontrast_histogram(
    __global const float* Lx,
    __global const float* Ly,
    int lx_step, int cols, int rows,
    float hmax, int nbins,
    __global int* histogram,
    __local int* local_hist)
{
    for (int i = get_local_id(0); i < nbins; i += get_local_size(0))
        local_hist[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    int x = get_global_id(0);
    int y = get_global_id(1);

    // Skip borders to match CPU compute_k_percentile
    if (x > 0 && x < cols - 1 && y > 0 && y < rows - 1)
    {
        int idx = y * lx_step + x;
        float lx_val = Lx[idx];
        float ly_val = Ly[idx];
        float mag_sq = lx_val * lx_val + ly_val * ly_val;

        if (mag_sq > 0.0f)
        {
            float mag = native_sqrt(mag_sq);
            int bin = (int)(nbins * mag / hmax);
            if (bin == nbins)
                bin--;
            atomic_inc(&local_hist[bin]);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Merge to global using atomics (one thread per workgroup)
    if (get_local_id(0) == 0 && get_local_id(1) == 0)
    {
        for (int i = 0; i < nbins; i++)
        {
            if (local_hist[i] > 0)
                atomic_add(&histogram[i], local_hist[i]);
        }
    }
}

/**
 * @brief Compute KAZE main orientation for keypoints (one pyramid level)
 * @details Matches CPU Compute_Main_Orientation:
 *          - Sample 13x13 window (i^2+j^2 < 36) with Gaussian weighting
 *          - Find dominant orientation using pi/3 sliding window
 *
 * @param Lx     First-order x-derivative, row-major float array
 * @param Ly     First-order y-derivative, row-major float array
 * @param lx_step    Row stride of Lx/Ly in float elements
 * @param lx_rows    Image height
 * @param lx_cols    Image width
 * @param keypoints_x  Keypoint x coordinates
 * @param keypoints_y  Keypoint y coordinates
 * @param keypoints_size  Keypoint sizes
 * @param keypoints_angle  Output angles in degrees
 * @param nkeypoints   Number of keypoints
 */
__kernel void
KAZE_compute_orientation_level(
    __global const float* Lx,
    __global const float* Ly,
    int lx_step, int lx_rows, int lx_cols,
    __global const float* keypoints_x,
    __global const float* keypoints_y,
    __global const float* keypoints_size,
    __global float* keypoints_angle,
    int nkeypoints)
{
    int idx = get_global_id(0);
    if (idx >= nkeypoints)
        return;

    float xf = keypoints_x[idx];
    float yf = keypoints_y[idx];
    float kpt_sz = keypoints_size[idx];
    int scale = (int)(kpt_sz / 2.0f + 0.5f);

    float max_mag_sq = -1.0f;
    float final_angle = 0.0f;

    // Store samples to avoid redundant global memory reads in the sliding window loop
    // Circular mask i^2+j^2 < 36 has 109 points.
    float samplesX[109];
    float samplesY[109];
    float samplesAng[109];
    int sample_count = 0;

    for (int i = -6; i <= 6; ++i) {
        for (int j = -6; j <= 6; ++j) {
            if (i*i + j*j < 36) {
                int ix = (int)(xf + i * scale + 0.5f);
                int iy = (int)(yf + j * scale + 0.5f);

                if (iy >= 0 && iy < lx_rows && ix >= 0 && ix < lx_cols) {
                    float gw = gaussian((float)iy - yf, (float)ix - xf, 2.5f * scale);
                    float rx = gw * Lx[iy * lx_step + ix];
                    float ry = gw * Ly[iy * lx_step + ix];
                    samplesX[sample_count] = rx;
                    samplesY[sample_count] = ry;
                    float ang = atan2(ry, rx);
                    if (ang < 0.0f) ang += 2.0f * M_PI_F;
                    samplesAng[sample_count] = ang;
                } else {
                    samplesX[sample_count] = 0.0f;
                    samplesY[sample_count] = 0.0f;
                    samplesAng[sample_count] = 0.0f;
                }
                sample_count++;
            }
        }
    }

    // Sliding window loop: Slides pi/3 window around feature point
    for (float ang1 = 0.0f; ang1 < 2.0f * M_PI_F; ang1 += 0.15f) {
        float ang2 = ang1 + (M_PI_F / 3.0f);
        if (ang2 > 2.0f * M_PI_F) {
            ang2 -= 2.0f * M_PI_F;
        }

        float currSumX = 0.0f;
        float currSumY = 0.0f;

        for (int k = 0; k < sample_count; ++k) {
            float ang = samplesAng[k];
            bool in_window = false;
            if (ang1 < ang2) {
                if (ang > ang1 && ang < ang2) in_window = true;
            } else {
                if (ang > ang1 || ang < ang2) in_window = true;
            }

            if (in_window) {
                currSumX += samplesX[k];
                currSumY += samplesY[k];
            }
        }

        float mag_sq = currSumX * currSumX + currSumY * currSumY;
        if (mag_sq > max_mag_sq) {
            max_mag_sq = mag_sq;
            final_angle = atan2(currSumY, currSumX);
        }
    }

    // Convert radians to degrees
    keypoints_angle[idx] = final_angle * (180.0f / M_PI_F);
}
