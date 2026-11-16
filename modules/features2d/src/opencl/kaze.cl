// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

inline float gaussian(float x, float y, float sigma)
{
    return exp(-(x*x + y*y) / (2.0f*sigma*sigma));
}

/**
 * @brief Compute KAZE upright 64-dimensional descriptor
 * @details Matches CPU Get_KAZE_Upright_Descriptor_64 exactly:
 *          - integer scale via round(kpt_size/2)
 *          - bilinear with y1=(int)(y-0.5), y2=(int)(y+0.5) convention
 *          - clamp-on-border (not skip)
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

    const int dsize = 64;
    const int sample_step = 5;
    const int pattern_size = 12;

    float kpt_x = keypoints_x[idx];
    float kpt_y = keypoints_y[idx];
    float kpt_sz = keypoints_size[idx];

    int scale = (int)(kpt_sz / 2.0f + 0.5f);  // cvRound equivalent
    float xf = kpt_x;
    float yf = kpt_y;

    float dx = 0.0f, dy = 0.0f, mdx = 0.0f, mdy = 0.0f;
    float gauss_s1, gauss_s2;
    float rx, ry;
    float sample_x, sample_y;
    int x1, y1, x2, y2;
    int kx, ky, i, j, dcount = 0;
    float fx, fy;
    float res1, res2, res3, res4;
    float len = 0.0f;

    float cx = -0.5f, cy = 0.5f;

    i = -8;
    while (i < pattern_size)
    {
        j = -8;
        i = i - 4;
        cx += 1.0f;
        cy = -0.5f;

        while (j < pattern_size)
        {
            dx = dy = mdx = mdy = 0.0f;
            cy += 1.0f;
            j = j - 4;

            ky = i + sample_step;
            kx = j + sample_step;

            float ys = yf + (float)(ky * scale);
            float xs = xf + (float)(kx * scale);

            for (int k = i; k < i + 9; k++)
            {
                for (int l = j; l < j + 9; l++)
                {
                    sample_y = (float)k * scale + yf;
                    sample_x = (float)l * scale + xf;

                    gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.5f * scale);

                    // Match CPU bilinear convention: y1=(int)(y-0.5), y2=(int)(y+0.5)
                    y1 = (int)(sample_y - 0.5f);
                    x1 = (int)(sample_x - 0.5f);
                    y1 = clamp(y1, 0, lx_rows - 1);
                    x1 = clamp(x1, 0, lx_cols - 1);

                    y2 = (int)(sample_y + 0.5f);
                    x2 = (int)(sample_x + 0.5f);
                    y2 = clamp(y2, 0, lx_rows - 1);
                    x2 = clamp(x2, 0, lx_cols - 1);

                    fx = sample_x - (float)x1;
                    fy = sample_y - (float)y1;

                    res1 = Lx[y1 * lx_step + x1];
                    res2 = Lx[y1 * lx_step + x2];
                    res3 = Lx[y2 * lx_step + x1];
                    res4 = Lx[y2 * lx_step + x2];
                    rx = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2
                       + (1.0f - fx)*fy*res3 + fx*fy*res4;

                    res1 = Ly[y1 * lx_step + x1];
                    res2 = Ly[y1 * lx_step + x2];
                    res3 = Ly[y2 * lx_step + x1];
                    res4 = Ly[y2 * lx_step + x2];
                    ry = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2
                       + (1.0f - fx)*fy*res3 + fx*fy*res4;

                    rx = gauss_s1 * rx;
                    ry = gauss_s1 * ry;

                    dx  += rx;
                    dy  += ry;
                    mdx += fabs(rx);
                    mdy += fabs(ry);
                }
            }

            gauss_s2 = gaussian(cx - 2.0f, cy - 2.0f, 1.5f);

            descriptors[idx * dsize + dcount++] = dx  * gauss_s2;
            descriptors[idx * dsize + dcount++] = dy  * gauss_s2;
            descriptors[idx * dsize + dcount++] = mdx * gauss_s2;
            descriptors[idx * dsize + dcount++] = mdy * gauss_s2;

            len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy) * gauss_s2 * gauss_s2;

            j += 9;
        }
        i += 9;
    }

    // L2 normalize
    len = sqrt(len);
    if (len > 1e-10f)
    {
        float len_inv = 1.0f / len;
        for (i = 0; i < dsize; i++)
            descriptors[idx * dsize + i] *= len_inv;
    }
}

/**
 * @brief Compute KAZE upright 128-dimensional descriptor (extended mode)
 * @details Matches CPU Get_KAZE_Upright_Descriptor_128: splits dx/dy by sign of the
 *          cross-component for 8 values per subregion.
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

    const int dsize = 128;
    const int sample_step = 5;
    const int pattern_size = 12;

    float kpt_x = keypoints_x[idx];
    float kpt_y = keypoints_y[idx];
    float kpt_sz = keypoints_size[idx];

    int scale = (int)(kpt_sz / 2.0f + 0.5f);
    float xf = kpt_x;
    float yf = kpt_y;

    float dxp = 0.0f, dxn = 0.0f, mdxp = 0.0f, mdxn = 0.0f;
    float dyp = 0.0f, dyn = 0.0f, mdyp = 0.0f, mdyn = 0.0f;
    float gauss_s1, gauss_s2;
    float rx, ry;
    float sample_x, sample_y;
    int x1, y1, x2, y2;
    int kx, ky, i, j, dcount = 0;
    float fx, fy;
    float res1, res2, res3, res4;
    float len = 0.0f;

    float cx = -0.5f, cy = 0.5f;

    i = -8;
    while (i < pattern_size)
    {
        j = -8;
        i = i - 4;
        cx += 1.0f;
        cy = -0.5f;

        while (j < pattern_size)
        {
            dxp = dxn = mdxp = mdxn = 0.0f;
            dyp = dyn = mdyp = mdyn = 0.0f;
            cy += 1.0f;
            j = j - 4;

            ky = i + sample_step;
            kx = j + sample_step;

            float ys = yf + (float)(ky * scale);
            float xs = xf + (float)(kx * scale);

            for (int k = i; k < i + 9; k++)
            {
                for (int l = j; l < j + 9; l++)
                {
                    sample_y = (float)k * scale + yf;
                    sample_x = (float)l * scale + xf;

                    gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.5f * scale);

                    y1 = (int)(sample_y - 0.5f);
                    x1 = (int)(sample_x - 0.5f);
                    y1 = clamp(y1, 0, lx_rows - 1);
                    x1 = clamp(x1, 0, lx_cols - 1);

                    y2 = (int)(sample_y + 0.5f);
                    x2 = (int)(sample_x + 0.5f);
                    y2 = clamp(y2, 0, lx_rows - 1);
                    x2 = clamp(x2, 0, lx_cols - 1);

                    fx = sample_x - (float)x1;
                    fy = sample_y - (float)y1;

                    res1 = Lx[y1 * lx_step + x1];
                    res2 = Lx[y1 * lx_step + x2];
                    res3 = Lx[y2 * lx_step + x1];
                    res4 = Lx[y2 * lx_step + x2];
                    rx = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2
                       + (1.0f - fx)*fy*res3 + fx*fy*res4;

                    res1 = Ly[y1 * lx_step + x1];
                    res2 = Ly[y1 * lx_step + x2];
                    res3 = Ly[y2 * lx_step + x1];
                    res4 = Ly[y2 * lx_step + x2];
                    ry = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2
                       + (1.0f - fx)*fy*res3 + fx*fy*res4;

                    rx = gauss_s1 * rx;
                    ry = gauss_s1 * ry;

                    if (ry >= 0.0f) { dxp += rx;  mdxp += fabs(rx); }
                    else            { dxn += rx;  mdxn += fabs(rx); }
                    if (rx >= 0.0f) { dyp += ry;  mdyp += fabs(ry); }
                    else            { dyn += ry;  mdyn += fabs(ry); }
                }
            }

            gauss_s2 = gaussian(cx - 2.0f, cy - 2.0f, 1.5f);

            descriptors[idx * dsize + dcount++] = dxp  * gauss_s2;
            descriptors[idx * dsize + dcount++] = dxn  * gauss_s2;
            descriptors[idx * dsize + dcount++] = mdxp * gauss_s2;
            descriptors[idx * dsize + dcount++] = mdxn * gauss_s2;
            descriptors[idx * dsize + dcount++] = dyp  * gauss_s2;
            descriptors[idx * dsize + dcount++] = dyn  * gauss_s2;
            descriptors[idx * dsize + dcount++] = mdyp * gauss_s2;
            descriptors[idx * dsize + dcount++] = mdyn * gauss_s2;

            len += (dxp*dxp + dxn*dxn + mdxp*mdxp + mdxn*mdxn
                  + dyp*dyp + dyn*dyn + mdyp*mdyp + mdyn*mdyn) * gauss_s2 * gauss_s2;

            j += 9;
        }
        i += 9;
    }

    len = sqrt(len);
    if (len > 1e-10f)
    {
        float len_inv = 1.0f / len;
        for (i = 0; i < dsize; i++)
            descriptors[idx * dsize + i] *= len_inv;
    }
}
