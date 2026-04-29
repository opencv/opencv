// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


/**
 * @brief This function computes the Perona and Malik conductivity coefficient g2
 * g2 = 1 / (1 + dL^2 / k^2)
 * @param lx First order image derivative in X-direction (horizontal)
 * @param ly First order image derivative in Y-direction (vertical)
 * @param dst Output image
 * @param k Contrast factor parameter
 */
__kernel void
AKAZE_pm_g2(__global const float* lx, __global const float* ly, __global float* dst,
    float k, int size)
{
    int i = get_global_id(0);
    // OpenCV plays with dimensions so we need explicit check for this
    if (!(i < size))
    {
        return;
    }

    const float k2inv = 1.0f / (k * k);
    dst[i] = 1.0f / (1.0f + ((lx[i] * lx[i] + ly[i] * ly[i]) * k2inv));
}

__kernel void
AKAZE_nld_step_scalar(__global const float* lt, int lt_step, int lt_offset, int rows, int cols,
    __global const float* lf, __global float* dst, float step_size)
{
    /* The labeling scheme for this five star stencil:
        [    a    ]
        [ -1 c +1 ]
        [    b    ]
    */
    // column-first indexing
    int i = get_global_id(1);
    int j = get_global_id(0);

    // OpenCV plays with dimensions so we need explicit check for this
    if (!(i < rows && j < cols))
    {
        return;
    }

    // get row indexes
    int a = (i - 1) * cols;
    int c = (i    ) * cols;
    int b = (i + 1) * cols;
    // compute stencil
    float res = 0.0f;
    if (i == 0) // first rows
    {
        if (j == 0 || j == (cols - 1))
        {
            res = 0.0f;
        } else
        {
            res = (lf[c + j] + lf[c + j + 1])*(lt[c + j + 1] - lt[c + j]) +
                  (lf[c + j] + lf[c + j - 1])*(lt[c + j - 1] - lt[c + j]) +
                  (lf[c + j] + lf[b + j    ])*(lt[b + j    ] - lt[c + j]);
        }
    } else if (i == (rows - 1)) // last row
    {
        if (j == 0 || j == (cols - 1))
        {
            res = 0.0f;
        } else
        {
            res = (lf[c + j] + lf[c + j + 1])*(lt[c + j + 1] - lt[c + j]) +
                  (lf[c + j] + lf[c + j - 1])*(lt[c + j - 1] - lt[c + j]) +
                  (lf[c + j] + lf[a + j    ])*(lt[a + j    ] - lt[c + j]);
        }
    } else // inner rows
    {
        if (j == 0) // first column
        {
            res = (lf[c + 0] + lf[c + 1])*(lt[c + 1] - lt[c + 0]) +
                  (lf[c + 0] + lf[b + 0])*(lt[b + 0] - lt[c + 0]) +
                  (lf[c + 0] + lf[a + 0])*(lt[a + 0] - lt[c + 0]);
        } else if (j == (cols - 1)) // last column
        {
            res = (lf[c + j] + lf[c + j - 1])*(lt[c + j - 1] - lt[c + j]) +
                  (lf[c + j] + lf[b + j    ])*(lt[b + j    ] - lt[c + j]) +
                  (lf[c + j] + lf[a + j    ])*(lt[a + j    ] - lt[c + j]);
        } else // inner stencil
        {
            res = (lf[c + j] + lf[c + j + 1])*(lt[c + j + 1] - lt[c + j]) +
                  (lf[c + j] + lf[c + j - 1])*(lt[c + j - 1] - lt[c + j]) +
                  (lf[c + j] + lf[b + j    ])*(lt[b + j    ] - lt[c + j]) +
                  (lf[c + j] + lf[a + j    ])*(lt[a + j    ] - lt[c + j]);
        }
    }

    dst[c + j] = res * step_size;
}

/**
 * @brief Compute determinant from hessians
 * @details Compute Ldet by (Lxx.mul(Lyy) - Lxy.mul(Lxy)) * sigma
 *
 * @param lxx spatial derivates
 * @param lxy spatial derivates
 * @param lyy spatial derivates
 * @param dst output determinant
 * @param sigma determinant will be scaled by this sigma
 */
__kernel void
AKAZE_compute_determinant(__global const float* lxx, __global const float* lxy, __global const float* lyy,
    __global float* dst, float sigma, int size)
{
    int i = get_global_id(0);
    // OpenCV plays with dimensions so we need explicit check for this
    if (!(i < size))
    {
        return;
    }

    dst[i] = (lxx[i] * lyy[i] - lxy[i] * lxy[i]) * sigma;
}

/**
 * @brief Find scale space extrema in 3x3 neighborhood
 * @details Detects local maxima in Hessian response within 3x3 neighborhood
 *
 * @param ldet Hessian determinant response
 * @param rows image height
 * @param cols image width
 * @param threshold detector response threshold
 * @param border border to ignore
 * @param keypoint_mask output binary mask of detected keypoints
 */
__kernel void
AKAZE_find_extrema_same_scale(__global const float* ldet, int rows, int cols,
    float threshold, int border, __global uchar* keypoint_mask)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Check bounds
    if (x < border || x >= cols - border || y < border || y >= rows - border)
    {
        return;
    }

    int idx = y * cols + x;
    float value = ldet[idx];

    // Filter by threshold
    if (value <= threshold)
    {
        keypoint_mask[idx] = 0;
        return;
    }

    // 3x3 local maxima check
    // Check horizontal neighbors
    if (value <= ldet[idx - 1] || value <= ldet[idx + 1])
    {
        keypoint_mask[idx] = 0;
        return;
    }

    // Check vertical neighbors
    int prev_row = (y - 1) * cols + x;
    int next_row = (y + 1) * cols + x;
    if (value <= ldet[prev_row] || value <= ldet[next_row])
    {
        keypoint_mask[idx] = 0;
        return;
    }

    // Check diagonal neighbors
    if (value <= ldet[prev_row - 1] || value <= ldet[prev_row + 1] ||
        value <= ldet[next_row - 1] || value <= ldet[next_row + 1])
    {
        keypoint_mask[idx] = 0;
        return;
    }

    // This is a local maximum
    keypoint_mask[idx] = 1;
}

/**
 * @brief Cross-scale non-maximum suppression (lower scale filtering)
 * @details For each keypoint in current level, project to lower level and suppress if weaker
 *
 * @param keypoints_current keypoint mask for current scale level
 * @param keypoints_lower keypoint mask for lower scale level
 * @param ldet_current Hessian response for current level
 * @param ldet_lower Hessian response for lower level
 * @param rows image height
 * @param cols image width
 * @param diff_ratio ratio to project from current to lower level
 * @param search_radius search radius in lower level
 */
__kernel void
AKAZE_cross_scale_filter_lower(__global const uchar* keypoints_current,
    __global uchar* keypoints_lower,
    __global const float* ldet_current,
    __global const float* ldet_lower,
    int rows, int cols,
    int lower_rows, int lower_cols,
    int diff_ratio, int search_radius)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= cols || y >= rows)
        return;

    int idx = y * cols + x;

    // Only process keypoints in current level
    if (keypoints_current[idx] == 0)
        return;

    // Project to lower scale level
    int p_x = x * diff_ratio;
    int p_y = y * diff_ratio;

    // Search for neighbor in lower level within radius
    int radius_sq = search_radius * search_radius;
    int found = 0;
    int neighbor_idx = -1;

    // Brute force search within radius
    for (int dy = -search_radius; dy <= search_radius; dy++)
    {
        for (int dx = -search_radius; dx <= search_radius; dx++)
        {
            int nx = p_x + dx;
            int ny = p_y + dy;

            if (nx >= 0 && nx < lower_cols && ny >= 0 && ny < lower_rows)
            {
                int nidx = ny * lower_cols + nx;
                if (keypoints_lower[nidx] == 1)
                {
                    if (dx * dx + dy * dy <= radius_sq)
                    {
                        found = 1;
                        neighbor_idx = nidx;
                        break;
                    }
                }
            }
        }
        if (found) break;
    }

    // If neighbor found and current response is higher, suppress neighbor
    if (found && ldet_current[idx] > ldet_lower[neighbor_idx])
    {
        keypoints_lower[neighbor_idx] = 0;
    }
}

/**
 * @brief Cross-scale non-maximum suppression (upper scale filtering)
 * @details For each keypoint in current level, project to upper level and suppress if weaker
 *
 * @param keypoints_current keypoint mask for current scale level
 * @param keypoints_upper keypoint mask for upper scale level
 * @param ldet_current Hessian response for current level
 * @param ldet_upper Hessian response for upper level
 * @param rows image height
 * @param cols image width
 * @param diff_ratio ratio to project from current to upper level
 * @param search_radius search radius in upper level
 */
__kernel void
AKAZE_cross_scale_filter_upper(__global const uchar* keypoints_current,
    __global uchar* keypoints_upper,
    __global const float* ldet_current,
    __global const float* ldet_upper,
    int rows, int cols,
    int upper_rows, int upper_cols,
    int diff_ratio, int search_radius)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= cols || y >= rows)
        return;

    int idx = y * cols + x;

    // Only process keypoints in current level
    if (keypoints_current[idx] == 0)
        return;

    // Project to upper scale level
    int p_x = x / diff_ratio;
    int p_y = y / diff_ratio;

    // Search for neighbor in upper level within radius
    int radius_sq = search_radius * search_radius;
    int found = 0;
    int neighbor_idx = -1;

    // Brute force search within radius
    for (int dy = -search_radius; dy <= search_radius; dy++)
    {
        for (int dx = -search_radius; dx <= search_radius; dx++)
        {
            int nx = p_x + dx;
            int ny = p_y + dy;

            if (nx >= 0 && nx < upper_cols && ny >= 0 && ny < upper_rows)
            {
                int nidx = ny * upper_cols + nx;
                if (keypoints_upper[nidx] == 1)
                {
                    if (dx * dx + dy * dy <= radius_sq)
                    {
                        found = 1;
                        neighbor_idx = nidx;
                        break;
                    }
                }
            }
        }
        if (found) break;
    }

    // If neighbor found and current response is higher, suppress neighbor
    if (found && ldet_current[idx] > ldet_upper[neighbor_idx])
    {
        keypoints_upper[neighbor_idx] = 0;
    }
}

/**
 * @brief Combined subpixel refinement and orientation for single level
 * @details Processes subpixel refinement and orientation together to eliminate intermediate transfers
 *
 * @param keypoints keypoint mask for this level
 * @param ldet Hessian response for this level
 * @param Lx gradient in x direction for this level
 * @param Ly gradient in y direction for this level
 * @param rows image height
 * @param cols image width
 * @param octave_ratio scale ratio for this level
 * @param esigma evolution sigma for this level
 * @param octave octave number for this level
 * @param level evolution level index
 * @param output_count atomic counter for number of refined keypoints
 * @param output_x output x coordinates
 * @param output_y output y coordinates
 * @param output_size output keypoint sizes
 * @param output_response output keypoint responses
 * @param output_octave output octave numbers
 * @param output_class_id output level indices
 * @param output_angle output orientation angles
 * @param max_output maximum number of keypoints that can be output
 */
__kernel void
AKAZE_subpixel_refinement_orientation(__global const uchar* keypoints,
    __global const float* ldet,
    __global const float* Lx,
    __global const float* Ly,
    int rows, int cols,
    float octave_ratio, float esigma, int octave, int level,
    __global int* output_count,
    __global float* output_x,
    __global float* output_y,
    __global float* output_size,
    __global float* output_response,
    __global int* output_octave,
    __global int* output_class_id,
    __global float* output_angle,
    int max_output)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= cols || y >= rows)
        return;

    int idx = y * cols + x;

    // Only process keypoints
    if (keypoints[idx] == 0)
        return;

    // Compute gradient for subpixel refinement
    float Dx = 0.5f * (ldet[idx + 1] - ldet[idx - 1]);
    float Dy = 0.5f * (ldet[idx + cols] - ldet[idx - cols]);

    // Compute Hessian
    float Dxx = ldet[idx + 1] + ldet[idx - 1] - 2.0f * ldet[idx];
    float Dyy = ldet[idx + cols] + ldet[idx - cols] - 2.0f * ldet[idx];
    float Dxy = 0.25f * (ldet[idx + cols + 1] + ldet[idx - cols - 1] -
                        ldet[idx - cols + 1] - ldet[idx + cols - 1]);

    // Solve 2x2 linear system
    float det = Dxx * Dyy - Dxy * Dxy;

    if (fabs(det) < 1e-10f)
        return;

    float inv_det = 1.0f / det;
    float dx = (-Dyy * Dx + Dxy * Dy) * inv_det;
    float dy = (Dxy * Dx - Dxx * Dy) * inv_det;

    if (fabs(dx) > 1.0f || fabs(dy) > 1.0f)
        return;

    int out_idx = atomic_inc(output_count);

    if (out_idx >= max_output)
        return;

    float refined_x = x * octave_ratio + dx * octave_ratio + 0.5f * (octave_ratio - 1.0f);
    float refined_y = y * octave_ratio + dy * octave_ratio + 0.5f * (octave_ratio - 1.0f);

    output_x[out_idx] = refined_x;
    output_y[out_idx] = refined_y;
    output_size[out_idx] = esigma * 3.0f; // derivative_factor(1.5) * 2.0(diameter) = 3.0
    output_response[out_idx] = ldet[idx];
    output_octave[out_idx] = octave;
    output_class_id[out_idx] = level;

    // Compute orientation using gradient histogram
    float scale_f = 0.5f * output_size[out_idx] / octave_ratio;
    int scale = (int)(scale_f + 0.5f);
    float x0_f = refined_x / octave_ratio;
    int x0 = (int)(x0_f + (x0_f >= 0 ? 0.5f : -0.5f));
    float y0_f = refined_y / octave_ratio;
    int y0 = (int)(y0_f + (y0_f >= 0 ? 0.5f : -0.5f));

    float histogram[36];
    for (int b = 0; b < 36; b++)
        histogram[b] = 0.0f;

    const int radius = 6 * scale;
    const int radius_sq = radius * radius;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            if (dx * dx + dy * dy > radius_sq)
                continue;

            int sx = x0 + dx;
            int sy = y0 + dy;

            if (sx < 0 || sx >= cols || sy < 0 || sy >= rows)
                continue;

            int sidx = sy * cols + sx;
            float gx = Lx[sidx];
            float gy = Ly[sidx];

            float magnitude = sqrt(gx * gx + gy * gy);
            float angle = atan2(gy, gx);
            if (angle < 0.0f)
                angle += 2.0f * M_PI_F;

            float dist = sqrt((float)(dx * dx + dy * dy));
            float weight = exp(-dist * dist / (2.0f * scale * scale));

            int bin = (int)(angle * 36 / (2.0f * M_PI_F));
            bin = bin % 36;
            histogram[bin] += magnitude * weight;
        }
    }

    float max_hist = 0.0f;
    int max_bin = 0;
    for (int b = 0; b < 36; b++) {
        if (histogram[b] > max_hist) {
            max_hist = histogram[b];
            max_bin = b;
        }
    }

    if (max_hist == 0.0f)
    {
        output_angle[out_idx] = 0.0f;
        return;
    }

    int prev_bin = (max_bin - 1 + 36) % 36;
    int next_bin = (max_bin + 1) % 36;

    float prev_val = histogram[prev_bin];
    float curr_val = histogram[max_bin];
    float next_val = histogram[next_bin];

    float denom = prev_val - 2.0f * curr_val + next_val;
    float delta = (fabs(denom) > 1e-10f) ? 0.5f * (prev_val - next_val) / denom : 0.0f;
    float refined_angle = (max_bin + delta) * (2.0f * M_PI_F / 36);

    output_angle[out_idx] = refined_angle * 180.0f / M_PI_F;
}

/**
 * @brief Compute MLDB descriptor for a keypoint (single level, upright version)
 * @details Samples binary comparisons from gradient responses matching CPU implementation
 *
 * @param keypoints_x keypoint x coordinates
 * @param keypoints_y keypoint y coordinates
 * @param keypoints_size keypoint sizes
 * @param keypoints_angle keypoint orientations (unused in upright version)
 * @param num_keypoints number of keypoints
 * @param Lx gradient in x direction
 * @param Ly gradient in y direction
 * @param Lt flow response
 * @param rows image height
 * @param cols image width
 * @param octave_ratio octave ratio for this level
 * @param descriptor_size size of descriptor in bytes
 * @param output_descriptors output descriptor matrix
 */
__kernel void
AKAZE_compute_mldb_descriptor_level(__global const float* keypoints_x,
    __global const float* keypoints_y,
    __global const float* keypoints_size,
    __global const float* keypoints_angle,
    int num_keypoints,
    __global const float* Lx,
    __global const float* Ly,
    __global const float* Lt,
    int rows, int cols, float octave_ratio,
    int descriptor_size,
    __global uchar* output_descriptors)
{
    int i = get_global_id(0);

    if (i >= num_keypoints)
        return;

    float kpt_x = keypoints_x[i];
    float kpt_y = keypoints_y[i];
    float kpt_size = keypoints_size[i];

    // cvRound implementation: rounds to nearest, half away from zero
    float scale_f = 0.5f * kpt_size / octave_ratio;
    int scale = (int)(scale_f + (scale_f >= 0 ? 0.5f : -0.5f));

    float xf = kpt_x / octave_ratio;
    float yf = kpt_y / octave_ratio;

    // Pattern size (default 10)
    const int pattern_size = 10;

    // Sample steps for 3 grids: 2x2, 3x3, 4x4
    const int sample_step[3] = {
        pattern_size,
        (pattern_size * 2 + 2) / 3,  // divUp(pattern_size * 2, 3)
        pattern_size / 2
    };

    // Buffer for M-LDB descriptor values (max 16 cells * 3 channels)
    float values[48];

    __global uchar* desc = &output_descriptors[i * descriptor_size];

    // Initialize descriptor to zero
    for (int b = 0; b < descriptor_size; b++)
        desc[b] = 0;

    int dcount1 = 0;

    // For the three grids (2x2, 3x3, 4x4)
    for (int z = 0; z < 3; z++) {
        int dcount2 = 0;
        const int step = sample_step[z];

        for (int i = -pattern_size; i < pattern_size; i += step) {
            for (int j = -pattern_size; j < pattern_size; j += step) {
                float di = 0.0f, dx = 0.0f, dy = 0.0f;
                int nsamples = 0;

                // Sample within each cell
                for (int k = 0; k < step; k++) {
                    for (int l = 0; l < step; l++) {
                        // Get the coordinates of the sample point
                        const float sample_y = yf + (l + j) * scale;
                        const float sample_x = xf + (k + i) * scale;

                        // cvRound implementation for sampling coordinates
                        const int y1 = (int)(sample_y + (sample_y >= 0 ? 0.5f : -0.5f));
                        const int x1 = (int)(sample_x + (sample_x >= 0 ? 0.5f : -0.5f));

                        if (y1 < 0 || y1 >= rows || x1 < 0 || x1 >= cols)
                            continue; // Boundaries

                        const int idx = y1 * cols + x1;

                        const float ri = Lt[idx];
                        const float rx = Lx[idx];
                        const float ry = Ly[idx];

                        di += ri;
                        dx += rx;
                        dy += ry;
                        nsamples++;
                    }
                }

                if (nsamples > 0) {
                    const float nsamples_inv = 1.0f / nsamples;
                    di *= nsamples_inv;
                    dx *= nsamples_inv;
                    dy *= nsamples_inv;
                }

                // Store values (3 channels: Lt, Lx, Ly)
                values[dcount2 * 3] = di;
                values[dcount2 * 3 + 1] = dx;
                values[dcount2 * 3 + 2] = dy;
                dcount2++;
            }
        }

        // Do binary comparison for this grid
        const int num = (z + 2) * (z + 2);
        const int chan = 3;

        // Apply CV_TOGGLE_FLT to handle signed floats correctly
        // This toggles the sign bit to allow correct integer comparison of floats
        int* ivalues = (int*)values;
        for (int i = 0; i < num * chan; i++) {
            ivalues[i] = ivalues[i] ^ (ivalues[i] < 0 ? 0x7fffffff : 0);
        }

        // Match CPU comparison order: iterate Cell FIRST, then Channel
        // This produces: [Cell0-Ch0, Cell0-Ch1, Cell0-Ch2, Cell1-Ch0, Cell1-Ch1, ...]
        for (int i = 0; i < num; i++) {
            for (int j = i + 1; j < num; j++) {
                for (int pos = 0; pos < chan; pos++) {
                    int ival = ivalues[chan * i + pos];
                    if (ival > ivalues[chan * j + pos]) {
                        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
                    }
                    dcount1++;
                }
            }
        }
    }
}
