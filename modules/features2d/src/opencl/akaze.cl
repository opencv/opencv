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
