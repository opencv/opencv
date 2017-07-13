// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


/**
 * @brief This function computes the Perona and Malik conductivity coefficient g2
 * g2 = 1 / (1 + dL^2 / k^2)
 * @param Lx First order image derivative in X-direction (horizontal)
 * @param Ly First order image derivative in Y-direction (vertical)
 * @param dst Output image
 * @param k Contrast factor parameter
 */
__kernel void
AKAZE_pm_g2(__global const float* lx, __global const float* ly, __global float* dst, float k)
{
    const float k2inv = 1.0f / (k * k);
    int i = get_global_id(0);
    dst[i] = 1.0f / (1.0f + ((lx[i] * lx[i] + ly[i] * ly[i]) * k2inv));
}

__kernel void
AKAZE_nld_step_scalar(__global const float* lt, int lt_step, int lt_offset,
    __global const float* lf, __global float* dst, float step_size)
{
    /* The labeling scheme for this five star stencil:
        [    a    ]
        [ -1 c +1 ]
        [    b    ]
    */
    int i = get_global_id(0);
    int j = get_global_id(1);
    size_t rows = get_global_size(0);
    size_t cols = get_global_size(1);

    // get row indexes
    int a = (i - 1) * lt_step;
    int c = (i    ) * lt_step;
    int b = (i + 1) * lt_step;
    int dst_idx = c;
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

    dst[dst_idx] = res * step_size;
}
