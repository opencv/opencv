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
