// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

__kernel void preCalculationPixNorm (__global char * pixNormsPtr,
                                     int pixNormsStep, int pixNormsOffset,
                                     int pixNormsRows, int pixNormsCols,
                                     const __global float * xx,
                                     const __global float * yy)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < pixNormsRows && j < pixNormsCols)
    {
        *(__global float*)(pixNormsPtr + pixNormsOffset + i*pixNormsStep + j*sizeof(float)) = sqrt(xx[j] * xx[j] + yy[i] * yy[i] + 1.0f);
    }
}