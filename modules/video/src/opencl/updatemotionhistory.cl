// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

__kernel void updateMotionHistory(__global const uchar * silh, int silh_step, int silh_offset,
                                  __global uchar * mhiptr, int mhi_step, int mhi_offset, int mhi_rows, int mhi_cols,
                                  float timestamp, float delbound)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < mhi_cols && y < mhi_rows)
    {
        int silh_index = mad24(y, silh_step, silh_offset + x);
        int mhi_index = mad24(y, mhi_step, mhi_offset + x * (int)sizeof(float));

        silh += silh_index;
        __global float * mhi = (__global float *)(mhiptr + mhi_index);

        float val = mhi[0];
        val = silh[0] ? timestamp : val < delbound ? 0 : val;
        mhi[0] = val;
    }
}
