/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

__kernel
    void buildWarpPlaneMaps
    (
    __global float * map_x,
    __global float * map_y,
    __constant float * KRT,
    int tl_u,
    int tl_v,
    int cols,
    int rows,
    int step_x,
    int step_y,
    float scale
    )
{
    int du = get_global_id(0);
    int dv = get_global_id(1);
    step_x /= sizeof(float);
    step_y /= sizeof(float);

    __constant float * ck_rinv = KRT;
    __constant float * ct      = KRT + 9;

    if (du < cols && dv < rows)
    {
        float u = tl_u + du;
        float v = tl_v + dv;
        float x, y;

        float x_ = u / scale - ct[0];
        float y_ = v / scale - ct[1];

        float z;
        x = ck_rinv[0] * x_ + ck_rinv[1] * y_ + ck_rinv[2] * (1 - ct[2]);
        y = ck_rinv[3] * x_ + ck_rinv[4] * y_ + ck_rinv[5] * (1 - ct[2]);
        z = ck_rinv[6] * x_ + ck_rinv[7] * y_ + ck_rinv[8] * (1 - ct[2]);

        x /= z;
        y /= z;

        map_x[dv * step_x + du] = x;
        map_y[dv * step_y + du] = y;
    }
}

__kernel
    void buildWarpCylindricalMaps
    (
    __global float * map_x,
    __global float * map_y,
    __constant float * ck_rinv,
    int tl_u,
    int tl_v,
    int cols,
    int rows,
    int step_x,
    int step_y,
    float scale
    )
{
    int du = get_global_id(0);
    int dv = get_global_id(1);
    step_x /= sizeof(float);
    step_y /= sizeof(float);

    if (du < cols && dv < rows)
    {
        float u = tl_u + du;
        float v = tl_v + dv;
        float x, y;

        u /= scale;
        float x_ = sin(u);
        float y_ = v / scale;
        float z_ = cos(u);

        float z;
        x = ck_rinv[0] * x_ + ck_rinv[1] * y_ + ck_rinv[2] * z_;
        y = ck_rinv[3] * x_ + ck_rinv[4] * y_ + ck_rinv[5] * z_;
        z = ck_rinv[6] * x_ + ck_rinv[7] * y_ + ck_rinv[8] * z_;

        if (z > 0) { x /= z; y /= z; }
        else x = y = -1;

        map_x[dv * step_x + du] = x;
        map_y[dv * step_y + du] = y;
    }
}

__kernel
    void buildWarpSphericalMaps
    (
    __global float * map_x,
    __global float * map_y,
    __constant float * ck_rinv,
    int tl_u,
    int tl_v,
    int cols,
    int rows,
    int step_x,
    int step_y,
    float scale
    )
{
    int du = get_global_id(0);
    int dv = get_global_id(1);
    step_x /= sizeof(float);
    step_y /= sizeof(float);

    if (du < cols && dv < rows)
    {
        float u = tl_u + du;
        float v = tl_v + dv;
        float x, y;

        v /= scale;
        u /= scale;

        float sinv = sin(v);
        float x_ = sinv * sin(u);
        float y_ = - cos(v);
        float z_ = sinv * cos(u);

        float z;
        x = ck_rinv[0] * x_ + ck_rinv[1] * y_ + ck_rinv[2] * z_;
        y = ck_rinv[3] * x_ + ck_rinv[4] * y_ + ck_rinv[5] * z_;
        z = ck_rinv[6] * x_ + ck_rinv[7] * y_ + ck_rinv[8] * z_;

        if (z > 0) { x /= z; y /= z; }
        else x = y = -1;

        map_x[dv * step_x + du] = x;
        map_y[dv * step_y + du] = y;
    }
}

__kernel
    void buildWarpAffineMaps
    (
    __global float * xmap,
    __global float * ymap,
    __constant float * c_warpMat,
    int cols,
    int rows,
    int step_x,
    int step_y
    )
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    step_x /= sizeof(float);
    step_y /= sizeof(float);

    if (x < cols && y < rows)
    {
        const float xcoo = c_warpMat[0] * x + c_warpMat[1] * y + c_warpMat[2];
        const float ycoo = c_warpMat[3] * x + c_warpMat[4] * y + c_warpMat[5];

        map_x[y * step_x + x] = xcoo;
        map_y[y * step_y + x] = ycoo;
    }
}

__kernel
    void buildWarpPerspectiveMaps
    (
    __global float * xmap,
    __global float * ymap,
    __constant float * c_warpMat,
    int cols,
    int rows,
    int step_x,
    int step_y
    )
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    step_x /= sizeof(float);
    step_y /= sizeof(float);

    if (x < cols && y < rows)
    {
        const float coeff = 1.0f / (c_warpMat[6] * x + c_warpMat[7] * y + c_warpMat[8]);

        const float xcoo = coeff * (c_warpMat[0] * x + c_warpMat[1] * y + c_warpMat[2]);
        const float ycoo = coeff * (c_warpMat[3] * x + c_warpMat[4] * y + c_warpMat[5]);

        map_x[y * step_x + x] = xcoo;
        map_y[y * step_y + x] = ycoo;
    }
}
