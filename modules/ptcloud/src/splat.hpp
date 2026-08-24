// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef _CODERS_SPLAT_INTERNAL_H_
#define _CODERS_SPLAT_INTERNAL_H_

#include "precomp.hpp"

#include <opencv2/core/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace cv {

// 3D Gaussian Splatting internals shared by the loader, the decoders and the renderer.
// Deliberately free of OpenGL so it stays testable on headless builds.
namespace splat {

enum
{
    // Column order of getGaussianSplatPlyProperties(), which decodeGaussianSplats() reads.
    RAW_STRIDE = 14,
    RAW_OFS_POS = 0,
    RAW_OFS_DC = 3,
    RAW_OFS_OPACITY = 6,
    RAW_OFS_SCALE = 7,
    RAW_OFS_ROT = 10,

    // Byte layout of a ".splat" record, which decodeGaussianSplatsPacked() reads.
    PACKED_STRIDE = 32,
    PACKED_OFS_POS = 0,
    PACKED_OFS_SCALE = 12,
    PACKED_OFS_RGBA = 24,
    PACKED_OFS_ROT = 28
};

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

inline float shDcToColor(float dc)
{
    return std::min(1.0f, std::max(0.0f, 0.5f + 0.28209479177387814f * dc));
}

// Sigma = R S S^T R^T, symmetric positive semi-definite for any rotation and scale.
inline Matx33f covariance(const Vec3f& scale, const Vec4f& rot)
{
    Matx33f rs = Quatf(rot[0], rot[1], rot[2], rot[3]).toRotMat3x3(QUAT_ASSUME_NOT_UNIT);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            rs(i, j) *= scale[j];
    return rs * rs.t();
}

// Back to front order of the rows of a Nx3 position matrix, as alpha blending needs.
inline void sortByDepth(const Mat& pos, const Vec3f& cam, std::vector<int>& order)
{
    CV_Assert(pos.type() == CV_32F && pos.cols == 3);

    const int n = pos.rows;
    Mat key(1, n, CV_32F);
    float* k = key.ptr<float>();

    parallel_for_(Range(0, n), [&](const Range& range)
    {
        for (int i = range.start; i < range.end; i++)
        {
            const float* p = pos.ptr<float>(i);
            Vec3f d(p[0] - cam[0], p[1] - cam[1], p[2] - cam[2]);
            k[i] = d.dot(d);
        }
    });

    order.resize(n);
    Mat idx(1, n, CV_32S, order.data());
    sortIdx(key, idx, SORT_EVERY_ROW | SORT_DESCENDING);
}

} // namespace splat
} // namespace cv

#endif
