// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"

#include "splat.hpp"

namespace cv {

std::vector<String> getGaussianSplatPlyProperties()
{
    return {
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3"
    };
}

static void storeCovariance(float* d, const Matx33f& cov)
{
    d[GAUSSIAN_SPLAT_COV + 0] = cov(0, 0);
    d[GAUSSIAN_SPLAT_COV + 1] = cov(0, 1);
    d[GAUSSIAN_SPLAT_COV + 2] = cov(0, 2);
    d[GAUSSIAN_SPLAT_COV + 3] = cov(1, 1);
    d[GAUSSIAN_SPLAT_COV + 4] = cov(1, 2);
    d[GAUSSIAN_SPLAT_COV + 5] = cov(2, 2);
}

void decodeGaussianSplats(InputArray attributes, OutputArray splats)
{
    CV_TRACE_FUNCTION();

    Mat raw = attributes.getMat();
    CV_Assert(raw.dims == 2 && raw.channels() == 1);
    CV_Assert(raw.type() == CV_32F && raw.cols == splat::RAW_STRIDE);

    splats.create(raw.rows, GAUSSIAN_SPLAT_STRIDE, CV_32F);
    Mat dst = splats.getMat();

    parallel_for_(Range(0, raw.rows), [&](const Range& range)
    {
        for (int i = range.start; i < range.end; i++)
        {
            const float* s = raw.ptr<float>(i);
            float* d = dst.ptr<float>(i);

            for (int k = 0; k < 3; k++)
                d[GAUSSIAN_SPLAT_POS + k] = s[splat::RAW_OFS_POS + k];

            Vec3f scale(std::exp(s[splat::RAW_OFS_SCALE + 0]),
                        std::exp(s[splat::RAW_OFS_SCALE + 1]),
                        std::exp(s[splat::RAW_OFS_SCALE + 2]));
            storeCovariance(d, splat::covariance(scale,
                Vec4f(s[splat::RAW_OFS_ROT + 0], s[splat::RAW_OFS_ROT + 1],
                      s[splat::RAW_OFS_ROT + 2], s[splat::RAW_OFS_ROT + 3])));

            for (int k = 0; k < 3; k++)
                d[GAUSSIAN_SPLAT_RGB + k] = splat::shDcToColor(s[splat::RAW_OFS_DC + k]);

            d[GAUSSIAN_SPLAT_ALPHA] = splat::sigmoid(s[splat::RAW_OFS_OPACITY]);
        }
    });
}

void decodeGaussianSplatsPacked(InputArray buf, OutputArray splats)
{
    CV_TRACE_FUNCTION();

    Mat raw = buf.getMat();
    CV_Assert(raw.depth() == CV_8U && raw.isContinuous());

    const size_t stride = (size_t)splat::PACKED_STRIDE;
    const size_t bytes = raw.total() * raw.elemSize();
    CV_Assert(bytes > 0 && bytes % stride == 0);

    const uchar* data = raw.ptr<uchar>();
    const int n = (int)(bytes / stride);

    splats.create(n, GAUSSIAN_SPLAT_STRIDE, CV_32F);
    Mat dst = splats.getMat();

    // Values arrive already activated, so only the covariance is built.
    parallel_for_(Range(0, n), [&](const Range& range)
    {
        for (int i = range.start; i < range.end; i++)
        {
            const uchar* s = data + (size_t)i * splat::PACKED_STRIDE;
            float* d = dst.ptr<float>(i);

            Vec3f pos, scale;
            memcpy(pos.val, s + splat::PACKED_OFS_POS, sizeof(pos.val));
            memcpy(scale.val, s + splat::PACKED_OFS_SCALE, sizeof(scale.val));

            for (int k = 0; k < 3; k++)
                d[GAUSSIAN_SPLAT_POS + k] = pos[k];

            Vec4f rot;
            for (int k = 0; k < 4; k++)
                rot[k] = (s[splat::PACKED_OFS_ROT + k] - 128.f) / 128.f;
            if (rot.dot(rot) < 1e-12f)
                rot = Vec4f(1.f, 0.f, 0.f, 0.f);

            storeCovariance(d, splat::covariance(scale, rot));

            for (int k = 0; k < 3; k++)
                d[GAUSSIAN_SPLAT_RGB + k] = s[splat::PACKED_OFS_RGBA + k] / 255.f;

            d[GAUSSIAN_SPLAT_ALPHA] = s[splat::PACKED_OFS_RGBA + 3] / 255.f;
        }
    });
}

} // namespace cv
