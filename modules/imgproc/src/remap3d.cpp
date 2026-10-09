// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <cmath>

namespace cv {
namespace {

template<typename T> class Remap3DInvoker : public ParallelLoopBody
{
public:
    Remap3DInvoker(const Mat& src, Mat& dst, const Mat& map, int interpolation,
                   int borderMode, const Scalar& borderValue)
        : src_(src), dst_(dst), map_(map), interpolation_(interpolation), borderMode_(borderMode)
    {
        for (int c = 0; c < 4; ++c)
            border_[c] = saturate_cast<T>(borderValue[c]);
    }

    void operator()(const Range& range) const CV_OVERRIDE
    {
        const int cn = src_.channels();
        const int bounds[] = {src_.size[2], src_.size[1], src_.size[0]};
        for (int row = range.start; row < range.end; ++row)
        {
            const int z = row / dst_.size[1], y = row % dst_.size[1];
            const Vec3f* mapRow = map_.ptr<Vec3f>(z, y);
            T* out = dst_.ptr<T>(z, y);
            for (int x = 0; x < dst_.size[2]; ++x, out += cn)
            {
                double p[3];
                bool outside = false;
                for (int axis = 0; axis < 3; ++axis)
                {
                    p[axis] = mapRow[x][axis];
                    CV_Assert(std::isfinite(p[axis]));
                    if (borderMode_ == BORDER_REPLICATE)
                        p[axis] = std::max(0.0, std::min(p[axis], double(bounds[axis] - 1)));
                    if (interpolation_ == INTER_NEAREST)
                        p[axis] = std::floor(p[axis] + 0.5);
                    // Check in floating point before converting potentially huge coordinates.
                    outside |= p[axis] <= -1.0 || p[axis] >= bounds[axis];
                }
                if (outside)
                {
                    for (int c = 0; c < cn; ++c)
                        out[c] = border_[c];
                    continue;
                }
                if (interpolation_ == INTER_NEAREST)
                {
                    const T* in = src_.ptr<T>(int(p[2]), int(p[1])) + size_t(p[0]) * cn;
                    for (int c = 0; c < cn; ++c)
                        out[c] = in[c];
                    continue;
                }

                int index[3][2];
                double weight[3][2];
                for (int axis = 0; axis < 3; ++axis)
                {
                    const int base = int(std::floor(p[axis]));
                    weight[axis][1] = p[axis] - base;
                    weight[axis][0] = 1.0 - weight[axis][1];
                    for (int i = 0; i < 2; ++i)
                        index[axis][i] = borderMode_ == BORDER_REPLICATE
                            ? std::min(base + i, bounds[axis] - 1) : base + i;
                }
                double value[4] = {};
                for (int dz = 0; dz < 2; ++dz)
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx)
                        {
                            const double w = weight[0][dx] * weight[1][dy] * weight[2][dz];
                            if (w == 0.0)
                                continue;
                            const int sx = index[0][dx], sy = index[1][dy], sz = index[2][dz];
                            const bool valid = (unsigned)sx < (unsigned)bounds[0] &&
                                               (unsigned)sy < (unsigned)bounds[1] &&
                                               (unsigned)sz < (unsigned)bounds[2];
                            const T* in = valid ? src_.ptr<T>(sz, sy) + size_t(sx) * cn : border_;
                            for (int c = 0; c < cn; ++c)
                                value[c] += w * in[c];
                        }
                for (int c = 0; c < cn; ++c)
                    out[c] = saturate_cast<T>(value[c]);
            }
        }
    }

private:
    const Mat& src_;
    Mat& dst_;
    const Mat& map_;
    int interpolation_, borderMode_;
    T border_[4];
};

} // namespace

void remap3D(InputArray _src, OutputArray _dst, InputArray _map,
             int interpolation, int borderMode, const Scalar& borderValue)
{
    CV_INSTRUMENT_REGION();
    const Mat src = _src.getMat(), map = _map.getMat();
    CV_Assert(!src.empty() && src.dims == 3);
    CV_Assert((src.depth() == CV_8U || src.depth() == CV_32F) && src.channels() <= 4);
    CV_Assert(!map.empty() && map.dims == 3 && map.type() == CV_32FC3);
    CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR);
    CV_Assert(borderMode == BORDER_CONSTANT || borderMode == BORDER_REPLICATE);
    CV_Assert(map.size[0] <= INT_MAX / map.size[1]);

    _dst.create(map.size, src.type());
    Mat dst = _dst.getMat();
    CV_Assert(dst.datastart != src.datastart && dst.datastart != map.datastart);
    const Range rows(0, map.size[0] * map.size[1]);
    if (src.depth() == CV_8U)
        parallel_for_(rows, Remap3DInvoker<uchar>(src, dst, map, interpolation, borderMode, borderValue));
    else
        parallel_for_(rows, Remap3DInvoker<float>(src, dst, map, interpolation, borderMode, borderValue));
}

} // namespace cv
