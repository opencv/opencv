// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>

namespace cv {

struct MapItem {
    int idx0, idx1;
    float w0, w1;
};

static inline MapItem computeLinearMapItem(int out_idx, int in_len, double inv_scale)
{
    MapItem item;
    if (in_len == 1)
    {
        item.idx0 = 0;
        item.idx1 = 0;
        item.w0 = 1.0f;
        item.w1 = 0.0f;
        return item;
    }
    double f_src = (out_idx + 0.5) * inv_scale - 0.5;
    int i0 = cvFloor(f_src);
    float w1 = (float)(f_src - i0);
    float w0 = 1.0f - w1;

    if (i0 < 0)
    {
        item.idx0 = 0;
        item.idx1 = 0;
        item.w0 = 1.0f;
        item.w1 = 0.0f;
    }
    else if (i0 >= in_len - 1)
    {
        item.idx0 = in_len - 1;
        item.idx1 = in_len - 1;
        item.w0 = 1.0f;
        item.w1 = 0.0f;
    }
    else
    {
        item.idx0 = i0;
        item.idx1 = i0 + 1;
        item.w0 = w0;
        item.w1 = w1;
    }
    return item;
}

template<typename T>
static void resize3DLinearImpl(const Mat& src, Mat& dst,
                               int out_d, int out_h, int out_w, int cn,
                               const std::vector<MapItem>& z_map,
                               const std::vector<MapItem>& y_map,
                               const std::vector<MapItem>& x_map)
{
    size_t step0 = src.step[0];
    size_t step1 = src.step[1];
    size_t step2 = src.step[2];

    size_t dstep0 = dst.step[0];
    size_t dstep1 = dst.step[1];
    size_t dstep2 = dst.step[2];

    parallel_for_(Range(0, out_d), [&](const Range& range) {
        for (int z = range.start; z < range.end; ++z)
        {
            const MapItem& zm = z_map[z];
            const uchar* src_z0 = src.data + zm.idx0 * step0;
            const uchar* src_z1 = src.data + zm.idx1 * step0;
            uchar* dst_z = dst.data + z * dstep0;

            float wz0 = zm.w0;
            float wz1 = zm.w1;

            for (int y = 0; y < out_h; ++y)
            {
                const MapItem& ym = y_map[y];
                const uchar* src_z0y0 = src_z0 + ym.idx0 * step1;
                const uchar* src_z0y1 = src_z0 + ym.idx1 * step1;
                const uchar* src_z1y0 = src_z1 + ym.idx0 * step1;
                const uchar* src_z1y1 = src_z1 + ym.idx1 * step1;

                uchar* dst_zy = dst_z + y * dstep1;

                float wy0 = ym.w0;
                float wy1 = ym.w1;

                float w00 = wz0 * wy0;
                float w01 = wz0 * wy1;
                float w10 = wz1 * wy0;
                float w11 = wz1 * wy1;

                for (int x = 0; x < out_w; ++x)
                {
                    const MapItem& xm = x_map[x];
                    size_t x0_byte = xm.idx0 * step2;
                    size_t x1_byte = xm.idx1 * step2;

                    const T* p000 = (const T*)(src_z0y0 + x0_byte);
                    const T* p001 = (const T*)(src_z0y0 + x1_byte);
                    const T* p010 = (const T*)(src_z0y1 + x0_byte);
                    const T* p011 = (const T*)(src_z0y1 + x1_byte);
                    const T* p100 = (const T*)(src_z1y0 + x0_byte);
                    const T* p101 = (const T*)(src_z1y0 + x1_byte);
                    const T* p110 = (const T*)(src_z1y1 + x0_byte);
                    const T* p111 = (const T*)(src_z1y1 + x1_byte);

                    T* dst_pixel = (T*)(dst_zy + x * dstep2);

                    float wx0 = xm.w0;
                    float wx1 = xm.w1;

                    float w000 = w00 * wx0;
                    float w001 = w00 * wx1;
                    float w010 = w01 * wx0;
                    float w011 = w01 * wx1;
                    float w100 = w10 * wx0;
                    float w101 = w10 * wx1;
                    float w110 = w11 * wx0;
                    float w111 = w11 * wx1;

                    for (int c = 0; c < cn; ++c)
                    {
                        float val = (float)p000[c] * w000 + (float)p001[c] * w001 +
                                    (float)p010[c] * w010 + (float)p011[c] * w011 +
                                    (float)p100[c] * w100 + (float)p101[c] * w101 +
                                    (float)p110[c] * w110 + (float)p111[c] * w111;

                        if (std::is_integral<T>::value)
                            dst_pixel[c] = saturate_cast<T>(val + (val >= 0 ? 0.5f : -0.5f));
                        else
                            dst_pixel[c] = saturate_cast<T>(val);
                    }
                }
            }
        }
    });
}

void resize3D( InputArray _src, OutputArray _dst,
               Vec3i dsize, double fx, double fy, double fz,
               int interpolation )
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat();
    CV_Assert( !src.empty() );
    CV_Assert( src.dims == 3 || src.dims == 4 );
    CV_Assert( interpolation == INTER_NEAREST || interpolation == INTER_LINEAR );

    int in_d = 0, in_h = 0, in_w = 0, cn = 0;
    bool is_4d = (src.dims == 4);

    if (is_4d)
    {
        in_d = src.size[0];
        in_h = src.size[1];
        in_w = src.size[2];
        cn = src.size[3] * src.channels();
    }
    else // dims == 3
    {
        in_d = src.size[0];
        in_h = src.size[1];
        in_w = src.size[2];
        cn = src.channels();
    }

    CV_Assert( in_d > 0 && in_h > 0 && in_w > 0 && cn > 0 );

    int out_d = dsize[0];
    int out_h = dsize[1];
    int out_w = dsize[2];

    if (out_d <= 0 || out_h <= 0 || out_w <= 0)
    {
        CV_Assert( fx > 0 && fy > 0 && fz > 0 );
        out_d = saturate_cast<int>(std::round(in_d * fz));
        out_h = saturate_cast<int>(std::round(in_h * fy));
        out_w = saturate_cast<int>(std::round(in_w * fx));
        CV_Assert( out_d > 0 && out_h > 0 && out_w > 0 );
    }
    else
    {
        if (fz == 0) fz = (double)out_d / in_d;
        if (fy == 0) fy = (double)out_h / in_h;
        if (fx == 0) fx = (double)out_w / in_w;
    }

    if (is_4d)
    {
        int out_sizes[4] = { out_d, out_h, out_w, src.size[3] };
        _dst.create(4, out_sizes, src.type());
    }
    else
    {
        int out_sizes[3] = { out_d, out_h, out_w };
        _dst.create(3, out_sizes, src.type());
    }

    Mat dst = _dst.getMat();

    if (in_d == out_d && in_h == out_h && in_w == out_w)
    {
        src.copyTo(dst);
        return;
    }

    double inv_scale_z = (double)in_d / out_d;
    double inv_scale_y = (double)in_h / out_h;
    double inv_scale_x = (double)in_w / out_w;

    if (interpolation == INTER_NEAREST)
    {
        size_t voxel_bytes = cn * src.elemSize1();

        std::vector<int> x_map(out_w);
        for (int x = 0; x < out_w; ++x)
        {
            x_map[x] = std::min(cvFloor(x * inv_scale_x), in_w - 1);
        }

        std::vector<int> y_map(out_h);
        for (int y = 0; y < out_h; ++y)
        {
            y_map[y] = std::min(cvFloor(y * inv_scale_y), in_h - 1);
        }

        parallel_for_(Range(0, out_d), [&](const Range& range) {
            for (int z = range.start; z < range.end; ++z)
            {
                int sz = std::min(cvFloor(z * inv_scale_z), in_d - 1);
                const uchar* src_z = src.data + sz * src.step[0];
                uchar* dst_z = dst.data + z * dst.step[0];

                for (int y = 0; y < out_h; ++y)
                {
                    int sy = y_map[y];
                    const uchar* src_zy = src_z + sy * src.step[1];
                    uchar* dst_zy = dst_z + y * dst.step[1];

                    for (int x = 0; x < out_w; ++x)
                    {
                        int sx = x_map[x];
                        const uchar* src_pixel = src_zy + sx * src.step[2];
                        uchar* dst_pixel = dst_zy + x * dst.step[2];

                        memcpy(dst_pixel, src_pixel, voxel_bytes);
                    }
                }
            }
        });
        return;
    }

    // INTER_LINEAR (Trilinear)
    std::vector<MapItem> z_map(out_d);
    for (int z = 0; z < out_d; ++z)
        z_map[z] = computeLinearMapItem(z, in_d, inv_scale_z);

    std::vector<MapItem> y_map(out_h);
    for (int y = 0; y < out_h; ++y)
        y_map[y] = computeLinearMapItem(y, in_h, inv_scale_y);

    std::vector<MapItem> x_map(out_w);
    for (int x = 0; x < out_w; ++x)
        x_map[x] = computeLinearMapItem(x, in_w, inv_scale_x);

    int depth = src.depth();
    switch (depth)
    {
    case CV_8U:
        resize3DLinearImpl<uchar>(src, dst, out_d, out_h, out_w, cn, z_map, y_map, x_map);
        break;
    case CV_8S:
        resize3DLinearImpl<schar>(src, dst, out_d, out_h, out_w, cn, z_map, y_map, x_map);
        break;
    case CV_16U:
        resize3DLinearImpl<ushort>(src, dst, out_d, out_h, out_w, cn, z_map, y_map, x_map);
        break;
    case CV_16S:
        resize3DLinearImpl<short>(src, dst, out_d, out_h, out_w, cn, z_map, y_map, x_map);
        break;
    case CV_32S:
        resize3DLinearImpl<int>(src, dst, out_d, out_h, out_w, cn, z_map, y_map, x_map);
        break;
    case CV_32F:
        resize3DLinearImpl<float>(src, dst, out_d, out_h, out_w, cn, z_map, y_map, x_map);
        break;
    case CV_64F:
        resize3DLinearImpl<double>(src, dst, out_d, out_h, out_w, cn, z_map, y_map, x_map);
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "Unsupported depth for resize3D");
    }
}

} // namespace cv
