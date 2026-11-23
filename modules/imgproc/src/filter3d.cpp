// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at https://opencv.org/license.html.

#include "precomp.hpp"
#include "filter3d.simd.hpp"
#include "filter3d.simd_declarations.hpp"

namespace cv {
namespace {

Mat prepareKernel(InputArray input)
{
    Mat kernel = input.getMat();
    CV_Assert(kernel.channels() == 1 && (kernel.dims == 1 ||
              (kernel.dims == 2 && (kernel.rows == 1 || kernel.cols == 1))) &&
              kernel.total() % 2 == 1);
    kernel.convertTo(kernel, CV_32F);
    CV_Assert(checkRange(kernel));
    return kernel.isContinuous() ? kernel : kernel.clone();
}

Mat slice(const Mat& volume, int z, int type)
{
    return Mat(volume.size[1], volume.size[2], type, volume.data + volume.step[0] * z,
               volume.step[1]);
}

void filterDepth(const Mat& src, Mat& dst, const Mat& kernel, const std::vector<int>& planes,
                 int firstRow)
{
    CV_CPU_DISPATCH(filterDepth, (src, dst, kernel, planes, firstRow), CV_CPU_DISPATCH_MODES_ALL);
}

bool isIdentity(const Mat& kernel)
{
    return kernel.total() == 1 && kernel.ptr<float>()[0] == 1.f;
}

} // namespace

void sepFilter3D(InputArray _src, OutputArray _dst,
                 InputArray _kernelZ, InputArray _kernelY, InputArray _kernelX, int borderType)
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat();
    const int depth = src.depth();
    CV_Assert(!src.empty() && (src.dims == 3 || (src.dims == 4 && src.channels() == 1)));
    CV_Assert(depth == CV_8U || depth == CV_16F || depth == CV_32F);
    borderType &= ~BORDER_ISOLATED;
    CV_Assert(borderType == BORDER_CONSTANT || borderType == BORDER_REPLICATE ||
              borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101);

    const int cn = src.dims == 3 ? src.channels() : src.size[3];
    CV_Assert(cn <= CV_CN_MAX && src.size[2] <= INT_MAX / cn);
    const int srcType = CV_MAKETYPE(depth, cn);
    const size_t pixelSize = CV_ELEM_SIZE(srcType);
    const int workType = CV_MAKETYPE(CV_32F, cn);
    const Mat kernelZ = prepareKernel(_kernelZ);
    const Mat kernelY = prepareKernel(_kernelY);
    const Mat kernelX = prepareKernel(_kernelX);
    const bool filterZ = !isIdentity(kernelZ);
    const bool filterXY = !isIdentity(kernelY) || !isIdentity(kernelX);

    _dst.create(src.size, src.type());
    Mat dst = _dst.getMat();
    CV_Assert(dst.step[2] == pixelSize);
    if (src.step[2] != pixelSize || (src.data < dst.dataend && dst.data < src.dataend))
        src = src.clone();
    if (!filterZ && !filterXY)
    {
        src.copyTo(dst);
        return;
    }

    const int count = static_cast<int>(kernelZ.total());
    const int planeCount = src.size[0], rows = src.size[1], cols = src.size[2];
    CV_Assert(count / 2 <= INT_MAX - planeCount);
    // Split planes into row stripes so volumes with fewer planes than threads keep all threads busy.
    // Each stripe also filters the halo rows that the height kernel reads from its neighbours.
    const int halo = filterXY ? static_cast<int>(kernelY.total()) / 2 : 0;
    const int stripes = std::max(1, std::min(divUp(getNumThreads(), planeCount),
                                             rows / std::max(16, 2 * halo)));
    const int scratchRows = std::min(rows, divUp(rows, stripes) + 2 * halo);
    parallel_for_(Range(0, planeCount * stripes), [&](const Range& range)
    {
        Mat scratch;
        std::vector<int> planes(filterZ ? count : 0);
        for (int task = range.start; task < range.end; ++task)
        {
            const int z = task / stripes, stripe = task % stripes;
            const int y0 = static_cast<int>(static_cast<int64>(stripe) * rows / stripes);
            const int y1 = static_cast<int>(static_cast<int64>(stripe + 1) * rows / stripes);
            const int haloTop = std::max(0, y0 - halo), haloBottom = std::min(rows, y1 + halo);
            Mat out = slice(dst, z, srcType).rowRange(y0, y1);
            // Rows [y0, y1) of the float32 plane. sepFilter2D reads the rows around this ROI
            // from its parent and extrapolates only at the parent's edges.
            Mat work;
            if (!filterZ && depth == CV_32F)
                work = slice(src, z, srcType).rowRange(y0, y1);
            else
            {
                Mat buffer;
                if (!filterXY && depth == CV_32F)
                    buffer = out;
                else
                {
                    scratch.create(scratchRows, cols, workType);
                    // A new header, so that the ROI parent is exactly rows [haloTop, haloBottom).
                    buffer = Mat(haloBottom - haloTop, cols, workType, scratch.data);
                }
                if (filterZ)
                {
                    for (int k = 0; k < count; ++k)
                        planes[k] = borderInterpolate(z + (k - count / 2), planeCount, borderType);
                    filterDepth(src, buffer, kernelZ, planes, haloTop);
                }
                else
                    slice(src, z, srcType).rowRange(haloTop, haloBottom).convertTo(buffer, CV_32F);
                work = buffer.rowRange(y0 - haloTop, y1 - haloTop);
            }

            if (filterXY)
            {
                Mat filtered = depth == CV_32F ? out : work;
                sepFilter2D(work, filtered, CV_32F, kernelX, kernelY, Point(-1, -1), 0, borderType);
            }

            if (depth != CV_32F)
                work.convertTo(out, depth);
        }
    });
}

} // namespace cv
