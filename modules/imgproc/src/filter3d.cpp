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
    CV_Assert(kernel.type() == CV_32F && (kernel.dims == 1 ||
              (kernel.dims == 2 && (kernel.rows == 1 || kernel.cols == 1))) &&
              kernel.total() % 2 == 1 && checkRange(kernel));
    return kernel.isContinuous() ? kernel : kernel.clone();
}

Mat slice(const Mat& volume, int z, int type)
{
    return Mat(volume.size[1], volume.size[2], type, volume.data + volume.step[0] * z,
               volume.step[1]);
}

void filterDepth16f(const Mat& src, Mat& dst, const Mat& kernel, const std::vector<int>& planes)
{
    CV_CPU_DISPATCH(filterDepth16f, (src, dst, kernel, planes), CV_CPU_DISPATCH_MODES_ALL);
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
    CV_Assert(count / 2 <= INT_MAX - src.size[0]);
    parallel_for_(Range(0, src.size[0]), [&](const Range& range)
    {
        Mat scratch;
        std::vector<int> planes(filterZ ? count : 0);
        for (int z = range.start; z < range.end; ++z)
        {
            Mat out = slice(dst, z, srcType);
            Mat work;
            if (!filterZ && depth == CV_32F)
                work = slice(src, z, srcType);
            else
            {
                if (!filterXY && depth == CV_32F)
                    work = out;
                else
                {
                    scratch.create(out.size(), workType);
                    work = scratch;
                }
                if (filterZ)
                {
                    for (int k = 0; k < count; ++k)
                        planes[k] = borderInterpolate(z + (k - count / 2), src.size[0], borderType);
                    if (depth == CV_8U)
                        cpu_baseline::filterDepth<uchar>(src, work, kernelZ, planes);
                    else if (depth == CV_16F)
                        filterDepth16f(src, work, kernelZ, planes);
                    else
                        cpu_baseline::filterDepth<float>(src, work, kernelZ, planes);
                }
                else
                    slice(src, z, srcType).convertTo(work, CV_32F);
            }

            if (filterXY)
            {
                Mat& filtered = depth == CV_32F ? out : work;
                sepFilter2D(work, filtered, CV_32F, kernelX, kernelY, Point(-1, -1), 0,
                            borderType | BORDER_ISOLATED);
            }

            if (depth != CV_32F)
                work.convertTo(out, depth);
        }
    });
}

} // namespace cv
