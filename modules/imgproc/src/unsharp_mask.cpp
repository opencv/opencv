// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <cmath>
#include <type_traits>

namespace cv {

template<typename T>
static void unsharpMaskThresholdImpl(const Mat& src, const Mat& blurred, Mat& dst,
                                     double amount, double threshold)
{
    int channels = src.channels();
    int rows = src.rows;
    int cols = src.cols;

    parallel_for_(Range(0, rows), [&](const Range& range) {
        for (int r = range.start; r < range.end; ++r)
        {
            const T* s_ptr = src.ptr<T>(r);
            const T* b_ptr = blurred.ptr<T>(r);
            T* d_ptr = dst.ptr<T>(r);

            for (int c = 0; c < cols * channels; ++c)
            {
                double s_val = (double)s_ptr[c];
                double b_val = (double)b_ptr[c];
                double diff = s_val - b_val;

                if (std::abs(diff) >= threshold)
                {
                    double val = s_val + amount * diff;
                    if (std::is_integral<T>::value)
                        d_ptr[c] = saturate_cast<T>(val + (val >= 0 ? 0.5 : -0.5));
                    else
                        d_ptr[c] = saturate_cast<T>(val);
                }
                else
                {
                    d_ptr[c] = s_ptr[c];
                }
            }
        }
    });
}

void unsharpMask( InputArray _src, OutputArray _dst,
                   double sigma, double amount, double threshold,
                   int borderType )
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat();
    CV_Assert( !src.empty() );
    CV_Assert( sigma > 0 );

    Mat blurred;
    GaussianBlur(src, blurred, Size(0, 0), sigma, sigma, borderType);

    if (threshold <= 0)
    {
        addWeighted(src, 1.0 + amount, blurred, -amount, 0, _dst);
        return;
    }

    _dst.create(src.size(), src.type());
    Mat dst = _dst.getMat();

    int depth = src.depth();
    switch (depth)
    {
    case CV_8U:
        unsharpMaskThresholdImpl<uchar>(src, blurred, dst, amount, threshold);
        break;
    case CV_8S:
        unsharpMaskThresholdImpl<schar>(src, blurred, dst, amount, threshold);
        break;
    case CV_16U:
        unsharpMaskThresholdImpl<ushort>(src, blurred, dst, amount, threshold);
        break;
    case CV_16S:
        unsharpMaskThresholdImpl<short>(src, blurred, dst, amount, threshold);
        break;
    case CV_32S:
        unsharpMaskThresholdImpl<int>(src, blurred, dst, amount, threshold);
        break;
    case CV_32F:
        unsharpMaskThresholdImpl<float>(src, blurred, dst, amount, threshold);
        break;
    case CV_64F:
        unsharpMaskThresholdImpl<double>(src, blurred, dst, amount, threshold);
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "Unsupported depth for unsharpMask");
    }
}

} // namespace cv
