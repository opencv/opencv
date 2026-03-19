#include "precomp.hpp"
#include <cmath>

namespace {

template <typename T>
void calculateCrossPowerSpectrum(const cv::Mat& dft1, const cv::Mat& dft2, cv::Mat& cps)
{
    for (int row = 0; row < dft1.rows; ++row)
    {
        auto* cpsp = cps.ptr<cv::Vec<T, 2>>(row);
        const auto* dft1p = dft1.ptr<cv::Vec<T, 2>>(row);
        const auto* dft2p = dft2.ptr<cv::Vec<T, 2>>(row);
        for (int col = 0; col < dft1.cols; ++col)
        {
            const T re = dft1p[col][0] * dft2p[col][0] + dft1p[col][1] * dft2p[col][1];
            const T im = dft1p[col][0] * dft2p[col][1] - dft1p[col][1] * dft2p[col][0];
            const T mag = std::sqrt(re * re + im * im);
            cpsp[col][0] = re / mag;
            cpsp[col][1] = im / mag;
        }
    }
}

cv::Mat calculateCrossPowerSpectrum(const cv::Mat& dft1, const cv::Mat& dft2)
{
    cv::Mat cps(dft1.rows, dft1.cols, dft1.type());
    if (dft1.type() == CV_32FC2)
        calculateCrossPowerSpectrum<float>(dft1, dft2, cps);
    else if (dft1.type() == CV_64FC2)
        calculateCrossPowerSpectrum<double>(dft1, dft2, cps);
    else
        CV_Error(cv::Error::StsNotImplemented, "Only CV_32FC2 and CV_64FC2 types are supported");
    return cps;
}

void fftshift(cv::Mat& out)
{
    int cx = out.cols / 2;
    int cy = out.rows / 2;
    cv::Mat q0(out, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(out, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(out, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(out, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

bool isOutOfBounds(const cv::Point2i& peak, const cv::Mat& mat, int size)
{
    return peak.x - size / 2 < 0 || peak.y - size / 2 < 0 || peak.x + size / 2 >= mat.cols ||
           peak.y + size / 2 >= mat.rows;
}

bool reduceL2size(int& L2size)
{
    L2size -= 2;
    return L2size >= 3;
}

int getL1size(int L2Usize, double L1ratio)
{
    int L1size = static_cast<int>(std::floor(L1ratio * L2Usize));
    return (L1size % 2) ? L1size : L1size + 1;
}

cv::Point2d getPeakSubpixel(const cv::Mat& mat)
{
    const auto m = moments(mat);
    return cv::Point2d(m.m10 / m.m00, m.m01 / m.m00);
}

bool accuracyReached(const cv::Point2d& L1peak, const cv::Point2d& L1mid)
{
    return std::abs(L1peak.x - L1mid.x) < 0.5 && std::abs(L1peak.y - L1mid.y) < 0.5;
}

cv::Point2d getSubpixelShift(const cv::Mat& L3,
                             const cv::Point2d& L3peak,
                             const cv::Point2d& L3mid,
                             int L2size)
{
    while (isOutOfBounds(L3peak, L3, L2size))
        if (!reduceL2size(L2size))
            return L3peak - L3mid;

    cv::Mat L2 = L3(cv::Rect(static_cast<int>(L3peak.x - L2size / 2),
                             static_cast<int>(L3peak.y - L2size / 2),
                             L2size,
                             L2size));
    cv::Point2d L2peak = getPeakSubpixel(L2);
    cv::Point2d L2mid(L2.cols / 2, L2.rows / 2);
    return L3peak - L3mid + L2peak - L2mid;
}

}  // namespace

cv::Point2d
    cv::phaseCorrelateIterative(InputArray _src1, InputArray _src2, int L2size, int maxIters)
{
    CV_INSTRUMENT_REGION();

    Mat src1 = _src1.getMat();
    Mat src2 = _src2.getMat();

    CV_Assert(src1.type() == src2.type());
    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.type() == CV_32FC1 || src1.type() == CV_64FC1);

    // apply DFT window to input images
    Mat window;
    createHanningWindow(window, src1.size(), _src1.type());
    Mat image1, image2;
    multiply(_src1, window, image1);
    multiply(_src2, window, image2);

    // compute the DFTs of input images
    dft(image1, image1, DFT_COMPLEX_OUTPUT);
    dft(image2, image2, DFT_COMPLEX_OUTPUT);

    // compute the phase correlation landscape L3
    Mat L3 = calculateCrossPowerSpectrum(image1, image2);
    dft(L3, L3, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT);
    fftshift(L3);
    Point2d L3mid(L3.cols / 2, L3.rows / 2);

    // calculate the maximum correlation location
    Point2i L3peak;
    minMaxLoc(L3, nullptr, nullptr, nullptr, &L3peak);

    // reduce the L2size as long as L2 is out of bounds of L3
    while (isOutOfBounds(L3peak, L3, L2size))
        if (!reduceL2size(L2size))
            return Point2d(L3peak) - L3mid;

    // extract the L2 maximum correlation neighborhood from L3
    Mat L2 = L3(Rect(L3peak.x - L2size / 2, L3peak.y - L2size / 2, L2size, L2size));

    // upsample L2 maximum correlation neighborhood to get L2U
    Mat L2U;
    const int L2Usize = 223;  // empirically determined optimal constant
    resize(L2, L2U, {L2Usize, L2Usize}, 0, 0, INTER_LINEAR);
    const Point2d L2Umid(L2U.cols / 2, L2U.rows / 2);

    // run the iterative refinement algorithm using the specified L1 ratio
    // gradually decrease L1 ratio if convergence is not achieved
    const double L1ratioBase = 0.45;  // empirically determined optimal constant
    const double L1ratioStep = 0.025;
    for (double L1ratio = L1ratioBase; getL1size(L2U.cols, L1ratio) > 0; L1ratio -= L1ratioStep)
    {
        Point2d L2Upeak = L2Umid;  // reset the accumulated L2U peak position
        const int L1size = getL1size(L2U.cols, L1ratio);  // calculate the current L1 size
        const Point2d L1mid(L1size / 2, L1size / 2);  // update the L1 mid position

        // perform the iterative refinement algorithm
        for (int iter = 0; iter < maxIters; ++iter)
        {
            // verify that the L1 region is within the L2U region
            if (isOutOfBounds(L2Upeak, L2U, L1size))
                break;

            // extract the L1 region from L2U
            const Mat L1 = L2U(Rect(static_cast<int>(L2Upeak.x - L1size / 2),
                                    static_cast<int>(L2Upeak.y - L1size / 2),
                                    L1size,
                                    L1size));

            // calculate the centroid location
            const Point2d L1peak = getPeakSubpixel(L1);

            // add the contribution of the current iteration to the accumulated L2U peak location
            L2Upeak += Point2d(std::round(L1peak.x - L1mid.x), std::round(L1peak.y - L1mid.y));

            // check for convergence
            if (accuracyReached(L1peak, L1mid))
                // return the refined subpixel image shift
                return Point2d(L3peak) - L3mid +
                       (L2Upeak - L2Umid + L1peak - L1mid) /
                           (static_cast<double>(L2Usize) / L2size);
        }
    }

    // iterative refinement failed to converge, return non-iterative subpixel shift
    return getSubpixelShift(L3, Point2d(L3peak), L3mid, L2size);
}
