#include "precomp.hpp"
#include <cmath>

namespace cv {

Mat CalculateCrossPowerSpectrum(Mat dft1, Mat dft2) {
  // TODO: double precision
  Mat cps(dft1.rows, dft1.cols, CV_32FC2);
  for (int row = 0; row < dft1.rows; ++row) {
    auto cpsp = cps.ptr<Vec<float, 2>>(row);
    auto dft1p = dft1.ptr<Vec<float, 2>>(row);
    const auto dft2p = dft2.ptr<Vec<float, 2>>(row);
    for (int col = 0; col < dft1.cols; ++col) {
      const float re =
          dft1p[col][0] * dft2p[col][0] + dft1p[col][1] * dft2p[col][1];
      const float im =
          dft1p[col][0] * dft2p[col][1] - dft1p[col][1] * dft2p[col][0];
      const float mag = std::sqrt(re * re + im * im);
      cpsp[col][0] = re / mag;
      cpsp[col][1] = im / mag;
    }
  }
  return cps;
}

void fftshift(InputOutputArray _out) {
  Mat out = _out.getMat();
  int cx = out.cols / 2;
  int cy = out.rows / 2;
  Mat q0(out, Rect(0, 0, cx, cy));
  Mat q1(out, Rect(cx, 0, cx, cy));
  Mat q2(out, Rect(0, cy, cx, cy));
  Mat q3(out, Rect(cx, cy, cx, cy));

  Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);
}

bool IsOutOfBounds(const Point2i &peak, const Mat &mat, int size) {
  return peak.x - size / 2 < 0 || peak.y - size / 2 < 0 ||
         peak.x + size / 2 >= mat.cols || peak.y + size / 2 >= mat.rows;
}

bool ReduceL2size(int &L2size) {
  L2size -= 2;
  return L2size >= 3;
}

int GetL1size(int L2Usize, double L1ratio) {
  int L1size = std::floor(L1ratio * L2Usize);
  return (L1size % 2) ? L1size : L1size + 1;
}

Point2d GetPeakSubpixel(const Mat &mat) {
  const auto m = moments(mat);
  return Point2d(m.m10 / m.m00, m.m01 / m.m00);
}

bool AccuracyReached(const Point2d &L1peak, const Point2d &L1mid) {
  return std::abs(L1peak.x - L1mid.x) < 0.5 &&
         std::abs(L1peak.y - L1mid.y) < 0.5;
}

Point2d GetSubpixelShift(const Mat &L3, const Point2d &L3peak,
                         const Point2d &L3mid, int L2size) {
  while (IsOutOfBounds(L3peak, L3, L2size))
    if (!ReduceL2size(L2size))
      return L3peak - L3mid;

  Mat L2 =
      L3(Rect(L3peak.x - L2size / 2, L3peak.y - L2size / 2, L2size, L2size));
  Point2d L2peak = GetPeakSubpixel(L2);
  Point2d L2mid(L2.cols / 2, L2.rows / 2);
  return L3peak - L3mid + L2peak - L2mid;
}

Point2d phaseCorrelateIterative(InputArray _src1, InputArray _src2) {
  CV_INSTRUMENT_REGION();

  Mat src1 = _src1.getMat();
  Mat src2 = _src2.getMat();

  CV_Assert(src1.type() == src2.type());
  CV_Assert(src1.type() == CV_32FC1 || src1.type() == CV_64FC1);
  CV_Assert(src1.size == src2.size);

  // apply DFT window to input images
  Mat window;
  createHanningWindow(window, src1.size(), _src1.type());
  Mat image1, image2;
  multiply(_src1, window, image1);
  multiply(_src2, window, image2);

  // compute the DFTs of input images
  dft(image1, image1, DFT_COMPLEX_OUTPUT);
  dft(image2, image2, DFT_COMPLEX_OUTPUT);

  // compute the normalized cross-power spectrum
  Mat crosspower =
      CalculateCrossPowerSpectrum(std::move(image1), std::move(image2));

  // compute the phase correlation landscape (L3)
  dft(crosspower, crosspower, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT);
  fftshift(crosspower);
  Mat L3 = crosspower;
  Point2d L3mid(L3.cols / 2, L3.rows / 2);

  // calculate the maximum correlation location
  Point2i L3peak;
  minMaxLoc(L3, nullptr, nullptr, nullptr, &L3peak);

  // reduce the L2size as long as the L2 is out of bounds of L3
  int L2size = 7;
  while (IsOutOfBounds(L3peak, L3, L2size))
    if (!ReduceL2size(L2size))
      return Point2d(L3peak) - L3mid;

  // extract the L2 maximum correlation neighborhood
  Mat L2 =
      L3(Rect(L3peak.x - L2size / 2, L3peak.y - L2size / 2, L2size, L2size));
  Point2d L2mid(L2.cols / 2, L2.rows / 2);

  // upsample L2 maximum correlation neighborhood to get L2U
  Mat L2U;
  const int L2Usize = 357;
  resize(L2, L2U, {L2Usize, L2Usize}, 0, 0, INTER_LINEAR);
  Point2d L2Umid(L2U.cols / 2, L2U.rows / 2);

  // initialize L1 centroid neighborhood parameters
  const double L1ratioBase = 0.45;
  const double L1ratioStep = 0.025;
  const int MaxIter = 10; // maximum number of iterative refinement iterations

  // run the iterative refinement algorithm using the specified L1 ratio,
  // gradually decrease L1 ratio if convergence is not achieved
  for (double L1ratio = L1ratioBase; GetL1size(L2U.cols, L1ratio) > 0;
       L1ratio -= L1ratioStep) {

    Point2d L2Upeak = L2Umid; // reset the accumulated L2U peak position
    const int L1size =
        GetL1size(L2U.cols, L1ratio); // calculate the current L1 size
    const Point2d L1mid(L1size / 2, L1size / 2); // update the L1 mid position

    // perform the iterative refinement algorithm
    for (int iter = 0; iter < MaxIter; ++iter) {

      // verify that the L1 region is withing the upsampled L2U region
      if (IsOutOfBounds(L2Upeak, L2U, L1size))
        break;

      // extract the L1 region
      const Mat L1 = L2U(
          Rect(L2Upeak.x - L1size / 2, L2Upeak.y - L1size / 2, L1size, L1size));

      // calculate the centroid location using the specified L1 mask
      const Point2d L1peak = GetPeakSubpixel(L1);

      // add the contribution of the current iteration to the accumulated L2U
      // peak location
      L2Upeak += Point2d(std::round(L1peak.x - L1mid.x),
                         std::round(L1peak.y - L1mid.y));

      // check for convergence
      if (AccuracyReached(L1peak, L1mid)) {
        // return the refined subpixel image shift
        const double upsampleCoeff = static_cast<double>(L2Usize) / L2size;
        return Point2d(L3peak) - L3mid +
               (L2Upeak - L2Umid + L1peak - L1mid) / upsampleCoeff;
      }
    }
  }

  // iterative refinement failed to converge,return non-iterative subpixel shift
  return GetSubpixelShift(L3, Point2d(L3peak), L3mid, L2size);
}
} // namespace cv
