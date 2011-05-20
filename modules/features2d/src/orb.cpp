/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/** Authors: Ethan Rublee, Vincent Rabaud, Gary Bradski */

#include "precomp.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{

/** Function that computes the Harris response in a 9 x 9 patch at a given point in an image
 * @param patch the 9 x 9 patch
 * @param k the k in the Harris formula
 * @param dX_offsets pre-computed offset to get all the interesting dX values
 * @param dY_offsets pre-computed offset to get all the interesting dY values
 * @return
 */
template<typename PatchType, typename SumType>
  inline float harris(const cv::Mat& patch, float k, const std::vector<int> &dX_offsets,
                      const std::vector<int> &dY_offsets)
  {
    float a = 0, b = 0, c = 0;

    static cv::Mat_<SumType> dX(9, 7), dY(7, 9);
    SumType * dX_data = reinterpret_cast<SumType*> (dX.data), *dY_data = reinterpret_cast<SumType*> (dY.data);
    SumType * dX_data_end = dX_data + 9 * 7;
    PatchType * patch_data = reinterpret_cast<PatchType*> (patch.data);
    int two_row_offset = 2 * patch.step1();
    std::vector<int>::const_iterator dX_offset = dX_offsets.begin(), dY_offset = dY_offsets.begin();
    // Compute the differences
    for (; dX_data != dX_data_end; ++dX_data, ++dY_data, ++dX_offset, ++dY_offset)
    {
      *dX_data = (SumType)(*(patch_data + *dX_offset)) - (SumType)(*(patch_data + *dX_offset - 2));
      *dY_data = (SumType)(*(patch_data + *dY_offset)) - (SumType)(*(patch_data + *dY_offset - two_row_offset));
    }

    // Compute the Scharr result
    dX_data = reinterpret_cast<SumType*> (dX.data);
    dY_data = reinterpret_cast<SumType*> (dY.data);
    for (size_t v = 0; v <= 6; v++, dY_data += 2)
    {
      for (size_t u = 0; u <= 6; u++, ++dX_data, ++dY_data)
      {
        // 1, 2 for Sobel, 3 and 10 for Scharr
        float Ix = 1 * (*dX_data + *(dX_data + 14)) + 2 * (*(dX_data + 7));
        float Iy = 1 * (*dY_data + *(dY_data + 2)) + 2 * (*(dY_data + 1));

        a += Ix * Ix;
        b += Iy * Iy;
        c += Ix * Iy;
      }
    }

    return ((a * b - c * c) - (k * ((a + b) * (a + b))));
  }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Class used to compute the cornerness of specific points in an image */
struct HarrisResponse
{
  /** Constructor
   * @param image the image on which the cornerness will be computed (only its step is used
   * @param k the k in the Harris formula
   */
  explicit HarrisResponse(const cv::Mat& image, double k = 0.04);

  /** Compute the cornerness for given keypoints
   * @param kpts points at which the cornerness is computed and stored
   */
  void operator()(std::vector<cv::KeyPoint>& kpts) const;
private:
  /** The cached image to analyze */
  cv::Mat image_;

  /** The k factor in the Harris corner detection */
  double k_;

  /** The offset in X to compute the differences */
  std::vector<int> dX_offsets_;

  /** The offset in Y to compute the differences */
  std::vector<int> dY_offsets_;
};

/** Constructor
 * @param image the image on which the cornerness will be computed (only its step is used
 * @param k the k in the Harris formula
 */
HarrisResponse::HarrisResponse(const cv::Mat& image, double k) :
  image_(image), k_(k)
{
  // Compute the offsets for the Harris corners once and for all
  dX_offsets_.resize(7 * 9);
  dY_offsets_.resize(7 * 9);
  std::vector<int>::iterator dX_offsets = dX_offsets_.begin(), dY_offsets = dY_offsets_.begin();
  unsigned int image_step = image.step1();
  for (size_t y = 0; y <= 6 * image_step; y += image_step)
  {
    int dX_offset = y + 2, dY_offset = y + 2 * image_step;
    for (size_t x = 0; x <= 6; ++x)
    {
      *(dX_offsets++) = dX_offset++;
      *(dY_offsets++) = dY_offset++;
    }
    for (size_t x = 7; x <= 8; ++x)
      *(dY_offsets++) = dY_offset++;
  }

  for (size_t y = 7 * image_step; y <= 8 * image_step; y += image_step)
  {
    int dX_offset = y + 2;
    for (size_t x = 0; x <= 6; ++x)
      *(dX_offsets++) = dX_offset++;
  }
}

/** Compute the cornerness for given keypoints
 * @param kpts points at which the cornerness is computed and stored
 */
void HarrisResponse::operator()(std::vector<cv::KeyPoint>& kpts) const
{
  // Those parameters are used to match the OpenCV computation of Harris corners
  float scale = (1 << 2) * 7.0 * 255.0;
  scale = 1.0 / scale;
  float scale_sq_sq = scale * scale * scale * scale;

  // define it to 1 if you want to compare to what OpenCV computes
#define HARRIS_TEST 0
#if HARRIS_TEST
  cv::Mat_<float> dst;
  cv::cornerHarris(image_, dst, 7, 3, k_);
#endif
  for (std::vector<cv::KeyPoint>::iterator kpt = kpts.begin(), kpt_end = kpts.end(); kpt != kpt_end; ++kpt)
  {
    cv::Mat patch = image_(cv::Rect(kpt->pt.x - 4, kpt->pt.y - 4, 9, 9));

    // Compute the response
    kpt->response = harris<uchar, int> (patch, k_, dX_offsets_, dY_offsets_) * scale_sq_sq;

#if HARRIS_TEST
    cv::Mat_<float> Ix(9, 9), Iy(9, 9);

    cv::Sobel(patch, Ix, CV_32F, 1, 0, 3, scale);
    cv::Sobel(patch, Iy, CV_32F, 0, 1, 3, scale);
    float a = 0, b = 0, c = 0;
    for (unsigned int y = 1; y <= 7; ++y)
    {
      for (unsigned int x = 1; x <= 7; ++x)
      {
        a += Ix(y, x) * Ix(y, x);
        b += Iy(y, x) * Iy(y, x);
        c += Ix(y, x) * Iy(y, x);
      }
    }
    //[ a c ]
    //[ c b ]
    float response = (float)((a * b - c * c) - k_ * ((a + b) * (a + b)));

    std::cout << kpt->response << " " << response << " " << dst(kpt->pt.y,kpt->pt.x) << std::endl;
#endif
  }
}

namespace
{
struct RoiPredicate
{
  RoiPredicate(const cv::Rect& r) :
    r(r)
  {
  }

  bool operator()(const cv::KeyPoint& keyPt) const
  {
    return !r.contains(keyPt.pt);
  }

  cv::Rect r;
};

void runByImageBorder(std::vector<cv::KeyPoint>& keypoints, cv::Size imageSize, int borderSize)
{
  if (borderSize > 0)
  {
    keypoints.erase(
                    std::remove_if(
                                   keypoints.begin(),
                                   keypoints.end(),
                                   RoiPredicate(
                                                cv::Rect(
                                                         cv::Point(borderSize, borderSize),
                                                         cv::Point(imageSize.width - borderSize,
                                                                   imageSize.height - borderSize)))), keypoints.end());
  }
}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool keypointResponseGreater(const cv::KeyPoint& lhs, const cv::KeyPoint& rhs)
{
  return lhs.response > rhs.response;
}

/** Simple function that returns the area in the rectangle x1<=x<=x2, y1<=y<=y2 given an integral image
 * @param integral_image
 * @param x1
 * @param y1
 * @param x2
 * @param y2
 * @return
 */
template<typename SumType>
  inline SumType integral_rectangle(const SumType * val_ptr, std::vector<int>::const_iterator offset)
  {
    return *(val_ptr + *offset) - *(val_ptr + *(offset + 1)) - *(val_ptr + *(offset + 2)) + *(val_ptr + *(offset + 3));
  }

template<typename SumType>
  void IC_Angle_Integral(const cv::Mat& integral_image, const int half_k, cv::KeyPoint& kpt,
                         const std::vector<int> &horizontal_offsets, const std::vector<int> &vertical_offsets)
  {
    SumType m_01 = 0, m_10 = 0;

    // Go line by line in the circular patch
    std::vector<int>::const_iterator horizontal_iterator = horizontal_offsets.begin(), vertical_iterator =
        vertical_offsets.begin();
    const SumType* val_ptr = &(integral_image.at<SumType> (kpt.pt.y, kpt.pt.x));
    for (int uv = 1; uv <= half_k; ++uv)
    {
      // Do the horizontal lines
      m_01 += uv * (-integral_rectangle(val_ptr, horizontal_iterator) + integral_rectangle(val_ptr,
                                                                                           horizontal_iterator + 4));
      horizontal_iterator += 8;

      // Do the vertical lines
      m_10 += uv * (-integral_rectangle(val_ptr, vertical_iterator)
          + integral_rectangle(val_ptr, vertical_iterator + 4));
      vertical_iterator += 8;
    }

    float x = m_10;
    float y = m_01;
    kpt.angle = cv::fastAtan2(y, x);
  }

template<typename PatchType, typename SumType>
  void IC_Angle(const cv::Mat& image, const int half_k, cv::KeyPoint& kpt, const std::vector<int> & u_max)
  {
    SumType m_01 = 0, m_10 = 0/*, m_00 = 0*/;

    const PatchType* val_center_ptr_plus = &(image.at<PatchType> (kpt.pt.y, kpt.pt.x)), *val_center_ptr_minus;

    // Treat the center line differently, v=0

    {
      const PatchType* val = val_center_ptr_plus - half_k;
      for (int u = -half_k; u <= half_k; ++u, ++val)
        m_10 += u * (SumType)(*val);
    }

    // Go line by line in the circular patch
    val_center_ptr_minus = val_center_ptr_plus - image.step1();
    val_center_ptr_plus += image.step1();
    for (int v = 1; v <= half_k; ++v, val_center_ptr_plus += image.step1(), val_center_ptr_minus -= image.step1())
    {
      // The beginning of the two lines
      const PatchType* val_ptr_plus = val_center_ptr_plus - u_max[v];
      const PatchType* val_ptr_minus = val_center_ptr_minus - u_max[v];

      // Proceed over the two lines
      SumType v_sum = 0;
      for (int u = -u_max[v]; u <= u_max[v]; ++u, ++val_ptr_plus, ++val_ptr_minus)
      {
        SumType val_plus = *val_ptr_plus, val_minus = *val_ptr_minus;
        v_sum += (val_plus - val_minus);
        m_10 += u * (val_plus + val_minus);
      }
      m_01 += v * v_sum;
    }

    float x = m_10;// / float(m_00);// / m_00;
    float y = m_01;// / float(m_00);// / m_00;
    kpt.angle = cv::fastAtan2(y, x);
  }

inline int smoothedSum(const int *center, const int* int_diff)
{
  // Points in order 01
  //                 32
  return *(center + int_diff[2]) - *(center + int_diff[3]) - *(center + int_diff[1]) + *(center + int_diff[0]);
}

inline char smoothed_comparison(const int * center, const int* diff, int l, int m)
{
  static const char score[] = {1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1 << 7};
  return (smoothedSum(center, diff + l) < smoothedSum(center, diff + l + 4)) ? score[m] : 0;
}
}

namespace cv
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ORB::OrbPatterns
{
public:
  // We divide in 30 wedges
  static const int kNumAngles = 30;

  /** Constructor
   * Add +1 to the step as this is the step of the integral image, not image
   * @param sz
   * @param normalized_step
   * @return
   */
  OrbPatterns(int sz, unsigned int normalized_step_size) :
    normalized_step_(normalized_step_size)
  {
    relative_patterns_.resize(kNumAngles);
    for (int i = 0; i < kNumAngles; i++)
      generateRelativePattern(i, sz, relative_patterns_[i]);
  }

  /** Generate the patterns and relative patterns
   * @param sz
   * @param normalized_step
   * @return
   */
  static std::vector<cv::Mat> generateRotatedPatterns()
  {
    std::vector<cv::Mat> rotated_patterns(kNumAngles);
    cv::Mat_<cv::Vec2i> pattern = cv::Mat(512, 1, CV_32SC2, bit_pattern_31_);
    for (int i = 0; i < kNumAngles; i++)
    {
      const cv::Mat rotation_matrix = getRotationMat(i);
      transform(pattern, rotated_patterns[i], rotation_matrix);
      // Make sure the pattern is now one channel, and 512*2
      rotated_patterns[i] = rotated_patterns[i].reshape(1, 512);
    }
    return rotated_patterns;
  }

  /** Compute the brief pattern for a given keypoint
   * @param angle the orientation of the keypoint
   * @param sum the integral image
   * @param pt the keypoint
   * @param descriptor the descriptor
   */
  void compute(const cv::KeyPoint& kpt, const cv::Mat& sum, unsigned char * desc) const
  {
    float angle = kpt.angle;

    // Compute the pointer to the center of the feature
    int img_y = (int)(kpt.pt.y + 0.5);
    int img_x = (int)(kpt.pt.x + 0.5);
    const int * center = reinterpret_cast<const int *> (sum.ptr(img_y)) + img_x;
    // Compute the pointer to the absolute pattern row
    const int * diff = relative_patterns_[angle2Wedge(angle)].ptr<int> (0);
    for (int i = 0, j = 0; i < 32; ++i, j += 64)
    {
      desc[i] = smoothed_comparison(center, diff, j, 7) | smoothed_comparison(center, diff, j + 8, 6)
          | smoothed_comparison(center, diff, j + 16, 5) | smoothed_comparison(center, diff, j + 24, 4)
          | smoothed_comparison(center, diff, j + 32, 3) | smoothed_comparison(center, diff, j + 40, 2)
          | smoothed_comparison(center, diff, j + 48, 1) | smoothed_comparison(center, diff, j + 56, 0);
    }
  }

  /** Compare the currently used normalized step of the integral image to a new one
   * @param integral_image the integral we want to use the pattern on
   * @return true if the two steps are equal
   */
  bool compareNormalizedStep(const cv::Mat & integral_image) const
  {
    return (normalized_step_ == integral_image.step1());
  }

  /** Compare the currently used normalized step of the integral image to a new one
   * @param step_size the normalized step size to compare to
   * @return true if the two steps are equal
   */
  bool compareNormalizedStep(unsigned int normalized_step_size) const
  {
    return (normalized_step_ == normalized_step_size);
  }

private:
  static inline int angle2Wedge(float angle)
  {
    return (angle / 360) * kNumAngles;
  }

  void generateRelativePattern(int angle_idx, int sz, cv::Mat & relative_pattern)
  {
    // Create the relative pattern
    relative_pattern.create(512, 4, CV_32SC1);
    int * relative_pattern_data = reinterpret_cast<int*> (relative_pattern.data);
    // Get the original rotated pattern
    const int * pattern_data;
    switch (sz)
    {
      default:
        pattern_data = reinterpret_cast<int*> (rotated_patterns_[angle_idx].data);
        break;
    }

    int half_kernel = ORB::kKernelWidth / 2;
    for (unsigned int i = 0; i < 512; ++i)
    {
      int center = *(pattern_data + 2 * i) + normalized_step_ * (*(pattern_data + 2 * i + 1));
      // Points in order 01
      //                 32
      // +1 is added for certain coordinates for the integral image
      *(relative_pattern_data++) = center - half_kernel - half_kernel * normalized_step_;
      *(relative_pattern_data++) = center + (half_kernel + 1) - half_kernel * normalized_step_;
      *(relative_pattern_data++) = center + (half_kernel + 1) + (half_kernel + 1) * normalized_step_;
      *(relative_pattern_data++) = center - half_kernel + (half_kernel + 1) * normalized_step_;
    }
  }

  static cv::Mat getRotationMat(int angle_idx)
  {
    float a = float(angle_idx) / kNumAngles * CV_PI * 2;
    return (cv::Mat_<float>(2, 2) << cos(a), -sin(a), sin(a), cos(a));
  }

  /** Contains the relative patterns (rotated ones in relative coordinates)
   */
  std::vector<cv::Mat_<int> > relative_patterns_;

  /** The step of the integral image
   */
  size_t normalized_step_;

  /** Pattern loaded from the include files
   */
  static std::vector<cv::Mat> rotated_patterns_;
  static int bit_pattern_31_[256 * 4]; //number of tests * 4 (x1,y1,x2,y2)

};

std::vector<cv::Mat> ORB::OrbPatterns::rotated_patterns_ = OrbPatterns::generateRotatedPatterns();

//this is the definition for BIT_PATTERN
#include "orb_pattern.i"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Constructor
 * @param detector_params parameters to use
 */
ORB::ORB(size_t n_features, const CommonParams & detector_params) :
  params_(detector_params), n_features_(n_features)
{
  // fill the extractors and descriptors for the corresponding scales
  int n_desired_features_per_scale = n_features / ((1.0 / std::pow(params_.scale_factor_, 2 * params_.n_levels_) - 1)
      / (1.0 / std::pow(params_.scale_factor_, 2) - 1));
  n_features_per_level_.resize(detector_params.n_levels_);
  for (unsigned int level = 0; level < detector_params.n_levels_; level++)
  {
    n_desired_features_per_scale /= std::pow(params_.scale_factor_, 2);
    n_features_per_level_[level] = n_desired_features_per_scale;
  }

  // pre-compute the end of a row in a circular patch
  half_patch_size_ = params_.patch_size_ / 2;
  u_max_.resize(half_patch_size_ + 1);
  for (int v = 0; v <= half_patch_size_ * sqrt(2) / 2 + 1; ++v)
    u_max_[v] = std::floor(sqrt(half_patch_size_ * half_patch_size_ - v * v) + 0.5);

  // Make sure we are symmetric
  for (int v = half_patch_size_, v_0 = 0; v >= half_patch_size_ * sqrt(2) / 2; --v)
  {
    while (u_max_[v_0] == u_max_[v_0 + 1])
      ++v_0;
    u_max_[v] = v_0;
    ++v_0;
  }
}

/** returns the descriptor size in bytes */
int ORB::descriptorSize() const {
  return kBytes;
}

/** Compute the ORB features and descriptors on an image
 * @param img the image to compute the features and descriptors on
 * @param mask the mask to apply
 * @param keypoints the resulting keypoints
 */
void ORB::operator()(const cv::Mat &image, const cv::Mat &mask, std::vector<cv::KeyPoint> & keypoints)
{
  cv::Mat empty_descriptors;
  this->operator ()(image, mask, keypoints, empty_descriptors, true, false);
}

/** Compute the ORB features and descriptors on an image
 * @param img the image to compute the features and descriptors on
 * @param mask the mask to apply
 * @param keypoints the resulting keypoints
 * @param descriptors the resulting descriptors
 * @param useProvidedKeypoints if true, the keypoints are used as an input
 */
void ORB::operator()(const cv::Mat &image, const cv::Mat &mask, std::vector<cv::KeyPoint> & keypoints,
                     cv::Mat & descriptors, bool useProvidedKeypoints)
{
  this->operator ()(image, mask, keypoints, descriptors, !useProvidedKeypoints, true);
}

/** Compute the ORB features and descriptors on an image
 * @param img the image to compute the features and descriptors on
 * @param mask the mask to apply
 * @param keypoints the resulting keypoints
 * @param descriptors the resulting descriptors
 * @param do_keypoints if true, the keypoints are computed, otherwise used as an input
 * @param do_descriptors if true, also computes the descriptors
 */
void ORB::operator()(const cv::Mat &image, const cv::Mat &mask, std::vector<cv::KeyPoint> & keypoints_in_out,
                     cv::Mat & descriptors, bool do_keypoints, bool do_descriptors)
{
  if ((!do_keypoints) && (!do_descriptors))
    return;

  if (do_keypoints)
    keypoints_in_out.clear();
  if (do_descriptors)
    descriptors.release();

  // Pre-compute the scale pyramids
  std::vector<cv::Mat> image_pyramid(params_.n_levels_), mask_pyramid(params_.n_levels_);
  for (unsigned int level = 0; level < params_.n_levels_; ++level)
  {
    // Compute the resized image
    if (level != params_.first_level_)
    {
      float scale = 1 / std::pow(params_.scale_factor_, level - params_.first_level_);
      cv::resize(image, image_pyramid[level], cv::Size(), scale, scale, cv::INTER_AREA);
      if (!mask.empty())
        cv::resize(mask, mask_pyramid[level], cv::Size(), scale, scale, cv::INTER_AREA);
    }
    else
    {
      image_pyramid[level] = image;
      mask_pyramid[level] = mask;
    }
  }

  // Pre-compute the keypoints (we keep the best over all scales, so this has to be done beforehand
  std::vector<std::vector<cv::KeyPoint> > all_keypoints;
  if (do_keypoints)
    computeKeyPoints(image_pyramid, mask_pyramid, all_keypoints);
  else
  {
    // Cluster the input keypoints
    all_keypoints.reserve(params_.n_levels_);
    for (std::vector<cv::KeyPoint>::iterator keypoint = keypoints_in_out.begin(), keypoint_end = keypoints_in_out.end(); keypoint
        != keypoint_end; ++keypoint)
      all_keypoints[keypoint->octave].push_back(*keypoint);
  }

  for (unsigned int level = 0; level < params_.n_levels_; ++level)
  {
    // Compute the resized image
    cv::Mat & working_mat = image_pyramid[level];

    // Compute the integral image
    cv::Mat integral_image;
    if (do_descriptors)
      // if we don't do the descriptors (and therefore, we only do the keypoints, it is faster to not compute the
      // integral image
      computeIntegralImage(working_mat, level, integral_image);

    // Compute the features
    std::vector<cv::KeyPoint> & keypoints = all_keypoints[level];
    if (do_keypoints)
      computeOrientation(working_mat, integral_image, level, keypoints);

    // Compute the descriptors
    cv::Mat desc;
    if (do_descriptors)
      computeDescriptors(working_mat, integral_image, level, keypoints, desc);

    // Copy to the output data
    if (!desc.empty())
    {
      if (do_keypoints)
      {
        // Rescale the coordinates
        if (level != params_.first_level_)
        {
          float scale = std::pow(params_.scale_factor_, level - params_.first_level_);
          for (std::vector<cv::KeyPoint>::iterator keypoint = keypoints.begin(), keypoint_end = keypoints.end(); keypoint
              != keypoint_end; ++keypoint)
            keypoint->pt *= scale;
        }
        // And add the keypoints to the output
        keypoints_in_out.insert(keypoints_in_out.end(), keypoints.begin(), keypoints.end());
      }

      if (do_descriptors)
      {
        if (descriptors.empty())
          desc.copyTo(descriptors);
        else
          descriptors.push_back(desc);
      }
    }
  }
}

/** Compute the ORB keypoints on an image
 * @param image_pyramid the image pyramid to compute the features and descriptors on
 * @param mask_pyramid the masks to apply at every level
 * @param keypoints the resulting keypoints, clustered per level
 */
void ORB::computeKeyPoints(const std::vector<cv::Mat>& image_pyramid, const std::vector<cv::Mat>& mask_pyramid,
                           std::vector<std::vector<cv::KeyPoint> >& all_keypoints_out) const
{
  all_keypoints_out.resize(params_.n_levels_);

  std::vector<cv::KeyPoint> all_keypoints;
  all_keypoints.reserve(2 * n_features_);

  for (unsigned int level = 0; level < params_.n_levels_; ++level)
  {
    all_keypoints_out[level].reserve(n_features_per_level_[level]);

    std::vector<cv::KeyPoint> keypoints;

    // Detect FAST features, 20 is a good threshold
    cv::FastFeatureDetector fd(20, true);
    fd.detect(image_pyramid[level], keypoints, mask_pyramid[level]);

    // Remove keypoints very close to the border
    // half_patch_size_ for orientation, 4 for Harris
    unsigned int border_safety = std::max(half_patch_size_, 4);
#if ((CV_MAJOR_VERSION >= 2) && ((CV_MINOR_VERSION >2) || ((CV_MINOR_VERSION == 2) && (CV_SUBMINOR_VERSION>=9))))
    cv::KeyPointsFilter::runByImageBorder(keypoints, image_pyramid[level].size(), border_safety);
#else
    ::runByImageBorder(keypoints, image_pyramid[level].size(), border_safety);
#endif

    // Keep more points than necessary as FAST does not give amazing corners
    if (keypoints.size() > 2 * n_features_per_level_[level])
    {
      std::nth_element(keypoints.begin(), keypoints.begin() + 2 * n_features_per_level_[level], keypoints.end(),
                       keypointResponseGreater);
      keypoints.resize(2 * n_features_per_level_[level]);
    }

    // Compute the Harris cornerness (better scoring than FAST)
    HarrisResponse h(image_pyramid[level]);
    h(keypoints);

    // Set the level of the coordinates
    for (std::vector<cv::KeyPoint>::iterator keypoint = keypoints.begin(), keypoint_end = keypoints.end(); keypoint
        != keypoint_end; ++keypoint)
      keypoint->octave = level;

    all_keypoints.insert(all_keypoints.end(), keypoints.begin(), keypoints.end());
  }

  // Only keep what we need
  if (all_keypoints.size() > n_features_)
  {
    std::nth_element(all_keypoints.begin(), all_keypoints.begin() + n_features_, all_keypoints.end(),
                     keypointResponseGreater);
    all_keypoints.resize(n_features_);
  }

  // Cluster the keypoints
  for (std::vector<cv::KeyPoint>::iterator keypoint = all_keypoints.begin(), keypoint_end = all_keypoints.end(); keypoint
      != keypoint_end; ++keypoint)
    all_keypoints_out[keypoint->octave].push_back(*keypoint);
}

/** Compute the ORB keypoint orientations
 * @param image the image to compute the features and descriptors on
 * @param integral_image the integral image of the iamge (can be empty, but the computation will be slower)
 * @param scale the scale at which we compute the orientation
 * @param keypoints the resulting keypoints
 */
void ORB::computeOrientation(const cv::Mat& image, const cv::Mat& integral_image, unsigned int scale,
                             std::vector<cv::KeyPoint>& keypoints) const
{
  // If using the integral image, some offsets will be pre-computed for speed
  std::vector<int> horizontal_offsets(8 * half_patch_size_), vertical_offsets(8 * half_patch_size_);

  // Process each keypoint
  for (std::vector<cv::KeyPoint>::iterator keypoint = keypoints.begin(), keypoint_end = keypoints.end(); keypoint
      != keypoint_end; ++keypoint)
  {
    //get a patch at the keypoint
    if (integral_image.empty())
    {
      switch (image.depth())
      {
        case CV_8U:
          IC_Angle<uchar, int> (image, half_patch_size_, *keypoint, u_max_);
          break;
        case CV_32S:
          IC_Angle<int, int> (image, half_patch_size_, *keypoint, u_max_);
          break;
        case CV_32F:
          IC_Angle<float, float> (image, half_patch_size_, *keypoint, u_max_);
          break;
        case CV_64F:
          IC_Angle<double, double> (image, half_patch_size_, *keypoint, u_max_);
          break;
      }
    }
    else
    {
      // use the integral image if you can
      switch (integral_image.depth())
      {
        case CV_32S:
          IC_Angle_Integral<int> (integral_image, half_patch_size_, *keypoint, orientation_horizontal_offsets_[scale],
                                  orientation_vertical_offsets_[scale]);
          break;
        case CV_32F:
          IC_Angle_Integral<float> (integral_image, half_patch_size_, *keypoint,
                                    orientation_horizontal_offsets_[scale], orientation_vertical_offsets_[scale]);
          break;
        case CV_64F:
          IC_Angle_Integral<double> (integral_image, half_patch_size_, *keypoint,
                                     orientation_horizontal_offsets_[scale], orientation_vertical_offsets_[scale]);
          break;
      }
    }
  }
}

/** Compute the integral image and upadte the cached values
 * @param image the image to compute the features and descriptors on
 * @param level the scale at which we compute the orientation
 * @param descriptors the resulting descriptors
 */
void ORB::computeIntegralImage(const cv::Mat & image, unsigned int level, cv::Mat &integral_image)
{
  integral(image, integral_image, CV_32S);
  integral_image_steps_.resize(params_.n_levels_, 0);

  if (integral_image_steps_[level] == integral_image.step1())
    return;

  // If the integral image dimensions have changed, recompute everything
  int integral_image_step = integral_image.step1();

  // Cache the step sizes
  integral_image_steps_[level] = integral_image_step;

  // Cache the offsets for the orientation
  orientation_horizontal_offsets_.resize(params_.n_levels_);
  orientation_vertical_offsets_.resize(params_.n_levels_);
  orientation_horizontal_offsets_[level].resize(8 * half_patch_size_);
  orientation_vertical_offsets_[level].resize(8 * half_patch_size_);
  for (int v = 1, offset_index = 0; v <= half_patch_size_; ++v)
  {
    // Compute the offsets to use if using the integral image
    for (int signed_v = -v; signed_v <= v; signed_v += 2 * v)
    {
      // the offsets are computed so that we can compute the integral image
      // elem at 0 - eleme at 1 - elem at 2 + elem at 3
      orientation_horizontal_offsets_[level][offset_index] = (signed_v + 1) * integral_image_step + u_max_[v] + 1;
      orientation_vertical_offsets_[level][offset_index] = (u_max_[v] + 1) * integral_image_step + signed_v + 1;
      ++offset_index;
      orientation_horizontal_offsets_[level][offset_index] = signed_v * integral_image_step + u_max_[v] + 1;
      orientation_vertical_offsets_[level][offset_index] = -u_max_[v] * integral_image_step + signed_v + 1;
      ++offset_index;
      orientation_horizontal_offsets_[level][offset_index] = (signed_v + 1) * integral_image_step - u_max_[v];
      orientation_vertical_offsets_[level][offset_index] = (u_max_[v] + 1) * integral_image_step + signed_v;
      ++offset_index;
      orientation_horizontal_offsets_[level][offset_index] = signed_v * integral_image_step - u_max_[v];
      orientation_vertical_offsets_[level][offset_index] = -u_max_[v] * integral_image_step + signed_v;
      ++offset_index;
    }
  }

  // Remove the previous version if dimensions are different
  patterns_.resize(params_.n_levels_, 0);
  if ((patterns_[level]) && (patterns_[level]->compareNormalizedStep(integral_image)))
  {
    delete patterns_[level];
    patterns_[level] = 0;
  }
  if (!patterns_[level])
    patterns_[level] = new OrbPatterns(params_.patch_size_, integral_image.step1());
}

/** Compute the ORB decriptors
 * @param image the image to compute the features and descriptors on
 * @param integral_image the integral image of the image (can be empty, but the computation will be slower)
 * @param level the scale at which we compute the orientation
 * @param keypoints the keypoints to use
 * @param descriptors the resulting descriptors
 */
void ORB::computeDescriptors(const cv::Mat& image, const cv::Mat& integral_image, unsigned int level,
                             std::vector<cv::KeyPoint>& keypoints, cv::Mat & descriptors) const
{
  //convert to grayscale if more than one color
  cv::Mat gray_image = image;
  if (image.type() != CV_8UC1)
    cv::cvtColor(image, gray_image, CV_BGR2GRAY);

  int border_safety = params_.patch_size_ + kKernelWidth / 2 + 2;
  //Remove keypoints very close to the border
  cv::KeyPointsFilter::runByImageBorder(keypoints, image.size(), border_safety);

  // Get the patterns to apply
  cv::Ptr<OrbPatterns> patterns = patterns_[level];

  //create the descriptor mat, keypoints.size() rows, BYTES cols
  descriptors = cv::Mat::zeros(keypoints.size(), kBytes, CV_8UC1);

  for (size_t i = 0; i < keypoints.size(); i++)
    // look up the test pattern
    patterns->compute(keypoints[i], integral_image, descriptors.ptr(i));
}

}
