/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (C) 2011  The Autonomous Systems Lab (ASL), ETH Zurich,
 *                Stefan Leutenegger, Simon Lynen and Margarita Chli.
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

/*
 BRISK - Binary Robust Invariant Scalable Keypoints
 Reference implementation of
 [1] Stefan Leutenegger,Margarita Chli and Roland Siegwart, BRISK:
 Binary Robust Invariant Scalable Keypoints, in Proceedings of
 the IEEE International Conference on Computer Vision (ICCV2011).
 */

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <stdlib.h>

#include "fast_score.hpp"

namespace cv
{

// a layer in the Brisk detector pyramid
class CV_EXPORTS BriskLayer
{
public:
  // constructor arguments
  struct CV_EXPORTS CommonParams
  {
    static const int HALFSAMPLE = 0;
    static const int TWOTHIRDSAMPLE = 1;
  };
  // construct a base layer
  BriskLayer(const cv::Mat& img, float scale = 1.0f, float offset = 0.0f);
  // derive a layer
  BriskLayer(const BriskLayer& layer, int mode);

  // Fast/Agast without non-max suppression
  void
  getAgastPoints(int threshold, std::vector<cv::KeyPoint>& keypoints);

  // get scores - attention, this is in layer coordinates, not scale=1 coordinates!
  inline int
  getAgastScore(int x, int y, int threshold) const;
  inline int
  getAgastScore_5_8(int x, int y, int threshold) const;
  inline int
  getAgastScore(float xf, float yf, int threshold, float scale = 1.0f) const;

  // accessors
  inline const cv::Mat&
  img() const
  {
    return img_;
  }
  inline const cv::Mat&
  scores() const
  {
    return scores_;
  }
  inline float
  scale() const
  {
    return scale_;
  }
  inline float
  offset() const
  {
    return offset_;
  }

  // half sampling
  static inline void
  halfsample(const cv::Mat& srcimg, cv::Mat& dstimg);
  // two third sampling
  static inline void
  twothirdsample(const cv::Mat& srcimg, cv::Mat& dstimg);

private:
  // access gray values (smoothed/interpolated)
  inline int
  value(const cv::Mat& mat, float xf, float yf, float scale) const;
  // the image
  cv::Mat img_;
  // its Fast scores
  cv::Mat_<uchar> scores_;
  // coordinate transformation
  float scale_;
  float offset_;
  // agast
  cv::Ptr<cv::FastFeatureDetector2> fast_9_16_;
  int pixel_5_8_[25];
  int pixel_9_16_[25];
};

class CV_EXPORTS BriskScaleSpace
{
public:
  // construct telling the octaves number:
  BriskScaleSpace(int _octaves = 3);
  ~BriskScaleSpace();

  // construct the image pyramids
  void
  constructPyramid(const cv::Mat& image);

  // get Keypoints
  void
  getKeypoints(const int _threshold, std::vector<cv::KeyPoint>& keypoints);

protected:
  // nonmax suppression:
  inline bool
  isMax2D(const int layer, const int x_layer, const int y_layer);
  // 1D (scale axis) refinement:
  inline float
  refine1D(const float s_05, const float s0, const float s05, float& max) const; // around octave
  inline float
  refine1D_1(const float s_05, const float s0, const float s05, float& max) const; // around intra
  inline float
  refine1D_2(const float s_05, const float s0, const float s05, float& max) const; // around octave 0 only
  // 2D maximum refinement:
  inline float
  subpixel2D(const int s_0_0, const int s_0_1, const int s_0_2, const int s_1_0, const int s_1_1, const int s_1_2,
             const int s_2_0, const int s_2_1, const int s_2_2, float& delta_x, float& delta_y) const;
  // 3D maximum refinement centered around (x_layer,y_layer)
  inline float
  refine3D(const int layer, const int x_layer, const int y_layer, float& x, float& y, float& scale, bool& ismax) const;

  // interpolated score access with recalculation when needed:
  inline int
  getScoreAbove(const int layer, const int x_layer, const int y_layer) const;
  inline int
  getScoreBelow(const int layer, const int x_layer, const int y_layer) const;

  // return the maximum of score patches above or below
  inline float
  getScoreMaxAbove(const int layer, const int x_layer, const int y_layer, const int threshold, bool& ismax,
                   float& dx, float& dy) const;
  inline float
  getScoreMaxBelow(const int layer, const int x_layer, const int y_layer, const int threshold, bool& ismax,
                   float& dx, float& dy) const;

  // the image pyramids:
  int layers_;
  std::vector<BriskLayer> pyramid_;

  // some constant parameters:
  static const float safetyFactor_;
  static const float basicSize_;
};

const float BRISK::basicSize_ = 12.0f;
const unsigned int BRISK::scales_ = 64;
const float BRISK::scalerange_ = 30.f; // 40->4 Octaves - else, this needs to be adjusted...
const unsigned int BRISK::n_rot_ = 1024; // discretization of the rotation look-up

const float BriskScaleSpace::safetyFactor_ = 1.0f;
const float BriskScaleSpace::basicSize_ = 12.0f;

// constructors
BRISK::BRISK(int thresh, int octaves_in, float patternScale)
{
  threshold = thresh;
  octaves = octaves_in;

  std::vector<float> rList;
  std::vector<int> nList;

  // this is the standard pattern found to be suitable also
  rList.resize(5);
  nList.resize(5);
  const double f = 0.85 * patternScale;

  rList[0] = (float)(f * 0.);
  rList[1] = (float)(f * 2.9);
  rList[2] = (float)(f * 4.9);
  rList[3] = (float)(f * 7.4);
  rList[4] = (float)(f * 10.8);

  nList[0] = 1;
  nList[1] = 10;
  nList[2] = 14;
  nList[3] = 15;
  nList[4] = 20;

  generateKernel(rList, nList, (float)(5.85 * patternScale), (float)(8.2 * patternScale));

}
BRISK::BRISK(std::vector<float> &radiusList, std::vector<int> &numberList, float dMax, float dMin,
                                                   std::vector<int> indexChange)
{
  generateKernel(radiusList, numberList, dMax, dMin, indexChange);
}

void
BRISK::generateKernel(std::vector<float> &radiusList, std::vector<int> &numberList, float dMax,
                                         float dMin, std::vector<int> indexChange)
{

  dMax_ = dMax;
  dMin_ = dMin;

  // get the total number of points
  const int rings = (int)radiusList.size();
  assert(radiusList.size()!=0&&radiusList.size()==numberList.size());
  points_ = 0; // remember the total number of points
  for (int ring = 0; ring < rings; ring++)
  {
    points_ += numberList[ring];
  }
  // set up the patterns
  patternPoints_ = new BriskPatternPoint[points_ * scales_ * n_rot_];
  BriskPatternPoint* patternIterator = patternPoints_;

  // define the scale discretization:
  static const float lb_scale = (float)(log(scalerange_) / log(2.0));
  static const float lb_scale_step = lb_scale / (scales_);

  scaleList_ = new float[scales_];
  sizeList_ = new unsigned int[scales_];

  const float sigma_scale = 1.3f;

  for (unsigned int scale = 0; scale < scales_; ++scale)
  {
    scaleList_[scale] = (float)pow((double) 2.0, (double) (scale * lb_scale_step));
    sizeList_[scale] = 0;

    // generate the pattern points look-up
    double alpha, theta;
    for (size_t rot = 0; rot < n_rot_; ++rot)
    {
      theta = double(rot) * 2 * CV_PI / double(n_rot_); // this is the rotation of the feature
      for (int ring = 0; ring < rings; ++ring)
      {
        for (int num = 0; num < numberList[ring]; ++num)
        {
          // the actual coordinates on the circle
          alpha = (double(num)) * 2 * CV_PI / double(numberList[ring]);
          patternIterator->x = (float)(scaleList_[scale] * radiusList[ring] * cos(alpha + theta)); // feature rotation plus angle of the point
          patternIterator->y = (float)(scaleList_[scale] * radiusList[ring] * sin(alpha + theta));
          // and the gaussian kernel sigma
          if (ring == 0)
          {
            patternIterator->sigma = sigma_scale * scaleList_[scale] * 0.5f;
          }
          else
          {
            patternIterator->sigma = (float)(sigma_scale * scaleList_[scale] * (double(radiusList[ring]))
                                     * sin(CV_PI / numberList[ring]));
          }
          // adapt the sizeList if necessary
          const unsigned int size = cvCeil(((scaleList_[scale] * radiusList[ring]) + patternIterator->sigma)) + 1;
          if (sizeList_[scale] < size)
          {
            sizeList_[scale] = size;
          }

          // increment the iterator
          ++patternIterator;
        }
      }
    }
  }

  // now also generate pairings
  shortPairs_ = new BriskShortPair[points_ * (points_ - 1) / 2];
  longPairs_ = new BriskLongPair[points_ * (points_ - 1) / 2];
  noShortPairs_ = 0;
  noLongPairs_ = 0;

  // fill indexChange with 0..n if empty
  unsigned int indSize = (unsigned int)indexChange.size();
  if (indSize == 0)
  {
    indexChange.resize(points_ * (points_ - 1) / 2);
    indSize = (unsigned int)indexChange.size();

    for (unsigned int i = 0; i < indSize; i++)
      indexChange[i] = i;
  }
  const float dMin_sq = dMin_ * dMin_;
  const float dMax_sq = dMax_ * dMax_;
  for (unsigned int i = 1; i < points_; i++)
  {
    for (unsigned int j = 0; j < i; j++)
    { //(find all the pairs)
      // point pair distance:
      const float dx = patternPoints_[j].x - patternPoints_[i].x;
      const float dy = patternPoints_[j].y - patternPoints_[i].y;
      const float norm_sq = (dx * dx + dy * dy);
      if (norm_sq > dMin_sq)
      {
        // save to long pairs
        BriskLongPair& longPair = longPairs_[noLongPairs_];
        longPair.weighted_dx = int((dx / (norm_sq)) * 2048.0 + 0.5);
        longPair.weighted_dy = int((dy / (norm_sq)) * 2048.0 + 0.5);
        longPair.i = i;
        longPair.j = j;
        ++noLongPairs_;
      }
      else if (norm_sq < dMax_sq)
      {
        // save to short pairs
        assert(noShortPairs_<indSize);
        // make sure the user passes something sensible
        BriskShortPair& shortPair = shortPairs_[indexChange[noShortPairs_]];
        shortPair.j = j;
        shortPair.i = i;
        ++noShortPairs_;
      }
    }
  }

  // no bits:
  strings_ = (int) ceil((float(noShortPairs_)) / 128.0) * 4 * 4;
}

// simple alternative:
inline int
BRISK::smoothedIntensity(const cv::Mat& image, const cv::Mat& integral, const float key_x,
                                            const float key_y, const unsigned int scale, const unsigned int rot,
                                            const unsigned int point) const
{

  // get the float position
  const BriskPatternPoint& briskPoint = patternPoints_[scale * n_rot_ * points_ + rot * points_ + point];
  const float xf = briskPoint.x + key_x;
  const float yf = briskPoint.y + key_y;
  const int x = int(xf);
  const int y = int(yf);
  const int& imagecols = image.cols;

  // get the sigma:
  const float sigma_half = briskPoint.sigma;
  const float area = 4.0f * sigma_half * sigma_half;

  // calculate output:
  int ret_val;
  if (sigma_half < 0.5)
  {
    //interpolation multipliers:
    const int r_x = (int)((xf - x) * 1024);
    const int r_y = (int)((yf - y) * 1024);
    const int r_x_1 = (1024 - r_x);
    const int r_y_1 = (1024 - r_y);
    const uchar* ptr = &image.at<uchar>(y, x);
    size_t step = image.step;
    // just interpolate:
    ret_val = r_x_1 * r_y_1 * ptr[0] + r_x * r_y_1 * ptr[1] +
              r_x * r_y * ptr[step] + r_x_1 * r_y * ptr[step+1];
    return (ret_val + 512) / 1024;
  }

  // this is the standard case (simple, not speed optimized yet):

  // scaling:
  const int scaling = (int)(4194304.0 / area);
  const int scaling2 = int(float(scaling) * area / 1024.0);

  // the integral image is larger:
  const int integralcols = imagecols + 1;

  // calculate borders
  const float x_1 = xf - sigma_half;
  const float x1 = xf + sigma_half;
  const float y_1 = yf - sigma_half;
  const float y1 = yf + sigma_half;

  const int x_left = int(x_1 + 0.5);
  const int y_top = int(y_1 + 0.5);
  const int x_right = int(x1 + 0.5);
  const int y_bottom = int(y1 + 0.5);

  // overlap area - multiplication factors:
  const float r_x_1 = float(x_left) - x_1 + 0.5f;
  const float r_y_1 = float(y_top) - y_1 + 0.5f;
  const float r_x1 = x1 - float(x_right) + 0.5f;
  const float r_y1 = y1 - float(y_bottom) + 0.5f;
  const int dx = x_right - x_left - 1;
  const int dy = y_bottom - y_top - 1;
  const int A = (int)((r_x_1 * r_y_1) * scaling);
  const int B = (int)((r_x1 * r_y_1) * scaling);
  const int C = (int)((r_x1 * r_y1) * scaling);
  const int D = (int)((r_x_1 * r_y1) * scaling);
  const int r_x_1_i = (int)(r_x_1 * scaling);
  const int r_y_1_i = (int)(r_y_1 * scaling);
  const int r_x1_i = (int)(r_x1 * scaling);
  const int r_y1_i = (int)(r_y1 * scaling);

  if (dx + dy > 2)
  {
    // now the calculation:
    uchar* ptr = image.data + x_left + imagecols * y_top;
    // first the corners:
    ret_val = A * int(*ptr);
    ptr += dx + 1;
    ret_val += B * int(*ptr);
    ptr += dy * imagecols + 1;
    ret_val += C * int(*ptr);
    ptr -= dx + 1;
    ret_val += D * int(*ptr);

    // next the edges:
    int* ptr_integral = (int*) integral.data + x_left + integralcols * y_top + 1;
    // find a simple path through the different surface corners
    const int tmp1 = (*ptr_integral);
    ptr_integral += dx;
    const int tmp2 = (*ptr_integral);
    ptr_integral += integralcols;
    const int tmp3 = (*ptr_integral);
    ptr_integral++;
    const int tmp4 = (*ptr_integral);
    ptr_integral += dy * integralcols;
    const int tmp5 = (*ptr_integral);
    ptr_integral--;
    const int tmp6 = (*ptr_integral);
    ptr_integral += integralcols;
    const int tmp7 = (*ptr_integral);
    ptr_integral -= dx;
    const int tmp8 = (*ptr_integral);
    ptr_integral -= integralcols;
    const int tmp9 = (*ptr_integral);
    ptr_integral--;
    const int tmp10 = (*ptr_integral);
    ptr_integral -= dy * integralcols;
    const int tmp11 = (*ptr_integral);
    ptr_integral++;
    const int tmp12 = (*ptr_integral);

    // assign the weighted surface integrals:
    const int upper = (tmp3 - tmp2 + tmp1 - tmp12) * r_y_1_i;
    const int middle = (tmp6 - tmp3 + tmp12 - tmp9) * scaling;
    const int left = (tmp9 - tmp12 + tmp11 - tmp10) * r_x_1_i;
    const int right = (tmp5 - tmp4 + tmp3 - tmp6) * r_x1_i;
    const int bottom = (tmp7 - tmp6 + tmp9 - tmp8) * r_y1_i;

    return (ret_val + upper + middle + left + right + bottom + scaling2 / 2) / scaling2;
  }

  // now the calculation:
  uchar* ptr = image.data + x_left + imagecols * y_top;
  // first row:
  ret_val = A * int(*ptr);
  ptr++;
  const uchar* end1 = ptr + dx;
  for (; ptr < end1; ptr++)
  {
    ret_val += r_y_1_i * int(*ptr);
  }
  ret_val += B * int(*ptr);
  // middle ones:
  ptr += imagecols - dx - 1;
  uchar* end_j = ptr + dy * imagecols;
  for (; ptr < end_j; ptr += imagecols - dx - 1)
  {
    ret_val += r_x_1_i * int(*ptr);
    ptr++;
    const uchar* end2 = ptr + dx;
    for (; ptr < end2; ptr++)
    {
      ret_val += int(*ptr) * scaling;
    }
    ret_val += r_x1_i * int(*ptr);
  }
  // last row:
  ret_val += D * int(*ptr);
  ptr++;
  const uchar* end3 = ptr + dx;
  for (; ptr < end3; ptr++)
  {
    ret_val += r_y1_i * int(*ptr);
  }
  ret_val += C * int(*ptr);

  return (ret_val + scaling2 / 2) / scaling2;
}

inline bool
RoiPredicate(const float minX, const float minY, const float maxX, const float maxY, const KeyPoint& keyPt)
{
  const Point2f& pt = keyPt.pt;
  return (pt.x < minX) || (pt.x >= maxX) || (pt.y < minY) || (pt.y >= maxY);
}

// computes the descriptor
void
BRISK::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& keypoints,
                   OutputArray _descriptors, bool useProvidedKeypoints) const
{
  bool doOrientation=true;
  if (useProvidedKeypoints)
    doOrientation = false;

  // If the user specified cv::noArray(), this will yield false. Otherwise it will return true.
  bool doDescriptors = _descriptors.needed();

  computeDescriptorsAndOrOrientation(_image, _mask, keypoints, _descriptors, doDescriptors, doOrientation,
                                       useProvidedKeypoints);
}

void
BRISK::computeDescriptorsAndOrOrientation(InputArray _image, InputArray _mask, vector<KeyPoint>& keypoints,
                                     OutputArray _descriptors, bool doDescriptors, bool doOrientation,
                                     bool useProvidedKeypoints) const
{
  Mat image = _image.getMat(), mask = _mask.getMat();
  if( image.type() != CV_8UC1 )
      cvtColor(image, image, CV_BGR2GRAY);

  if (!useProvidedKeypoints)
  {
    doOrientation = true;
    computeKeypointsNoOrientation(_image, _mask, keypoints);
  }

  //Remove keypoints very close to the border
  size_t ksize = keypoints.size();
  std::vector<int> kscales; // remember the scale per keypoint
  kscales.resize(ksize);
  static const float log2 = 0.693147180559945f;
  static const float lb_scalerange = (float)(log(scalerange_) / (log2));
  std::vector<cv::KeyPoint>::iterator beginning = keypoints.begin();
  std::vector<int>::iterator beginningkscales = kscales.begin();
  static const float basicSize06 = basicSize_ * 0.6f;
  for (size_t k = 0; k < ksize; k++)
  {
    unsigned int scale;
      scale = std::max((int) (scales_ / lb_scalerange * (log(keypoints[k].size / (basicSize06)) / log2) + 0.5), 0);
      // saturate
      if (scale >= scales_)
        scale = scales_ - 1;
      kscales[k] = scale;
    const int border = sizeList_[scale];
    const int border_x = image.cols - border;
    const int border_y = image.rows - border;
    if (RoiPredicate((float)border, (float)border, (float)border_x, (float)border_y, keypoints[k]))
    {
      keypoints.erase(beginning + k);
      kscales.erase(beginningkscales + k);
      if (k == 0)
      {
        beginning = keypoints.begin();
        beginningkscales = kscales.begin();
      }
      ksize--;
      k--;
    }
  }

  // first, calculate the integral image over the whole image:
  // current integral image
  cv::Mat _integral; // the integral image
  cv::integral(image, _integral);

  int* _values = new int[points_]; // for temporary use

  // resize the descriptors:
  cv::Mat descriptors;
  if (doDescriptors)
  {
    _descriptors.create((int)ksize, strings_, CV_8U);
    descriptors = _descriptors.getMat();
    descriptors.setTo(0);
  }

  // now do the extraction for all keypoints:

  // temporary variables containing gray values at sample points:
  int t1;
  int t2;

  // the feature orientation
  uchar* ptr = descriptors.data;
  for (size_t k = 0; k < ksize; k++)
  {
    cv::KeyPoint& kp = keypoints[k];
    const int& scale = kscales[k];
    int* pvalues = _values;
    const float& x = kp.pt.x;
    const float& y = kp.pt.y;

    if (doOrientation)
    {
        // get the gray values in the unrotated pattern
        for (unsigned int i = 0; i < points_; i++)
        {
          *(pvalues++) = smoothedIntensity(image, _integral, x, y, scale, 0, i);
        }

        int direction0 = 0;
        int direction1 = 0;
        // now iterate through the long pairings
        const BriskLongPair* max = longPairs_ + noLongPairs_;
        for (BriskLongPair* iter = longPairs_; iter < max; ++iter)
        {
          t1 = *(_values + iter->i);
          t2 = *(_values + iter->j);
          const int delta_t = (t1 - t2);
          // update the direction:
          const int tmp0 = delta_t * (iter->weighted_dx) / 1024;
          const int tmp1 = delta_t * (iter->weighted_dy) / 1024;
          direction0 += tmp0;
          direction1 += tmp1;
        }
        kp.angle = (float)(atan2((float) direction1, (float) direction0) / CV_PI * 180.0);
        if (kp.angle < 0)
          kp.angle += 360.f;
    }

    if (!doDescriptors)
      continue;

    int theta;
    if (kp.angle==-1)
    {
        // don't compute the gradient direction, just assign a rotation of 0°
        theta = 0;
    }
    else
    {
        theta = (int) (n_rot_ * (kp.angle / (360.0)) + 0.5);
        if (theta < 0)
          theta += n_rot_;
        if (theta >= int(n_rot_))
          theta -= n_rot_;
    }

    // now also extract the stuff for the actual direction:
    // let us compute the smoothed values
    int shifter = 0;

    //unsigned int mean=0;
    pvalues = _values;
    // get the gray values in the rotated pattern
    for (unsigned int i = 0; i < points_; i++)
    {
      *(pvalues++) = smoothedIntensity(image, _integral, x, y, scale, theta, i);
    }

    // now iterate through all the pairings
    unsigned int* ptr2 = (unsigned int*) ptr;
    const BriskShortPair* max = shortPairs_ + noShortPairs_;
    for (BriskShortPair* iter = shortPairs_; iter < max; ++iter)
    {
      t1 = *(_values + iter->i);
      t2 = *(_values + iter->j);
      if (t1 > t2)
      {
        *ptr2 |= ((1) << shifter);

      } // else already initialized with zero
      // take care of the iterators:
      ++shifter;
      if (shifter == 32)
      {
        shifter = 0;
        ++ptr2;
      }
    }

    ptr += strings_;
  }

  // clean-up
  delete[] _values;
}

int
BRISK::descriptorSize() const
{
  return strings_;
}

int
BRISK::descriptorType() const
{
  return CV_8U;
}

BRISK::~BRISK()
{
  delete[] patternPoints_;
  delete[] shortPairs_;
  delete[] longPairs_;
  delete[] scaleList_;
  delete[] sizeList_;
}

void
BRISK::operator()(InputArray image, InputArray mask, vector<KeyPoint>& keypoints) const
{
  computeKeypointsNoOrientation(image, mask, keypoints);
  computeDescriptorsAndOrOrientation(image, mask, keypoints, cv::noArray(), false, true, true);
}

void
BRISK::computeKeypointsNoOrientation(InputArray _image, InputArray _mask, vector<KeyPoint>& keypoints) const
{
  Mat image = _image.getMat(), mask = _mask.getMat();
  if( image.type() != CV_8UC1 )
      cvtColor(_image, image, CV_BGR2GRAY);

  BriskScaleSpace briskScaleSpace(octaves);
  briskScaleSpace.constructPyramid(image);
  briskScaleSpace.getKeypoints(threshold, keypoints);

  // remove invalid points
  removeInvalidPoints(mask, keypoints);
}


void
BRISK::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const
{
    (*this)(image, mask, keypoints);
}

void
BRISK::computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const
{
    (*this)(image, Mat(), keypoints, descriptors, true);
}

// construct telling the octaves number:
BriskScaleSpace::BriskScaleSpace(int _octaves)
{
  if (_octaves == 0)
    layers_ = 1;
  else
    layers_ = 2 * _octaves;
}
BriskScaleSpace::~BriskScaleSpace()
{

}
// construct the image pyramids
void
BriskScaleSpace::constructPyramid(const cv::Mat& image)
{

  // set correct size:
  pyramid_.clear();

  // fill the pyramid:
  pyramid_.push_back(BriskLayer(image.clone()));
  if (layers_ > 1)
  {
    pyramid_.push_back(BriskLayer(pyramid_.back(), BriskLayer::CommonParams::TWOTHIRDSAMPLE));
  }
  const int octaves2 = layers_;

  for (uchar i = 2; i < octaves2; i += 2)
  {
    pyramid_.push_back(BriskLayer(pyramid_[i - 2], BriskLayer::CommonParams::HALFSAMPLE));
    pyramid_.push_back(BriskLayer(pyramid_[i - 1], BriskLayer::CommonParams::HALFSAMPLE));
  }
}

void
BriskScaleSpace::getKeypoints(const int threshold_, std::vector<cv::KeyPoint>& keypoints)
{
  // make sure keypoints is empty
  keypoints.resize(0);
  keypoints.reserve(2000);

  // assign thresholds
  int safeThreshold_ = (int)(threshold_ * safetyFactor_);
  std::vector<std::vector<cv::KeyPoint> > agastPoints;
  agastPoints.resize(layers_);

  // go through the octaves and intra layers and calculate fast corner scores:
  for (int i = 0; i < layers_; i++)
  {
    // call OAST16_9 without nms
    BriskLayer& l = pyramid_[i];
    l.getAgastPoints(safeThreshold_, agastPoints[i]);
  }

  if (layers_ == 1)
  {
    // just do a simple 2d subpixel refinement...
    const size_t num = agastPoints[0].size();
    for (size_t n = 0; n < num; n++)
    {
      const cv::Point2f& point = agastPoints.at(0)[n].pt;
      // first check if it is a maximum:
      if (!isMax2D(0, (int)point.x, (int)point.y))
        continue;

      // let's do the subpixel and float scale refinement:
      BriskLayer& l = pyramid_[0];
      int s_0_0 = l.getAgastScore(point.x - 1, point.y - 1, 1);
      int s_1_0 = l.getAgastScore(point.x, point.y - 1, 1);
      int s_2_0 = l.getAgastScore(point.x + 1, point.y - 1, 1);
      int s_2_1 = l.getAgastScore(point.x + 1, point.y, 1);
      int s_1_1 = l.getAgastScore(point.x, point.y, 1);
      int s_0_1 = l.getAgastScore(point.x - 1, point.y, 1);
      int s_0_2 = l.getAgastScore(point.x - 1, point.y + 1, 1);
      int s_1_2 = l.getAgastScore(point.x, point.y + 1, 1);
      int s_2_2 = l.getAgastScore(point.x + 1, point.y + 1, 1);
      float delta_x, delta_y;
      float max = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, delta_x, delta_y);

      // store:
      keypoints.push_back(cv::KeyPoint(float(point.x) + delta_x, float(point.y) + delta_y, basicSize_, -1, max, 0));

    }

    return;
  }

  float x, y, scale, score;
  for (int i = 0; i < layers_; i++)
  {
    BriskLayer& l = pyramid_[i];
    const size_t num = agastPoints[i].size();
    if (i == layers_ - 1)
    {
      for (size_t n = 0; n < num; n++)
      {
        const cv::Point2f& point = agastPoints.at(i)[n].pt;
        // consider only 2D maxima...
        if (!isMax2D(i, (int)point.x, (int)point.y))
          continue;

        bool ismax;
        float dx, dy;
        getScoreMaxBelow(i, (int)point.x, (int)point.y, l.getAgastScore(point.x, point.y, safeThreshold_), ismax, dx, dy);
        if (!ismax)
          continue;

        // get the patch on this layer:
        int s_0_0 = l.getAgastScore(point.x - 1, point.y - 1, 1);
        int s_1_0 = l.getAgastScore(point.x, point.y - 1, 1);
        int s_2_0 = l.getAgastScore(point.x + 1, point.y - 1, 1);
        int s_2_1 = l.getAgastScore(point.x + 1, point.y, 1);
        int s_1_1 = l.getAgastScore(point.x, point.y, 1);
        int s_0_1 = l.getAgastScore(point.x - 1, point.y, 1);
        int s_0_2 = l.getAgastScore(point.x - 1, point.y + 1, 1);
        int s_1_2 = l.getAgastScore(point.x, point.y + 1, 1);
        int s_2_2 = l.getAgastScore(point.x + 1, point.y + 1, 1);
        float delta_x, delta_y;
        float max = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, delta_x, delta_y);

        // store:
        keypoints.push_back(
            cv::KeyPoint((float(point.x) + delta_x) * l.scale() + l.offset(),
                         (float(point.y) + delta_y) * l.scale() + l.offset(), basicSize_ * l.scale(), -1, max, i));
      }
    }
    else
    {
      // not the last layer:
      for (size_t n = 0; n < num; n++)
      {
        const cv::Point2f& point = agastPoints.at(i)[n].pt;

        // first check if it is a maximum:
        if (!isMax2D(i, (int)point.x, (int)point.y))
          continue;

        // let's do the subpixel and float scale refinement:
        bool ismax=false;
        score = refine3D(i, (int)point.x, (int)point.y, x, y, scale, ismax);
        if (!ismax)
        {
          continue;
        }

        // finally store the detected keypoint:
        if (score > float(threshold_))
        {
          keypoints.push_back(cv::KeyPoint(x, y, basicSize_ * scale, -1, score, i));
        }
      }
    }
  }
}

// interpolated score access with recalculation when needed:
inline int
BriskScaleSpace::getScoreAbove(const int layer, const int x_layer, const int y_layer) const
{
  assert(layer<layers_-1);
  const BriskLayer& l = pyramid_[layer + 1];
  if (layer % 2 == 0)
  { // octave
    const int sixths_x = 4 * x_layer - 1;
    const int x_above = sixths_x / 6;
    const int sixths_y = 4 * y_layer - 1;
    const int y_above = sixths_y / 6;
    const int r_x = (sixths_x % 6);
    const int r_x_1 = 6 - r_x;
    const int r_y = (sixths_y % 6);
    const int r_y_1 = 6 - r_y;
    uchar score = 0xFF
        & ((r_x_1 * r_y_1 * l.getAgastScore(x_above, y_above, 1) + r_x * r_y_1
                                                                   * l.getAgastScore(x_above + 1, y_above, 1)
            + r_x_1 * r_y * l.getAgastScore(x_above, y_above + 1, 1)
            + r_x * r_y * l.getAgastScore(x_above + 1, y_above + 1, 1) + 18)
           / 36);

    return score;
  }
  else
  { // intra
    const int eighths_x = 6 * x_layer - 1;
    const int x_above = eighths_x / 8;
    const int eighths_y = 6 * y_layer - 1;
    const int y_above = eighths_y / 8;
    const int r_x = (eighths_x % 8);
    const int r_x_1 = 8 - r_x;
    const int r_y = (eighths_y % 8);
    const int r_y_1 = 8 - r_y;
    uchar score = 0xFF
        & ((r_x_1 * r_y_1 * l.getAgastScore(x_above, y_above, 1) + r_x * r_y_1
                                                                   * l.getAgastScore(x_above + 1, y_above, 1)
            + r_x_1 * r_y * l.getAgastScore(x_above, y_above + 1, 1)
            + r_x * r_y * l.getAgastScore(x_above + 1, y_above + 1, 1) + 32)
           / 64);
    return score;
  }
}
inline int
BriskScaleSpace::getScoreBelow(const int layer, const int x_layer, const int y_layer) const
{
  assert(layer);
  const BriskLayer& l = pyramid_[layer - 1];
  int sixth_x;
  int quarter_x;
  float xf;
  int sixth_y;
  int quarter_y;
  float yf;

  // scaling:
  float offs;
  float area;
  int scaling;
  int scaling2;

  if (layer % 2 == 0)
  { // octave
    sixth_x = 8 * x_layer + 1;
    xf = float(sixth_x) / 6.0f;
    sixth_y = 8 * y_layer + 1;
    yf = float(sixth_y) / 6.0f;

    // scaling:
    offs = 2.0f / 3.0f;
    area = 4.0f * offs * offs;
    scaling = (int)(4194304.0 / area);
    scaling2 = (int)(float(scaling) * area);
  }
  else
  {
    quarter_x = 6 * x_layer + 1;
    xf = float(quarter_x) / 4.0f;
    quarter_y = 6 * y_layer + 1;
    yf = float(quarter_y) / 4.0f;

    // scaling:
    offs = 3.0f / 4.0f;
    area = 4.0f * offs * offs;
    scaling = (int)(4194304.0 / area);
    scaling2 = (int)(float(scaling) * area);
  }

  // calculate borders
  const float x_1 = xf - offs;
  const float x1 = xf + offs;
  const float y_1 = yf - offs;
  const float y1 = yf + offs;

  const int x_left = int(x_1 + 0.5);
  const int y_top = int(y_1 + 0.5);
  const int x_right = int(x1 + 0.5);
  const int y_bottom = int(y1 + 0.5);

  // overlap area - multiplication factors:
  const float r_x_1 = float(x_left) - x_1 + 0.5f;
  const float r_y_1 = float(y_top) - y_1 + 0.5f;
  const float r_x1 = x1 - float(x_right) + 0.5f;
  const float r_y1 = y1 - float(y_bottom) + 0.5f;
  const int dx = x_right - x_left - 1;
  const int dy = y_bottom - y_top - 1;
  const int A = (int)((r_x_1 * r_y_1) * scaling);
  const int B = (int)((r_x1 * r_y_1) * scaling);
  const int C = (int)((r_x1 * r_y1) * scaling);
  const int D = (int)((r_x_1 * r_y1) * scaling);
  const int r_x_1_i = (int)(r_x_1 * scaling);
  const int r_y_1_i = (int)(r_y_1 * scaling);
  const int r_x1_i = (int)(r_x1 * scaling);
  const int r_y1_i = (int)(r_y1 * scaling);

  // first row:
  int ret_val = A * int(l.getAgastScore(x_left, y_top, 1));
  for (int X = 1; X <= dx; X++)
  {
    ret_val += r_y_1_i * int(l.getAgastScore(x_left + X, y_top, 1));
  }
  ret_val += B * int(l.getAgastScore(x_left + dx + 1, y_top, 1));
  // middle ones:
  for (int Y = 1; Y <= dy; Y++)
  {
    ret_val += r_x_1_i * int(l.getAgastScore(x_left, y_top + Y, 1));

    for (int X = 1; X <= dx; X++)
    {
      ret_val += int(l.getAgastScore(x_left + X, y_top + Y, 1)) * scaling;
    }
    ret_val += r_x1_i * int(l.getAgastScore(x_left + dx + 1, y_top + Y, 1));
  }
  // last row:
  ret_val += D * int(l.getAgastScore(x_left, y_top + dy + 1, 1));
  for (int X = 1; X <= dx; X++)
  {
    ret_val += r_y1_i * int(l.getAgastScore(x_left + X, y_top + dy + 1, 1));
  }
  ret_val += C * int(l.getAgastScore(x_left + dx + 1, y_top + dy + 1, 1));

  return ((ret_val + scaling2 / 2) / scaling2);
}

inline bool
BriskScaleSpace::isMax2D(const int layer, const int x_layer, const int y_layer)
{
  const cv::Mat& scores = pyramid_[layer].scores();
  const int scorescols = scores.cols;
  uchar* data = scores.data + y_layer * scorescols + x_layer;
  // decision tree:
  const uchar center = (*data);
  data--;
  const uchar s_10 = *data;
  if (center < s_10)
    return false;
  data += 2;
  const uchar s10 = *data;
  if (center < s10)
    return false;
  data -= (scorescols + 1);
  const uchar s0_1 = *data;
  if (center < s0_1)
    return false;
  data += 2 * scorescols;
  const uchar s01 = *data;
  if (center < s01)
    return false;
  data--;
  const uchar s_11 = *data;
  if (center < s_11)
    return false;
  data += 2;
  const uchar s11 = *data;
  if (center < s11)
    return false;
  data -= 2 * scorescols;
  const uchar s1_1 = *data;
  if (center < s1_1)
    return false;
  data -= 2;
  const uchar s_1_1 = *data;
  if (center < s_1_1)
    return false;

  // reject neighbor maxima
  std::vector<int> delta;
  // put together a list of 2d-offsets to where the maximum is also reached
  if (center == s_1_1)
  {
    delta.push_back(-1);
    delta.push_back(-1);
  }
  if (center == s0_1)
  {
    delta.push_back(0);
    delta.push_back(-1);
  }
  if (center == s1_1)
  {
    delta.push_back(1);
    delta.push_back(-1);
  }
  if (center == s_10)
  {
    delta.push_back(-1);
    delta.push_back(0);
  }
  if (center == s10)
  {
    delta.push_back(1);
    delta.push_back(0);
  }
  if (center == s_11)
  {
    delta.push_back(-1);
    delta.push_back(1);
  }
  if (center == s01)
  {
    delta.push_back(0);
    delta.push_back(1);
  }
  if (center == s11)
  {
    delta.push_back(1);
    delta.push_back(1);
  }
  const unsigned int deltasize = (unsigned int)delta.size();
  if (deltasize != 0)
  {
    // in this case, we have to analyze the situation more carefully:
    // the values are gaussian blurred and then we really decide
    data = scores.data + y_layer * scorescols + x_layer;
    int smoothedcenter = 4 * center + 2 * (s_10 + s10 + s0_1 + s01) + s_1_1 + s1_1 + s_11 + s11;
    for (unsigned int i = 0; i < deltasize; i += 2)
    {
      data = scores.data + (y_layer - 1 + delta[i + 1]) * scorescols + x_layer + delta[i] - 1;
      int othercenter = *data;
      data++;
      othercenter += 2 * (*data);
      data++;
      othercenter += *data;
      data += scorescols;
      othercenter += 2 * (*data);
      data--;
      othercenter += 4 * (*data);
      data--;
      othercenter += 2 * (*data);
      data += scorescols;
      othercenter += *data;
      data++;
      othercenter += 2 * (*data);
      data++;
      othercenter += *data;
      if (othercenter > smoothedcenter)
        return false;
    }
  }
  return true;
}

// 3D maximum refinement centered around (x_layer,y_layer)
inline float
BriskScaleSpace::refine3D(const int layer, const int x_layer, const int y_layer, float& x, float& y, float& scale,
                          bool& ismax) const
{
  ismax = true;
  const BriskLayer& thisLayer = pyramid_[layer];
  const int center = thisLayer.getAgastScore(x_layer, y_layer, 1);

  // check and get above maximum:
  float delta_x_above = 0, delta_y_above = 0;
  float max_above = getScoreMaxAbove(layer, x_layer, y_layer, center, ismax, delta_x_above, delta_y_above);

  if (!ismax)
    return 0.0f;

  float max; // to be returned

  if (layer % 2 == 0)
  { // on octave
    // treat the patch below:
    float delta_x_below, delta_y_below;
    float max_below_float;
    int max_below = 0;
    if (layer == 0)
    {
      // guess the lower intra octave...
      const BriskLayer& l = pyramid_[0];
      int s_0_0 = l.getAgastScore_5_8(x_layer - 1, y_layer - 1, 1);
      max_below = s_0_0;
      int s_1_0 = l.getAgastScore_5_8(x_layer, y_layer - 1, 1);
      max_below = std::max(s_1_0, max_below);
      int s_2_0 = l.getAgastScore_5_8(x_layer + 1, y_layer - 1, 1);
      max_below = std::max(s_2_0, max_below);
      int s_2_1 = l.getAgastScore_5_8(x_layer + 1, y_layer, 1);
      max_below = std::max(s_2_1, max_below);
      int s_1_1 = l.getAgastScore_5_8(x_layer, y_layer, 1);
      max_below = std::max(s_1_1, max_below);
      int s_0_1 = l.getAgastScore_5_8(x_layer - 1, y_layer, 1);
      max_below = std::max(s_0_1, max_below);
      int s_0_2 = l.getAgastScore_5_8(x_layer - 1, y_layer + 1, 1);
      max_below = std::max(s_0_2, max_below);
      int s_1_2 = l.getAgastScore_5_8(x_layer, y_layer + 1, 1);
      max_below = std::max(s_1_2, max_below);
      int s_2_2 = l.getAgastScore_5_8(x_layer + 1, y_layer + 1, 1);
      max_below = std::max(s_2_2, max_below);

      max_below_float = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, delta_x_below,
                                   delta_y_below);
      max_below_float = (float)max_below;
    }
    else
    {
      max_below_float = getScoreMaxBelow(layer, x_layer, y_layer, center, ismax, delta_x_below, delta_y_below);
      if (!ismax)
        return 0;
    }

    // get the patch on this layer:
    int s_0_0 = thisLayer.getAgastScore(x_layer - 1, y_layer - 1, 1);
    int s_1_0 = thisLayer.getAgastScore(x_layer, y_layer - 1, 1);
    int s_2_0 = thisLayer.getAgastScore(x_layer + 1, y_layer - 1, 1);
    int s_2_1 = thisLayer.getAgastScore(x_layer + 1, y_layer, 1);
    int s_1_1 = thisLayer.getAgastScore(x_layer, y_layer, 1);
    int s_0_1 = thisLayer.getAgastScore(x_layer - 1, y_layer, 1);
    int s_0_2 = thisLayer.getAgastScore(x_layer - 1, y_layer + 1, 1);
    int s_1_2 = thisLayer.getAgastScore(x_layer, y_layer + 1, 1);
    int s_2_2 = thisLayer.getAgastScore(x_layer + 1, y_layer + 1, 1);
    float delta_x_layer, delta_y_layer;
    float max_layer = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, delta_x_layer,
                                 delta_y_layer);

    // calculate the relative scale (1D maximum):
    if (layer == 0)
    {
      scale = refine1D_2(max_below_float, std::max(float(center), max_layer), max_above, max);
    }
    else
      scale = refine1D(max_below_float, std::max(float(center), max_layer), max_above, max);

    if (scale > 1.0)
    {
      // interpolate the position:
      const float r0 = (1.5f - scale) / .5f;
      const float r1 = 1.0f - r0;
      x = (r0 * delta_x_layer + r1 * delta_x_above + float(x_layer)) * thisLayer.scale() + thisLayer.offset();
      y = (r0 * delta_y_layer + r1 * delta_y_above + float(y_layer)) * thisLayer.scale() + thisLayer.offset();
    }
    else
    {
      if (layer == 0)
      {
        // interpolate the position:
        const float r0 = (scale - 0.5f) / 0.5f;
        const float r_1 = 1.0f - r0;
        x = r0 * delta_x_layer + r_1 * delta_x_below + float(x_layer);
        y = r0 * delta_y_layer + r_1 * delta_y_below + float(y_layer);
      }
      else
      {
        // interpolate the position:
        const float r0 = (scale - 0.75f) / 0.25f;
        const float r_1 = 1.0f - r0;
        x = (r0 * delta_x_layer + r_1 * delta_x_below + float(x_layer)) * thisLayer.scale() + thisLayer.offset();
        y = (r0 * delta_y_layer + r_1 * delta_y_below + float(y_layer)) * thisLayer.scale() + thisLayer.offset();
      }
    }
  }
  else
  {
    // on intra
    // check the patch below:
    float delta_x_below, delta_y_below;
    float max_below = getScoreMaxBelow(layer, x_layer, y_layer, center, ismax, delta_x_below, delta_y_below);
    if (!ismax)
      return 0.0f;

    // get the patch on this layer:
    int s_0_0 = thisLayer.getAgastScore(x_layer - 1, y_layer - 1, 1);
    int s_1_0 = thisLayer.getAgastScore(x_layer, y_layer - 1, 1);
    int s_2_0 = thisLayer.getAgastScore(x_layer + 1, y_layer - 1, 1);
    int s_2_1 = thisLayer.getAgastScore(x_layer + 1, y_layer, 1);
    int s_1_1 = thisLayer.getAgastScore(x_layer, y_layer, 1);
    int s_0_1 = thisLayer.getAgastScore(x_layer - 1, y_layer, 1);
    int s_0_2 = thisLayer.getAgastScore(x_layer - 1, y_layer + 1, 1);
    int s_1_2 = thisLayer.getAgastScore(x_layer, y_layer + 1, 1);
    int s_2_2 = thisLayer.getAgastScore(x_layer + 1, y_layer + 1, 1);
    float delta_x_layer, delta_y_layer;
    float max_layer = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, delta_x_layer,
                                 delta_y_layer);

    // calculate the relative scale (1D maximum):
    scale = refine1D_1(max_below, std::max(float(center), max_layer), max_above, max);
    if (scale > 1.0)
    {
      // interpolate the position:
      const float r0 = 4.0f - scale * 3.0f;
      const float r1 = 1.0f - r0;
      x = (r0 * delta_x_layer + r1 * delta_x_above + float(x_layer)) * thisLayer.scale() + thisLayer.offset();
      y = (r0 * delta_y_layer + r1 * delta_y_above + float(y_layer)) * thisLayer.scale() + thisLayer.offset();
    }
    else
    {
      // interpolate the position:
      const float r0 = scale * 3.0f - 2.0f;
      const float r_1 = 1.0f - r0;
      x = (r0 * delta_x_layer + r_1 * delta_x_below + float(x_layer)) * thisLayer.scale() + thisLayer.offset();
      y = (r0 * delta_y_layer + r_1 * delta_y_below + float(y_layer)) * thisLayer.scale() + thisLayer.offset();
    }
  }

  // calculate the absolute scale:
  scale *= thisLayer.scale();

  // that's it, return the refined maximum:
  return max;
}

// return the maximum of score patches above or below
inline float
BriskScaleSpace::getScoreMaxAbove(const int layer, const int x_layer, const int y_layer, const int threshold,
                                  bool& ismax, float& dx, float& dy) const
{

  ismax = false;
  // relevant floating point coordinates
  float x_1;
  float x1;
  float y_1;
  float y1;

  // the layer above
  assert(layer+1<layers_);
  const BriskLayer& layerAbove = pyramid_[layer + 1];

  if (layer % 2 == 0)
  {
    // octave
    x_1 = float(4 * (x_layer) - 1 - 2) / 6.0f;
    x1 = float(4 * (x_layer) - 1 + 2) / 6.0f;
    y_1 = float(4 * (y_layer) - 1 - 2) / 6.0f;
    y1 = float(4 * (y_layer) - 1 + 2) / 6.0f;
  }
  else
  {
    // intra
    x_1 = float(6 * (x_layer) - 1 - 3) / 8.0f;
    x1 = float(6 * (x_layer) - 1 + 3) / 8.0f;
    y_1 = float(6 * (y_layer) - 1 - 3) / 8.0f;
    y1 = float(6 * (y_layer) - 1 + 3) / 8.0f;
  }

  // check the first row
  int max_x = (int)x_1 + 1;
  int max_y = (int)y_1 + 1;
  float tmp_max;
  float maxval = (float)layerAbove.getAgastScore(x_1, y_1, 1);
  if (maxval > threshold)
    return 0;
  for (int x = (int)x_1 + 1; x <= int(x1); x++)
  {
    tmp_max = (float)layerAbove.getAgastScore(float(x), y_1, 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > maxval)
    {
      maxval = tmp_max;
      max_x = x;
    }
  }
  tmp_max = (float)layerAbove.getAgastScore(x1, y_1, 1);
  if (tmp_max > threshold)
    return 0;
  if (tmp_max > maxval)
  {
    maxval = tmp_max;
    max_x = int(x1);
  }

  // middle rows
  for (int y = (int)y_1 + 1; y <= int(y1); y++)
  {
    tmp_max = (float)layerAbove.getAgastScore(x_1, float(y), 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > maxval)
    {
      maxval = tmp_max;
      max_x = int(x_1 + 1);
      max_y = y;
    }
    for (int x = (int)x_1 + 1; x <= int(x1); x++)
    {
      tmp_max = (float)layerAbove.getAgastScore(x, y, 1);
      if (tmp_max > threshold)
        return 0;
      if (tmp_max > maxval)
      {
        maxval = tmp_max;
        max_x = x;
        max_y = y;
      }
    }
    tmp_max = (float)layerAbove.getAgastScore(x1, float(y), 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > maxval)
    {
      maxval = tmp_max;
      max_x = int(x1);
      max_y = y;
    }
  }

  // bottom row
  tmp_max = (float)layerAbove.getAgastScore(x_1, y1, 1);
  if (tmp_max > maxval)
  {
    maxval = tmp_max;
    max_x = int(x_1 + 1);
    max_y = int(y1);
  }
  for (int x = (int)x_1 + 1; x <= int(x1); x++)
  {
    tmp_max = (float)layerAbove.getAgastScore(float(x), y1, 1);
    if (tmp_max > maxval)
    {
      maxval = tmp_max;
      max_x = x;
      max_y = int(y1);
    }
  }
  tmp_max = (float)layerAbove.getAgastScore(x1, y1, 1);
  if (tmp_max > maxval)
  {
    maxval = tmp_max;
    max_x = int(x1);
    max_y = int(y1);
  }

  //find dx/dy:
  int s_0_0 = layerAbove.getAgastScore(max_x - 1, max_y - 1, 1);
  int s_1_0 = layerAbove.getAgastScore(max_x, max_y - 1, 1);
  int s_2_0 = layerAbove.getAgastScore(max_x + 1, max_y - 1, 1);
  int s_2_1 = layerAbove.getAgastScore(max_x + 1, max_y, 1);
  int s_1_1 = layerAbove.getAgastScore(max_x, max_y, 1);
  int s_0_1 = layerAbove.getAgastScore(max_x - 1, max_y, 1);
  int s_0_2 = layerAbove.getAgastScore(max_x - 1, max_y + 1, 1);
  int s_1_2 = layerAbove.getAgastScore(max_x, max_y + 1, 1);
  int s_2_2 = layerAbove.getAgastScore(max_x + 1, max_y + 1, 1);
  float dx_1, dy_1;
  float refined_max = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, dx_1, dy_1);

  // calculate dx/dy in above coordinates
  float real_x = float(max_x) + dx_1;
  float real_y = float(max_y) + dy_1;
  bool returnrefined = true;
  if (layer % 2 == 0)
  {
    dx = (real_x * 6.0f + 1.0f) / 4.0f - float(x_layer);
    dy = (real_y * 6.0f + 1.0f) / 4.0f - float(y_layer);
  }
  else
  {
    dx = (real_x * 8.0f + 1.0f) / 6.0f - float(x_layer);
    dy = (real_y * 8.0f + 1.0f) / 6.0f - float(y_layer);
  }

  // saturate
  if (dx > 1.0f)
  {
    dx = 1.0f;
    returnrefined = false;
  }
  if (dx < -1.0f)
  {
    dx = -1.0f;
    returnrefined = false;
  }
  if (dy > 1.0f)
  {
    dy = 1.0f;
    returnrefined = false;
  }
  if (dy < -1.0f)
  {
    dy = -1.0f;
    returnrefined = false;
  }

  // done and ok.
  ismax = true;
  if (returnrefined)
  {
    return std::max(refined_max, maxval);
  }
  return maxval;
}

inline float
BriskScaleSpace::getScoreMaxBelow(const int layer, const int x_layer, const int y_layer, const int threshold,
                                  bool& ismax, float& dx, float& dy) const
{
  ismax = false;

  // relevant floating point coordinates
  float x_1;
  float x1;
  float y_1;
  float y1;

  if (layer % 2 == 0)
  {
    // octave
    x_1 = float(8 * (x_layer) + 1 - 4) / 6.0f;
    x1 = float(8 * (x_layer) + 1 + 4) / 6.0f;
    y_1 = float(8 * (y_layer) + 1 - 4) / 6.0f;
    y1 = float(8 * (y_layer) + 1 + 4) / 6.0f;
  }
  else
  {
    x_1 = float(6 * (x_layer) + 1 - 3) / 4.0f;
    x1 = float(6 * (x_layer) + 1 + 3) / 4.0f;
    y_1 = float(6 * (y_layer) + 1 - 3) / 4.0f;
    y1 = float(6 * (y_layer) + 1 + 3) / 4.0f;
  }

  // the layer below
  assert(layer>0);
  const BriskLayer& layerBelow = pyramid_[layer - 1];

  // check the first row
  int max_x = (int)x_1 + 1;
  int max_y = (int)y_1 + 1;
  float tmp_max;
  float max = (float)layerBelow.getAgastScore(x_1, y_1, 1);
  if (max > threshold)
    return 0;
  for (int x = (int)x_1 + 1; x <= int(x1); x++)
  {
    tmp_max = (float)layerBelow.getAgastScore(float(x), y_1, 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > max)
    {
      max = tmp_max;
      max_x = x;
    }
  }
  tmp_max = (float)layerBelow.getAgastScore(x1, y_1, 1);
  if (tmp_max > threshold)
    return 0;
  if (tmp_max > max)
  {
    max = tmp_max;
    max_x = int(x1);
  }

  // middle rows
  for (int y = (int)y_1 + 1; y <= int(y1); y++)
  {
    tmp_max = (float)layerBelow.getAgastScore(x_1, float(y), 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > max)
    {
      max = tmp_max;
      max_x = int(x_1 + 1);
      max_y = y;
    }
    for (int x = (int)x_1 + 1; x <= int(x1); x++)
    {
      tmp_max = (float)layerBelow.getAgastScore(x, y, 1);
      if (tmp_max > threshold)
        return 0;
      if (tmp_max == max)
      {
        const int t1 = 2
            * (layerBelow.getAgastScore(x - 1, y, 1) + layerBelow.getAgastScore(x + 1, y, 1)
               + layerBelow.getAgastScore(x, y + 1, 1) + layerBelow.getAgastScore(x, y - 1, 1))
                       + (layerBelow.getAgastScore(x + 1, y + 1, 1) + layerBelow.getAgastScore(x - 1, y + 1, 1)
                          + layerBelow.getAgastScore(x + 1, y - 1, 1) + layerBelow.getAgastScore(x - 1, y - 1, 1));
        const int t2 = 2
            * (layerBelow.getAgastScore(max_x - 1, max_y, 1) + layerBelow.getAgastScore(max_x + 1, max_y, 1)
               + layerBelow.getAgastScore(max_x, max_y + 1, 1) + layerBelow.getAgastScore(max_x, max_y - 1, 1))
                       + (layerBelow.getAgastScore(max_x + 1, max_y + 1, 1) + layerBelow.getAgastScore(max_x - 1,
                                                                                                       max_y + 1, 1)
                          + layerBelow.getAgastScore(max_x + 1, max_y - 1, 1)
                          + layerBelow.getAgastScore(max_x - 1, max_y - 1, 1));
        if (t1 > t2)
        {
          max_x = x;
          max_y = y;
        }
      }
      if (tmp_max > max)
      {
        max = tmp_max;
        max_x = x;
        max_y = y;
      }
    }
    tmp_max = (float)layerBelow.getAgastScore(x1, float(y), 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > max)
    {
      max = tmp_max;
      max_x = int(x1);
      max_y = y;
    }
  }

  // bottom row
  tmp_max = (float)layerBelow.getAgastScore(x_1, y1, 1);
  if (tmp_max > max)
  {
    max = tmp_max;
    max_x = int(x_1 + 1);
    max_y = int(y1);
  }
  for (int x = (int)x_1 + 1; x <= int(x1); x++)
  {
    tmp_max = (float)layerBelow.getAgastScore(float(x), y1, 1);
    if (tmp_max > max)
    {
      max = tmp_max;
      max_x = x;
      max_y = int(y1);
    }
  }
  tmp_max = (float)layerBelow.getAgastScore(x1, y1, 1);
  if (tmp_max > max)
  {
    max = tmp_max;
    max_x = int(x1);
    max_y = int(y1);
  }

  //find dx/dy:
  int s_0_0 = layerBelow.getAgastScore(max_x - 1, max_y - 1, 1);
  int s_1_0 = layerBelow.getAgastScore(max_x, max_y - 1, 1);
  int s_2_0 = layerBelow.getAgastScore(max_x + 1, max_y - 1, 1);
  int s_2_1 = layerBelow.getAgastScore(max_x + 1, max_y, 1);
  int s_1_1 = layerBelow.getAgastScore(max_x, max_y, 1);
  int s_0_1 = layerBelow.getAgastScore(max_x - 1, max_y, 1);
  int s_0_2 = layerBelow.getAgastScore(max_x - 1, max_y + 1, 1);
  int s_1_2 = layerBelow.getAgastScore(max_x, max_y + 1, 1);
  int s_2_2 = layerBelow.getAgastScore(max_x + 1, max_y + 1, 1);
  float dx_1, dy_1;
  float refined_max = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, dx_1, dy_1);

  // calculate dx/dy in above coordinates
  float real_x = float(max_x) + dx_1;
  float real_y = float(max_y) + dy_1;
  bool returnrefined = true;
  if (layer % 2 == 0)
  {
    dx = (float)((real_x * 6.0 + 1.0) / 8.0) - float(x_layer);
    dy = (float)((real_y * 6.0 + 1.0) / 8.0) - float(y_layer);
  }
  else
  {
    dx = (float)((real_x * 4.0 - 1.0) / 6.0) - float(x_layer);
    dy = (float)((real_y * 4.0 - 1.0) / 6.0) - float(y_layer);
  }

  // saturate
  if (dx > 1.0)
  {
    dx = 1.0f;
    returnrefined = false;
  }
  if (dx < -1.0f)
  {
    dx = -1.0f;
    returnrefined = false;
  }
  if (dy > 1.0f)
  {
    dy = 1.0f;
    returnrefined = false;
  }
  if (dy < -1.0f)
  {
    dy = -1.0f;
    returnrefined = false;
  }

  // done and ok.
  ismax = true;
  if (returnrefined)
  {
    return std::max(refined_max, max);
  }
  return max;
}

inline float
BriskScaleSpace::refine1D(const float s_05, const float s0, const float s05, float& max) const
{
  int i_05 = int(1024.0 * s_05 + 0.5);
  int i0 = int(1024.0 * s0 + 0.5);
  int i05 = int(1024.0 * s05 + 0.5);

  //   16.0000  -24.0000    8.0000
  //  -40.0000   54.0000  -14.0000
  //   24.0000  -27.0000    6.0000

  int three_a = 16 * i_05 - 24 * i0 + 8 * i05;
  // second derivative must be negative:
  if (three_a >= 0)
  {
    if (s0 >= s_05 && s0 >= s05)
    {
      max = s0;
      return 1.0f;
    }
    if (s_05 >= s0 && s_05 >= s05)
    {
      max = s_05;
      return 0.75f;
    }
    if (s05 >= s0 && s05 >= s_05)
    {
      max = s05;
      return 1.5f;
    }
  }

  int three_b = -40 * i_05 + 54 * i0 - 14 * i05;
  // calculate max location:
  float ret_val = -float(three_b) / float(2 * three_a);
  // saturate and return
  if (ret_val < 0.75)
    ret_val = 0.75;
  else if (ret_val > 1.5)
    ret_val = 1.5; // allow to be slightly off bounds ...?
  int three_c = +24 * i_05 - 27 * i0 + 6 * i05;
  max = float(three_c) + float(three_a) * ret_val * ret_val + float(three_b) * ret_val;
  max /= 3072.0f;
  return ret_val;
}

inline float
BriskScaleSpace::refine1D_1(const float s_05, const float s0, const float s05, float& max) const
{
  int i_05 = int(1024.0 * s_05 + 0.5);
  int i0 = int(1024.0 * s0 + 0.5);
  int i05 = int(1024.0 * s05 + 0.5);

  //  4.5000   -9.0000    4.5000
  //-10.5000   18.0000   -7.5000
  //  6.0000   -8.0000    3.0000

  int two_a = 9 * i_05 - 18 * i0 + 9 * i05;
  // second derivative must be negative:
  if (two_a >= 0)
  {
    if (s0 >= s_05 && s0 >= s05)
    {
      max = s0;
      return 1.0f;
    }
    if (s_05 >= s0 && s_05 >= s05)
    {
      max = s_05;
      return 0.6666666666666666666666666667f;
    }
    if (s05 >= s0 && s05 >= s_05)
    {
      max = s05;
      return 1.3333333333333333333333333333f;
    }
  }

  int two_b = -21 * i_05 + 36 * i0 - 15 * i05;
  // calculate max location:
  float ret_val = -float(two_b) / float(2 * two_a);
  // saturate and return
  if (ret_val < 0.6666666666666666666666666667f)
    ret_val = 0.666666666666666666666666667f;
  else if (ret_val > 1.33333333333333333333333333f)
    ret_val = 1.333333333333333333333333333f;
  int two_c = +12 * i_05 - 16 * i0 + 6 * i05;
  max = float(two_c) + float(two_a) * ret_val * ret_val + float(two_b) * ret_val;
  max /= 2048.0f;
  return ret_val;
}

inline float
BriskScaleSpace::refine1D_2(const float s_05, const float s0, const float s05, float& max) const
{
  int i_05 = int(1024.0 * s_05 + 0.5);
  int i0 = int(1024.0 * s0 + 0.5);
  int i05 = int(1024.0 * s05 + 0.5);

  //   18.0000  -30.0000   12.0000
  //  -45.0000   65.0000  -20.0000
  //   27.0000  -30.0000    8.0000

  int a = 2 * i_05 - 4 * i0 + 2 * i05;
  // second derivative must be negative:
  if (a >= 0)
  {
    if (s0 >= s_05 && s0 >= s05)
    {
      max = s0;
      return 1.0f;
    }
    if (s_05 >= s0 && s_05 >= s05)
    {
      max = s_05;
      return 0.7f;
    }
    if (s05 >= s0 && s05 >= s_05)
    {
      max = s05;
      return 1.5f;
    }
  }

  int b = -5 * i_05 + 8 * i0 - 3 * i05;
  // calculate max location:
  float ret_val = -float(b) / float(2 * a);
  // saturate and return
  if (ret_val < 0.7f)
    ret_val = 0.7f;
  else if (ret_val > 1.5f)
    ret_val = 1.5f; // allow to be slightly off bounds ...?
  int c = +3 * i_05 - 3 * i0 + 1 * i05;
  max = float(c) + float(a) * ret_val * ret_val + float(b) * ret_val;
  max /= 1024;
  return ret_val;
}

inline float
BriskScaleSpace::subpixel2D(const int s_0_0, const int s_0_1, const int s_0_2, const int s_1_0, const int s_1_1,
                            const int s_1_2, const int s_2_0, const int s_2_1, const int s_2_2, float& delta_x,
                            float& delta_y) const
{

  // the coefficients of the 2d quadratic function least-squares fit:
  int tmp1 = s_0_0 + s_0_2 - 2 * s_1_1 + s_2_0 + s_2_2;
  int coeff1 = 3 * (tmp1 + s_0_1 - ((s_1_0 + s_1_2) << 1) + s_2_1);
  int coeff2 = 3 * (tmp1 - ((s_0_1 + s_2_1) << 1) + s_1_0 + s_1_2);
  int tmp2 = s_0_2 - s_2_0;
  int tmp3 = (s_0_0 + tmp2 - s_2_2);
  int tmp4 = tmp3 - 2 * tmp2;
  int coeff3 = -3 * (tmp3 + s_0_1 - s_2_1);
  int coeff4 = -3 * (tmp4 + s_1_0 - s_1_2);
  int coeff5 = (s_0_0 - s_0_2 - s_2_0 + s_2_2) << 2;
  int coeff6 = -(s_0_0 + s_0_2 - ((s_1_0 + s_0_1 + s_1_2 + s_2_1) << 1) - 5 * s_1_1 + s_2_0 + s_2_2) << 1;

  // 2nd derivative test:
  int H_det = 4 * coeff1 * coeff2 - coeff5 * coeff5;

  if (H_det == 0)
  {
    delta_x = 0.0f;
    delta_y = 0.0f;
    return float(coeff6) / 18.0f;
  }

  if (!(H_det > 0 && coeff1 < 0))
  {
    // The maximum must be at the one of the 4 patch corners.
    int tmp_max = coeff3 + coeff4 + coeff5;
    delta_x = 1.0f;
    delta_y = 1.0f;

    int tmp = -coeff3 + coeff4 - coeff5;
    if (tmp > tmp_max)
    {
      tmp_max = tmp;
      delta_x = -1.0f;
      delta_y = 1.0f;
    }
    tmp = coeff3 - coeff4 - coeff5;
    if (tmp > tmp_max)
    {
      tmp_max = tmp;
      delta_x = 1.0f;
      delta_y = -1.0f;
    }
    tmp = -coeff3 - coeff4 + coeff5;
    if (tmp > tmp_max)
    {
      tmp_max = tmp;
      delta_x = -1.0f;
      delta_y = -1.0f;
    }
    return float(tmp_max + coeff1 + coeff2 + coeff6) / 18.0f;
  }

  // this is hopefully the normal outcome of the Hessian test
  delta_x = float(2 * coeff2 * coeff3 - coeff4 * coeff5) / float(-H_det);
  delta_y = float(2 * coeff1 * coeff4 - coeff3 * coeff5) / float(-H_det);
  // TODO: this is not correct, but easy, so perform a real boundary maximum search:
  bool tx = false;
  bool tx_ = false;
  bool ty = false;
  bool ty_ = false;
  if (delta_x > 1.0)
    tx = true;
  else if (delta_x < -1.0)
    tx_ = true;
  if (delta_y > 1.0)
    ty = true;
  if (delta_y < -1.0)
    ty_ = true;

  if (tx || tx_ || ty || ty_)
  {
    // get two candidates:
    float delta_x1 = 0.0f, delta_x2 = 0.0f, delta_y1 = 0.0f, delta_y2 = 0.0f;
    if (tx)
    {
      delta_x1 = 1.0f;
      delta_y1 = -float(coeff4 + coeff5) / float(2 * coeff2);
      if (delta_y1 > 1.0f)
        delta_y1 = 1.0f;
      else if (delta_y1 < -1.0f)
        delta_y1 = -1.0f;
    }
    else if (tx_)
    {
      delta_x1 = -1.0f;
      delta_y1 = -float(coeff4 - coeff5) / float(2 * coeff2);
      if (delta_y1 > 1.0f)
        delta_y1 = 1.0f;
      else if (delta_y1 < -1.0)
        delta_y1 = -1.0f;
    }
    if (ty)
    {
      delta_y2 = 1.0f;
      delta_x2 = -float(coeff3 + coeff5) / float(2 * coeff1);
      if (delta_x2 > 1.0f)
        delta_x2 = 1.0f;
      else if (delta_x2 < -1.0f)
        delta_x2 = -1.0f;
    }
    else if (ty_)
    {
      delta_y2 = -1.0f;
      delta_x2 = -float(coeff3 - coeff5) / float(2 * coeff1);
      if (delta_x2 > 1.0f)
        delta_x2 = 1.0f;
      else if (delta_x2 < -1.0f)
        delta_x2 = -1.0f;
    }
    // insert both options for evaluation which to pick
    float max1 = (coeff1 * delta_x1 * delta_x1 + coeff2 * delta_y1 * delta_y1 + coeff3 * delta_x1 + coeff4 * delta_y1
                  + coeff5 * delta_x1 * delta_y1 + coeff6)
                 / 18.0f;
    float max2 = (coeff1 * delta_x2 * delta_x2 + coeff2 * delta_y2 * delta_y2 + coeff3 * delta_x2 + coeff4 * delta_y2
                  + coeff5 * delta_x2 * delta_y2 + coeff6)
                 / 18.0f;
    if (max1 > max2)
    {
      delta_x = delta_x1;
      delta_y = delta_x1;
      return max1;
    }
    else
    {
      delta_x = delta_x2;
      delta_y = delta_x2;
      return max2;
    }
  }

  // this is the case of the maximum inside the boundaries:
  return (coeff1 * delta_x * delta_x + coeff2 * delta_y * delta_y + coeff3 * delta_x + coeff4 * delta_y
          + coeff5 * delta_x * delta_y + coeff6)
         / 18.0f;
}

// construct a layer
BriskLayer::BriskLayer(const cv::Mat& img_in, float scale_in, float offset_in)
{
  img_ = img_in;
  scores_ = cv::Mat_<uchar>::zeros(img_in.rows, img_in.cols);
  // attention: this means that the passed image reference must point to persistent memory
  scale_ = scale_in;
  offset_ = offset_in;
  // create an agast detector
  fast_9_16_ = new FastFeatureDetector2(1, true, FastFeatureDetector::TYPE_9_16);
  makeOffsets(pixel_5_8_, (int)img_.step, 8);
  makeOffsets(pixel_9_16_, (int)img_.step, 16);
}
// derive a layer
BriskLayer::BriskLayer(const BriskLayer& layer, int mode)
{
  if (mode == CommonParams::HALFSAMPLE)
  {
    img_.create(layer.img().rows / 2, layer.img().cols / 2, CV_8U);
    halfsample(layer.img(), img_);
    scale_ = layer.scale() * 2;
    offset_ = 0.5f * scale_ - 0.5f;
  }
  else
  {
    img_.create(2 * (layer.img().rows / 3), 2 * (layer.img().cols / 3), CV_8U);
    twothirdsample(layer.img(), img_);
    scale_ = layer.scale() * 1.5f;
    offset_ = 0.5f * scale_ - 0.5f;
  }
  scores_ = cv::Mat::zeros(img_.rows, img_.cols, CV_8U);
  fast_9_16_ = new FastFeatureDetector2(1, false, FastFeatureDetector::TYPE_9_16);
  makeOffsets(pixel_5_8_, (int)img_.step, 8);
  makeOffsets(pixel_9_16_, (int)img_.step, 16);
}

// Fast/Agast
// wraps the agast class
void
BriskLayer::getAgastPoints(int threshold, std::vector<KeyPoint>& keypoints)
{
  fast_9_16_->set("threshold", threshold);
  fast_9_16_->detect(img_, keypoints);

  // also write scores
  const size_t num = keypoints.size();

  for (size_t i = 0; i < num; i++)
    scores_((int)keypoints[i].pt.y, (int)keypoints[i].pt.x) = saturate_cast<uchar>(keypoints[i].response);
}

inline int
BriskLayer::getAgastScore(int x, int y, int threshold) const
{
  if (x < 3 || y < 3)
    return 0;
  if (x >= img_.cols - 3 || y >= img_.rows - 3)
    return 0;
  uchar& score = (uchar&)scores_(y, x);
  if (score > 2)
  {
    return score;
  }
  score = (uchar)cornerScore<16>(&img_.at<uchar>(y, x), pixel_9_16_, threshold - 1);
  if (score < threshold)
    score = 0;
  return score;
}

inline int
BriskLayer::getAgastScore_5_8(int x, int y, int threshold) const
{
  if (x < 2 || y < 2)
    return 0;
  if (x >= img_.cols - 2 || y >= img_.rows - 2)
    return 0;
  int score = cornerScore<8>(&img_.at<uchar>(y, x), pixel_5_8_, threshold - 1);
  if (score < threshold)
    score = 0;
  return score;
}

inline int
BriskLayer::getAgastScore(float xf, float yf, int threshold_in, float scale_in) const
{
  if (scale_in <= 1.0f)
  {
    // just do an interpolation inside the layer
    const int x = int(xf);
    const float rx1 = xf - float(x);
    const float rx = 1.0f - rx1;
    const int y = int(yf);
    const float ry1 = yf - float(y);
    const float ry = 1.0f - ry1;

    return (uchar)(rx * ry * getAgastScore(x, y, threshold_in) + rx1 * ry * getAgastScore(x + 1, y, threshold_in)
           + rx * ry1 * getAgastScore(x, y + 1, threshold_in) + rx1 * ry1 * getAgastScore(x + 1, y + 1, threshold_in));
  }
  else
  {
    // this means we overlap area smoothing
    const float halfscale = scale_in / 2.0f;
    // get the scores first:
    for (int x = int(xf - halfscale); x <= int(xf + halfscale + 1.0f); x++)
    {
      for (int y = int(yf - halfscale); y <= int(yf + halfscale + 1.0f); y++)
      {
        getAgastScore(x, y, threshold_in);
      }
    }
    // get the smoothed value
    return value(scores_, xf, yf, scale_in);
  }
}

// access gray values (smoothed/interpolated)
inline int
BriskLayer::value(const cv::Mat& mat, float xf, float yf, float scale_in) const
{
  assert(!mat.empty());
  // get the position
  const int x = cvFloor(xf);
  const int y = cvFloor(yf);
  const cv::Mat& image = mat;
  const int& imagecols = image.cols;

  // get the sigma_half:
  const float sigma_half = scale_in / 2;
  const float area = 4.0f * sigma_half * sigma_half;
  // calculate output:
  int ret_val;
  if (sigma_half < 0.5)
  {
    //interpolation multipliers:
    const int r_x = (int)((xf - x) * 1024);
    const int r_y = (int)((yf - y) * 1024);
    const int r_x_1 = (1024 - r_x);
    const int r_y_1 = (1024 - r_y);
    uchar* ptr = image.data + x + y * imagecols;
    // just interpolate:
    ret_val = (r_x_1 * r_y_1 * int(*ptr));
    ptr++;
    ret_val += (r_x * r_y_1 * int(*ptr));
    ptr += imagecols;
    ret_val += (r_x * r_y * int(*ptr));
    ptr--;
    ret_val += (r_x_1 * r_y * int(*ptr));
    return 0xFF & ((ret_val + 512) / 1024 / 1024);
  }

  // this is the standard case (simple, not speed optimized yet):

  // scaling:
  const int scaling = (int)(4194304.0f / area);
  const int scaling2 = (int)(float(scaling) * area / 1024.0f);

  // calculate borders
  const float x_1 = xf - sigma_half;
  const float x1 = xf + sigma_half;
  const float y_1 = yf - sigma_half;
  const float y1 = yf + sigma_half;

  const int x_left = int(x_1 + 0.5);
  const int y_top = int(y_1 + 0.5);
  const int x_right = int(x1 + 0.5);
  const int y_bottom = int(y1 + 0.5);

  // overlap area - multiplication factors:
  const float r_x_1 = float(x_left) - x_1 + 0.5f;
  const float r_y_1 = float(y_top) - y_1 + 0.5f;
  const float r_x1 = x1 - float(x_right) + 0.5f;
  const float r_y1 = y1 - float(y_bottom) + 0.5f;
  const int dx = x_right - x_left - 1;
  const int dy = y_bottom - y_top - 1;
  const int A = (int)((r_x_1 * r_y_1) * scaling);
  const int B = (int)((r_x1 * r_y_1) * scaling);
  const int C = (int)((r_x1 * r_y1) * scaling);
  const int D = (int)((r_x_1 * r_y1) * scaling);
  const int r_x_1_i = (int)(r_x_1 * scaling);
  const int r_y_1_i = (int)(r_y_1 * scaling);
  const int r_x1_i = (int)(r_x1 * scaling);
  const int r_y1_i = (int)(r_y1 * scaling);

  // now the calculation:
  uchar* ptr = image.data + x_left + imagecols * y_top;
  // first row:
  ret_val = A * int(*ptr);
  ptr++;
  const uchar* end1 = ptr + dx;
  for (; ptr < end1; ptr++)
  {
    ret_val += r_y_1_i * int(*ptr);
  }
  ret_val += B * int(*ptr);
  // middle ones:
  ptr += imagecols - dx - 1;
  uchar* end_j = ptr + dy * imagecols;
  for (; ptr < end_j; ptr += imagecols - dx - 1)
  {
    ret_val += r_x_1_i * int(*ptr);
    ptr++;
    const uchar* end2 = ptr + dx;
    for (; ptr < end2; ptr++)
    {
      ret_val += int(*ptr) * scaling;
    }
    ret_val += r_x1_i * int(*ptr);
  }
  // last row:
  ret_val += D * int(*ptr);
  ptr++;
  const uchar* end3 = ptr + dx;
  for (; ptr < end3; ptr++)
  {
    ret_val += r_y1_i * int(*ptr);
  }
  ret_val += C * int(*ptr);

  return 0xFF & ((ret_val + scaling2 / 2) / scaling2 / 1024);
}

// half sampling
inline void
BriskLayer::halfsample(const cv::Mat& srcimg, cv::Mat& dstimg)
{
  // make sure the destination image is of the right size:
  assert(srcimg.cols/2==dstimg.cols);
  assert(srcimg.rows/2==dstimg.rows);

  // handle non-SSE case
  resize(srcimg, dstimg, dstimg.size(), 0, 0, INTER_AREA);
}

inline void
BriskLayer::twothirdsample(const cv::Mat& srcimg, cv::Mat& dstimg)
{
  // make sure the destination image is of the right size:
  assert((srcimg.cols/3)*2==dstimg.cols);
  assert((srcimg.rows/3)*2==dstimg.rows);

  resize(srcimg, dstimg, dstimg.size(), 0, 0, INTER_AREA);
}

}
