/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

//
// 2D dense optical flow algorithm from the following paper:
// Michael Tao, Jiamin Bai, Pushmeet Kohli, and Sylvain Paris.
// "SimpleFlow: A Non-iterative, Sublinear Optical Flow Algorithm"
// Computer Graphics Forum (Eurographics 2012)
// http://graphics.berkeley.edu/papers/Tao-SAN-2012-05/
//

namespace cv
{

static const uchar MASK_TRUE_VALUE = (uchar)255;

inline static float dist(const Vec3b& p1, const Vec3b& p2) {
  return (float)((p1[0] - p2[0]) * (p1[0] - p2[0]) +
         (p1[1] - p2[1]) * (p1[1] - p2[1]) +
         (p1[2] - p2[2]) * (p1[2] - p2[2]));
}

inline static float dist(const Vec2f& p1, const Vec2f& p2) {
  return (p1[0] - p2[0]) * (p1[0] - p2[0]) +
         (p1[1] - p2[1]) * (p1[1] - p2[1]);
}

inline static float dist(const Point2f& p1, const Point2f& p2) {
  return (p1.x - p2.x) * (p1.x - p2.x) +
         (p1.y - p2.y) * (p1.y - p2.y);
}

inline static float dist(float x1, float y1, float x2, float y2) {
  return (x1 - x2) * (x1 - x2) +
         (y1 - y2) * (y1 - y2);
}

inline static int dist(int x1, int y1, int x2, int y2) {
  return (x1 - x2) * (x1 - x2) +
         (y1 - y2) * (y1 - y2);
}

template<class T>
inline static T min(T t1, T t2, T t3) {
  return (t1 <= t2 && t1 <= t3) ? t1 : min(t2, t3);
}

static void removeOcclusions(const Mat& flow,
                             const Mat& flow_inv,
                             float occ_thr,
                             Mat& confidence) {
  const int rows = flow.rows;
  const int cols = flow.cols;
  if (!confidence.data) {
    confidence = Mat::zeros(rows, cols, CV_32F);
  }
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (dist(flow.at<Vec2f>(r, c), -flow_inv.at<Vec2f>(r, c)) > occ_thr) {
        confidence.at<float>(r, c) = 0;
      } else {
        confidence.at<float>(r, c) = 1;
      }
    }
  }
}

static void wd(Mat& d, int top_shift, int bottom_shift, int left_shift, int right_shift, float sigma) {
  for (int dr = -top_shift, r = 0; dr <= bottom_shift; ++dr, ++r) {
    for (int dc = -left_shift, c = 0; dc <= right_shift; ++dc, ++c) {
      d.at<float>(r, c) = (float)-(dr*dr + dc*dc);
    }
  }
  d *= 1.0 / (2.0 * sigma * sigma);
  exp(d, d);
}

static void wc(const Mat& image, Mat& d, int r0, int c0,
               int top_shift, int bottom_shift, int left_shift, int right_shift, float sigma) {
  const Vec3b centeral_point = image.at<Vec3b>(r0, c0);
  int left_border = c0-left_shift, right_border = c0+right_shift;
  for (int dr = r0-top_shift, r = 0; dr <= r0+bottom_shift; ++dr, ++r) {
    const Vec3b *row = image.ptr<Vec3b>(dr);
    float *d_row = d.ptr<float>(r);
    for (int dc = left_border, c = 0; dc <= right_border; ++dc, ++c) {
      d_row[c] = -dist(centeral_point, row[dc]);
    }
  }
  d *= 1.0 / (2.0 * sigma * sigma);
  exp(d, d);
}

static void crossBilateralFilter(const Mat& image,
                                 const Mat& edge_image,
                                 const Mat confidence,
                                 Mat& dst, int d,
                                 float sigma_color, float sigma_space,
                                 bool flag=false) {
  const int rows = image.rows;
  const int cols = image.cols;
  Mat image_extended, edge_image_extended, confidence_extended;
  copyMakeBorder(image, image_extended, d, d, d, d, BORDER_DEFAULT);
  copyMakeBorder(edge_image, edge_image_extended, d, d, d, d, BORDER_DEFAULT);
  copyMakeBorder(confidence, confidence_extended, d, d, d, d, BORDER_CONSTANT, Scalar(0));
  Mat weights_space(2*d+1, 2*d+1, CV_32F);
  wd(weights_space, d, d, d, d, sigma_space);
  Mat weights(2*d+1, 2*d+1, CV_32F);
  Mat weighted_sum(2*d+1, 2*d+1, CV_32F);

  std::vector<Mat> image_extended_channels;
  split(image_extended, image_extended_channels);

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      wc(edge_image_extended, weights, row+d, col+d, d, d, d, d, sigma_color);

      Range window_rows(row,row+2*d+1);
      Range window_cols(col,col+2*d+1);

      multiply(weights, confidence_extended(window_rows, window_cols), weights);
      multiply(weights, weights_space, weights);
      float weights_sum = (float)sum(weights)[0];

      for (int ch = 0; ch < 2; ++ch) {
        multiply(weights, image_extended_channels[ch](window_rows, window_cols), weighted_sum);
        float total_sum = (float)sum(weighted_sum)[0];

        dst.at<Vec2f>(row, col)[ch] = (flag && fabs(weights_sum) < 1e-9)
          ? image.at<float>(row, col)
          : total_sum / weights_sum;
      }
    }
  }
}

static void calcConfidence(const Mat& prev,
                           const Mat& next,
                           const Mat& flow,
                           Mat& confidence,
                           int max_flow) {
  const int rows = prev.rows;
  const int cols = prev.cols;
  confidence = Mat::zeros(rows, cols, CV_32F);

  for (int r0 = 0; r0 < rows; ++r0) {
    for (int c0 = 0; c0 < cols; ++c0) {
      Vec2f flow_at_point = flow.at<Vec2f>(r0, c0);
      int u0 = cvRound(flow_at_point[0]);
      if (r0 + u0 < 0) { u0 = -r0; }
      if (r0 + u0 >= rows) { u0 = rows - 1 - r0; }
      int v0 = cvRound(flow_at_point[1]);
      if (c0 + v0 < 0) { v0 = -c0; }
      if (c0 + v0 >= cols) { v0 = cols - 1 - c0; }

      const int top_row_shift = -std::min(r0 + u0, max_flow);
      const int bottom_row_shift = std::min(rows - 1 - (r0 + u0), max_flow);
      const int left_col_shift = -std::min(c0 + v0, max_flow);
      const int right_col_shift = std::min(cols - 1 - (c0 + v0), max_flow);

      bool first_flow_iteration = true;
      float sum_e = 0, min_e = 0;

      for (int u = top_row_shift; u <= bottom_row_shift; ++u) {
        for (int v = left_col_shift; v <= right_col_shift; ++v) {
          float e = dist(prev.at<Vec3b>(r0, c0), next.at<Vec3b>(r0 + u0 + u, c0 + v0 + v));
          if (first_flow_iteration) {
            sum_e = e;
            min_e = e;
            first_flow_iteration = false;
          } else {
            sum_e += e;
            min_e = std::min(min_e, e);
          }
        }
      }
      int windows_square = (bottom_row_shift - top_row_shift + 1) *
                           (right_col_shift - left_col_shift + 1);
      confidence.at<float>(r0, c0) = (windows_square == 0) ? 0
                                                           : sum_e / windows_square - min_e;
      CV_Assert(confidence.at<float>(r0, c0) >= 0);
    }
  }
}

static void calcOpticalFlowSingleScaleSF(const Mat& prev_extended,
                                         const Mat& next_extended,
                                         const Mat& mask,
                                         Mat& flow,
                                         int averaging_radius,
                                         int max_flow,
                                         float sigma_dist,
                                         float sigma_color) {
  const int averaging_radius_2 = averaging_radius << 1;
  const int rows = prev_extended.rows - averaging_radius_2;
  const int cols = prev_extended.cols - averaging_radius_2;

  Mat weight_window(averaging_radius_2 + 1, averaging_radius_2 + 1, CV_32F);
  Mat space_weight_window(averaging_radius_2 + 1, averaging_radius_2 + 1, CV_32F);

  wd(space_weight_window, averaging_radius, averaging_radius, averaging_radius, averaging_radius, sigma_dist);

  for (int r0 = 0; r0 < rows; ++r0) {
    for (int c0 = 0; c0 < cols; ++c0) {
      if (!mask.at<uchar>(r0, c0)) {
        continue;
      }

      // TODO: do smth with this creepy staff
      Vec2f flow_at_point = flow.at<Vec2f>(r0, c0);
      int u0 = cvRound(flow_at_point[0]);
      if (r0 + u0 < 0) { u0 = -r0; }
      if (r0 + u0 >= rows) { u0 = rows - 1 - r0; }
      int v0 = cvRound(flow_at_point[1]);
      if (c0 + v0 < 0) { v0 = -c0; }
      if (c0 + v0 >= cols) { v0 = cols - 1 - c0; }

      const int top_row_shift = -std::min(r0 + u0, max_flow);
      const int bottom_row_shift = std::min(rows - 1 - (r0 + u0), max_flow);
      const int left_col_shift = -std::min(c0 + v0, max_flow);
      const int right_col_shift = std::min(cols - 1 - (c0 + v0), max_flow);

      float min_cost = FLT_MAX, best_u = (float)u0, best_v = (float)v0;

      wc(prev_extended, weight_window, r0 + averaging_radius, c0 + averaging_radius,
         averaging_radius, averaging_radius, averaging_radius, averaging_radius, sigma_color);
      multiply(weight_window, space_weight_window, weight_window);

      const int prev_extended_top_window_row = r0;
      const int prev_extended_left_window_col = c0;

      for (int u = top_row_shift; u <= bottom_row_shift; ++u) {
        const int next_extended_top_window_row = r0 + u0 + u;
        for (int v = left_col_shift; v <= right_col_shift; ++v) {
          const int next_extended_left_window_col = c0 + v0 + v;

          float cost = 0;
          for (int r = 0; r <= averaging_radius_2; ++r) {
            const Vec3b *prev_extended_window_row = prev_extended.ptr<Vec3b>(prev_extended_top_window_row + r);
            const Vec3b *next_extended_window_row = next_extended.ptr<Vec3b>(next_extended_top_window_row + r);
            const float* weight_window_row = weight_window.ptr<float>(r);
            for (int c = 0; c <= averaging_radius_2; ++c) {
              cost += weight_window_row[c] *
                           dist(prev_extended_window_row[prev_extended_left_window_col + c],
                                next_extended_window_row[next_extended_left_window_col + c]);
            }
          }
          // cost should be divided by sum(weight_window), but because
          // we interested only in min(cost) and sum(weight_window) is constant
          // for every point - we remove it

          if (cost < min_cost) {
            min_cost = cost;
            best_u = (float)(u + u0);
            best_v = (float)(v + v0);
          }
        }
      }
      flow.at<Vec2f>(r0, c0) = Vec2f(best_u, best_v);
    }
  }
}

static Mat upscaleOpticalFlow(int new_rows,
                               int new_cols,
                               const Mat& image,
                               const Mat& confidence,
                               Mat& flow,
                               int averaging_radius,
                               float sigma_dist,
                               float sigma_color) {
  crossBilateralFilter(flow, image, confidence, flow, averaging_radius, sigma_color, sigma_dist, true);
  Mat new_flow;
  resize(flow, new_flow, Size(new_cols, new_rows), 0, 0, INTER_NEAREST);
  new_flow *= 2;
  return new_flow;
}

static Mat calcIrregularityMat(const Mat& flow, int radius) {
  const int rows = flow.rows;
  const int cols = flow.cols;
  Mat irregularity = Mat::zeros(rows, cols, CV_32F);
  for (int r = 0; r < rows; ++r) {
    const int start_row = std::max(0, r - radius);
    const int end_row = std::min(rows - 1, r + radius);
    for (int c = 0; c < cols; ++c) {
      const int start_col = std::max(0, c - radius);
      const int end_col = std::min(cols - 1, c + radius);
      for (int dr = start_row; dr <= end_row; ++dr) {
        for (int dc = start_col; dc <= end_col; ++dc) {
          const float diff = dist(flow.at<Vec2f>(r, c), flow.at<Vec2f>(dr, dc));
          if (diff > irregularity.at<float>(r, c)) {
            irregularity.at<float>(r, c) = diff;
          }
        }
      }
    }
  }
  return irregularity;
}

static void selectPointsToRecalcFlow(const Mat& flow,
                                     int irregularity_metric_radius,
                                     float speed_up_thr,
                                     int curr_rows,
                                     int curr_cols,
                                     const Mat& prev_speed_up,
                                     Mat& speed_up,
                                     Mat& mask) {
  const int prev_rows = flow.rows;
  const int prev_cols = flow.cols;

  Mat is_flow_regular = calcIrregularityMat(flow, irregularity_metric_radius)
                              < speed_up_thr;
  Mat done = Mat::zeros(prev_rows, prev_cols, CV_8U);
  speed_up = Mat::zeros(curr_rows, curr_cols, CV_8U);
  mask = Mat::zeros(curr_rows, curr_cols, CV_8U);

  for (int r = 0; r < is_flow_regular.rows; ++r) {
    for (int c = 0; c < is_flow_regular.cols; ++c) {
      if (!done.at<uchar>(r, c)) {
        if (is_flow_regular.at<uchar>(r, c) &&
            2*r + 1 < curr_rows && 2*c + 1< curr_cols) {

          bool all_flow_in_region_regular = true;
          int speed_up_at_this_point = prev_speed_up.at<uchar>(r, c);
          int step = (1 << speed_up_at_this_point) - 1;
          int prev_top = r;
          int prev_bottom = std::min(r + step, prev_rows - 1);
          int prev_left = c;
          int prev_right = std::min(c + step, prev_cols - 1);

          for (int rr = prev_top; rr <= prev_bottom; ++rr) {
            for (int cc = prev_left; cc <= prev_right; ++cc) {
              done.at<uchar>(rr, cc) = 1;
              if (!is_flow_regular.at<uchar>(rr, cc)) {
                all_flow_in_region_regular = false;
              }
            }
          }

          int curr_top = std::min(2 * r, curr_rows - 1);
          int curr_bottom = std::min(2*(r + step) + 1, curr_rows - 1);
          int curr_left = std::min(2 * c, curr_cols - 1);
          int curr_right = std::min(2*(c + step) + 1, curr_cols - 1);

          if (all_flow_in_region_regular &&
              curr_top != curr_bottom &&
              curr_left != curr_right) {
            mask.at<uchar>(curr_top, curr_left) = MASK_TRUE_VALUE;
            mask.at<uchar>(curr_bottom, curr_left) = MASK_TRUE_VALUE;
            mask.at<uchar>(curr_top, curr_right) = MASK_TRUE_VALUE;
            mask.at<uchar>(curr_bottom, curr_right) = MASK_TRUE_VALUE;
            for (int rr = curr_top; rr <= curr_bottom; ++rr) {
              for (int cc = curr_left; cc <= curr_right; ++cc) {
                speed_up.at<uchar>(rr, cc) = (uchar)(speed_up_at_this_point + 1);
              }
            }
          } else {
            for (int rr = curr_top; rr <= curr_bottom; ++rr) {
              for (int cc = curr_left; cc <= curr_right; ++cc) {
                mask.at<uchar>(rr, cc) = MASK_TRUE_VALUE;
              }
            }
          }
        } else {
          done.at<uchar>(r, c) = 1;
          for (int dr = 0; dr <= 1; ++dr) {
            int nr = 2*r + dr;
            for (int dc = 0; dc <= 1; ++dc) {
              int nc = 2*c + dc;
              if (nr < curr_rows && nc < curr_cols) {
                mask.at<uchar>(nr, nc) = MASK_TRUE_VALUE;
              }
            }
          }
        }
      }
    }
  }
}

static inline float extrapolateValueInRect(int height, int width,
                                            float v11, float v12,
                                            float v21, float v22,
                                            int r, int c) {
  if (r == 0 && c == 0) { return v11;}
  if (r == 0 && c == width) { return v12;}
  if (r == height && c == 0) { return v21;}
  if (r == height && c == width) { return v22;}

  float qr = float(r) / height;
  float pr = 1.0f - qr;
  float qc = float(c) / width;
  float pc = 1.0f - qc;

  return v11*pr*pc + v12*pr*qc + v21*qr*pc + v22*qc*qr;
}

static void extrapolateFlow(Mat& flow,
                            const Mat& speed_up) {
  const int rows = flow.rows;
  const int cols = flow.cols;
  Mat done = Mat::zeros(rows, cols, CV_8U);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (!done.at<uchar>(r, c) && speed_up.at<uchar>(r, c) > 1) {
        int step = (1 << speed_up.at<uchar>(r, c)) - 1;
        int top = r;
        int bottom = std::min(r + step, rows - 1);
        int left = c;
        int right = std::min(c + step, cols - 1);

        int height = bottom - top;
        int width = right - left;
        for (int rr = top; rr <= bottom; ++rr) {
          for (int cc = left; cc <= right; ++cc) {
            done.at<uchar>(rr, cc) = 1;
            Vec2f flow_at_point;
            Vec2f top_left = flow.at<Vec2f>(top, left);
            Vec2f top_right = flow.at<Vec2f>(top, right);
            Vec2f bottom_left = flow.at<Vec2f>(bottom, left);
            Vec2f bottom_right = flow.at<Vec2f>(bottom, right);

            flow_at_point[0] = extrapolateValueInRect(height, width,
                                                      top_left[0], top_right[0],
                                                      bottom_left[0], bottom_right[0],
                                                      rr-top, cc-left);

            flow_at_point[1] = extrapolateValueInRect(height, width,
                                                      top_left[1], top_right[1],
                                                      bottom_left[1], bottom_right[1],
                                                      rr-top, cc-left);
            flow.at<Vec2f>(rr, cc) = flow_at_point;
          }
        }
      }
    }
  }
}

static void buildPyramidWithResizeMethod(const Mat& src,
                                  std::vector<Mat>& pyramid,
                                  int layers,
                                  int interpolation_type) {
  pyramid.push_back(src);
  for (int i = 1; i <= layers; ++i) {
    Mat prev = pyramid[i - 1];
    if (prev.rows <= 1 || prev.cols <= 1) {
      break;
    }

    Mat next;
    resize(prev, next, Size((prev.cols + 1) / 2, (prev.rows + 1) / 2), 0, 0, interpolation_type);
    pyramid.push_back(next);
  }
}

CV_EXPORTS_W void calcOpticalFlowSF(InputArray _from,
                                    InputArray _to,
                                    OutputArray _resulted_flow,
                                    int layers,
                                    int averaging_radius,
                                    int max_flow,
                                    double sigma_dist,
                                    double sigma_color,
                                    int postprocess_window,
                                    double sigma_dist_fix,
                                    double sigma_color_fix,
                                    double occ_thr,
                                    int upscale_averaging_radius,
                                    double upscale_sigma_dist,
                                    double upscale_sigma_color,
                                    double speed_up_thr)
{
  Mat from = _from.getMat();
  Mat to = _to.getMat();

  std::vector<Mat> pyr_from_images;
  std::vector<Mat> pyr_to_images;

  buildPyramidWithResizeMethod(from, pyr_from_images, layers - 1, INTER_CUBIC);
  buildPyramidWithResizeMethod(to, pyr_to_images, layers - 1, INTER_CUBIC);

  CV_Assert((int)pyr_from_images.size() == layers && (int)pyr_to_images.size() == layers);

  Mat curr_from, curr_to, prev_from, prev_to;
  Mat curr_from_extended, curr_to_extended;

  curr_from = pyr_from_images[layers - 1];
  curr_to = pyr_to_images[layers - 1];

  copyMakeBorder(curr_from, curr_from_extended,
                 averaging_radius, averaging_radius, averaging_radius, averaging_radius,
                 BORDER_DEFAULT);
  copyMakeBorder(curr_to, curr_to_extended,
                 averaging_radius, averaging_radius, averaging_radius, averaging_radius,
                 BORDER_DEFAULT);

  Mat mask = Mat::ones(curr_from.size(), CV_8U);
  Mat mask_inv = Mat::ones(curr_from.size(), CV_8U);

  Mat flow = Mat::zeros(curr_from.size(), CV_32FC2);
  Mat flow_inv = Mat::zeros(curr_to.size(), CV_32FC2);

  Mat confidence;
  Mat confidence_inv;


  calcOpticalFlowSingleScaleSF(curr_from_extended,
                               curr_to_extended,
                               mask,
                               flow,
                               averaging_radius,
                               max_flow,
                               (float)sigma_dist,
                               (float)sigma_color);

  calcOpticalFlowSingleScaleSF(curr_to_extended,
                               curr_from_extended,
                               mask_inv,
                               flow_inv,
                               averaging_radius,
                               max_flow,
                               (float)sigma_dist,
                               (float)sigma_color);

  removeOcclusions(flow,
                   flow_inv,
                   (float)occ_thr,
                   confidence);

  removeOcclusions(flow_inv,
                   flow,
                   (float)occ_thr,
                   confidence_inv);

  Mat speed_up = Mat::zeros(curr_from.size(), CV_8U);
  Mat speed_up_inv = Mat::zeros(curr_from.size(), CV_8U);

  for (int curr_layer = layers - 2; curr_layer >= 0; --curr_layer) {
    curr_from = pyr_from_images[curr_layer];
    curr_to = pyr_to_images[curr_layer];
    prev_from = pyr_from_images[curr_layer + 1];
    prev_to = pyr_to_images[curr_layer + 1];

    copyMakeBorder(curr_from, curr_from_extended,
                   averaging_radius, averaging_radius, averaging_radius, averaging_radius,
                   BORDER_DEFAULT);
    copyMakeBorder(curr_to, curr_to_extended,
                   averaging_radius, averaging_radius, averaging_radius, averaging_radius,
                   BORDER_DEFAULT);

    const int curr_rows = curr_from.rows;
    const int curr_cols = curr_from.cols;

    Mat new_speed_up, new_speed_up_inv;

    selectPointsToRecalcFlow(flow,
                             averaging_radius,
                             (float)speed_up_thr,
                             curr_rows,
                             curr_cols,
                             speed_up,
                             new_speed_up,
                             mask);

    selectPointsToRecalcFlow(flow_inv,
                             averaging_radius,
                             (float)speed_up_thr,
                             curr_rows,
                             curr_cols,
                             speed_up_inv,
                             new_speed_up_inv,
                             mask_inv);

    speed_up = new_speed_up;
    speed_up_inv = new_speed_up_inv;

    flow = upscaleOpticalFlow(curr_rows,
                              curr_cols,
                              prev_from,
                              confidence,
                              flow,
                              upscale_averaging_radius,
                              (float)upscale_sigma_dist,
                              (float)upscale_sigma_color);

    flow_inv = upscaleOpticalFlow(curr_rows,
                                  curr_cols,
                                  prev_to,
                                  confidence_inv,
                                  flow_inv,
                                  upscale_averaging_radius,
                                  (float)upscale_sigma_dist,
                                  (float)upscale_sigma_color);

    calcConfidence(curr_from, curr_to, flow, confidence, max_flow);
    calcOpticalFlowSingleScaleSF(curr_from_extended,
                                 curr_to_extended,
                                 mask,
                                 flow,
                                 averaging_radius,
                                 max_flow,
                                 (float)sigma_dist,
                                 (float)sigma_color);

    calcConfidence(curr_to, curr_from, flow_inv, confidence_inv, max_flow);
    calcOpticalFlowSingleScaleSF(curr_to_extended,
                                 curr_from_extended,
                                 mask_inv,
                                 flow_inv,
                                 averaging_radius,
                                 max_flow,
                                 (float)sigma_dist,
                                 (float)sigma_color);

    extrapolateFlow(flow, speed_up);
    extrapolateFlow(flow_inv, speed_up_inv);

    //TODO: should we remove occlusions for the last stage?
    removeOcclusions(flow, flow_inv, (float)occ_thr, confidence);
    removeOcclusions(flow_inv, flow, (float)occ_thr, confidence_inv);
  }

  crossBilateralFilter(flow, curr_from, confidence, flow,
                       postprocess_window, (float)sigma_color_fix, (float)sigma_dist_fix);

  GaussianBlur(flow, flow, Size(3, 3), 5);

  _resulted_flow.create(flow.size(), CV_32FC2);
  Mat resulted_flow = _resulted_flow.getMat();
  int from_to[] = {0,1 , 1,0};
  mixChannels(&flow, 1, &resulted_flow, 1, from_to, 2);
}

CV_EXPORTS_W void calcOpticalFlowSF(InputArray from,
                                    InputArray to,
                                    OutputArray flow,
                                    int layers,
                                    int averaging_block_size,
                                    int max_flow) {
  calcOpticalFlowSF(from, to, flow, layers, averaging_block_size, max_flow,
                    4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);
}

}
