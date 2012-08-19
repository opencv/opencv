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
#include "simpleflow.hpp"

//
// 2D dense optical flow algorithm from the following paper:
// Michael Tao, Jiamin Bai, Pushmeet Kohli, and Sylvain Paris.
// "SimpleFlow: A Non-iterative, Sublinear Optical Flow Algorithm"
// Computer Graphics Forum (Eurographics 2012)
// http://graphics.berkeley.edu/papers/Tao-SAN-2012-05/
//

namespace cv
{

WeightedCrossBilateralFilter::WeightedCrossBilateralFilter(
  const Mat& _image,
  int _windowSize,
  double _sigmaDist,
  double _sigmaColor)
  : image(_image), 
    windowSize(_windowSize), 
    sigmaDist(_sigmaDist), 
    sigmaColor(_sigmaColor) {
    
  expDist.resize(2*windowSize*windowSize+1);
  const double sigmaDistSqr = 2 * sigmaDist * sigmaDist;
  for (int i = 0; i <= 2*windowSize*windowSize; ++i) {
    expDist[i] = exp(-i/sigmaDistSqr);
  }

  const double sigmaColorSqr = 2 * sigmaColor * sigmaColor;
  wc.resize(image.rows);
  for (int row = 0; row < image.rows; ++row) {
    wc[row].resize(image.cols);
    for (int col = 0; col < image.cols; ++col) {
      int beginRow = max(0, row - windowSize);
      int beginCol = max(0, col - windowSize);
      int endRow = min(image.rows - 1, row + windowSize);
      int endCol = min(image.cols - 1, col + windowSize);
      wc[row][col] = build<double>(endRow - beginRow + 1, endCol - beginCol + 1);

      for (int r = beginRow; r <= endRow; ++r) {
        for (int c = beginCol; c <= endCol; ++c) {
          wc[row][col][r - beginRow][c - beginCol] = 
            exp(-dist(image.at<Vec3b>(row, col), 
                      image.at<Vec3b>(r, c)) 
                / sigmaColorSqr);
        }
      }
    }
  }
}

Mat WeightedCrossBilateralFilter::apply(Mat& matrix, Mat& weights) {
  int rows = matrix.rows;
  int cols = matrix.cols;

  Mat result = Mat::zeros(rows, cols, CV_64F);
  for (int row = 0; row < rows; ++row) {
    for(int col = 0; col < cols; ++col) {
      result.at<double>(row, col) = 
        convolution(matrix, row, col, weights);
    }
  }
  return result;
}

double WeightedCrossBilateralFilter::convolution(Mat& matrix, 
                                                 int row, int col, 
                                                 Mat& weights) {
  double result = 0, weightsSum = 0;
  int beginRow = max(0, row - windowSize);
  int beginCol = max(0, col - windowSize);
  int endRow = min(matrix.rows - 1, row + windowSize);
  int endCol = min(matrix.cols - 1, col + windowSize);
  for (int r = beginRow; r <= endRow; ++r) {
    double* ptr = matrix.ptr<double>(r);
    for (int c = beginCol; c <= endCol; ++c) {
      const double w = expDist[dist(row, col, r, c)] *
                       wc[row][col][r - beginRow][c - beginCol] *
                       weights.at<double>(r, c);
      result += ptr[c] * w;
      weightsSum += w;
    }
  }
  return result / weightsSum;
}

static void removeOcclusions(const Flow& flow, 
                             const Flow& flow_inv,
                             double occ_thr,
                             Mat& confidence) {
  const int rows = flow.u.rows;
  const int cols = flow.v.cols;
  int occlusions = 0;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (dist(flow.u.at<double>(r, c), flow.v.at<double>(r, c),
               -flow_inv.u.at<double>(r, c), -flow_inv.v.at<double>(r, c)) > occ_thr) {
        confidence.at<double>(r, c) = 0;
        occlusions++;
      }
    }
  }
}

static Mat wd(int top_shift, int bottom_shift, int left_shift, int right_shift, double sigma) {
  const double factor = 1.0 / (2.0 * sigma * sigma);
  Mat d = Mat(top_shift + bottom_shift + 1, right_shift + left_shift + 1, CV_64F);
  for (int dr = -top_shift, r = 0; dr <= bottom_shift; ++dr, ++r) {
    for (int dc = -left_shift, c = 0; dc <= right_shift; ++dc, ++c) {
      d.at<double>(r, c) = -(dr*dr + dc*dc) * factor;
    }
  }
  Mat ed;
  exp(d, ed);
  return ed;
}

static Mat wc(const Mat& image, int r0, int c0, int top_shift, int bottom_shift, int left_shift, int right_shift, double sigma) {
  const double factor = 1.0 / (2.0 * sigma * sigma);
  Mat d = Mat(top_shift + bottom_shift + 1, right_shift + left_shift + 1, CV_64F);
  for (int dr = r0-top_shift, r = 0; dr <= r0+bottom_shift; ++dr, ++r) {
    for (int dc = c0-left_shift, c = 0; dc <= c0+right_shift; ++dc, ++c) {
      d.at<double>(r, c) = -dist(image.at<Vec3b>(r0, c0), image.at<Vec3b>(dr, dc)) * factor;
    }
  }
  Mat ed;
  exp(d, ed);
  return ed;
}

inline static void dist(const Mat& m1, const Mat& m2, Mat& result) {
  const int rows = m1.rows;
  const int cols = m1.cols;
  for (int r = 0; r < rows; ++r) {
    const Vec3b *m1_row = m1.ptr<Vec3b>(r);
    const Vec3b *m2_row = m2.ptr<Vec3b>(r);
    double* row = result.ptr<double>(r);
    for (int c = 0; c < cols; ++c) {
      row[c] = dist(m1_row[c], m2_row[c]);
    }
  }
}

static void calcOpticalFlowSingleScaleSF(const Mat& prev, 
                                         const Mat& next,
                                         const Mat& mask,
                                         Flow& flow,
                                         Mat& confidence,
                                         int averaging_radius, 
                                         int max_flow,
                                         double sigma_dist,
                                         double sigma_color) {
  const int rows = prev.rows;
  const int cols = prev.cols;
  confidence = Mat::zeros(rows, cols, CV_64F);

  for (int r0 = 0; r0 < rows; ++r0) {
    for (int c0 = 0; c0 < cols; ++c0) {
      int u0 = floor(flow.u.at<double>(r0, c0) + 0.5);
      int v0 = floor(flow.v.at<double>(r0, c0) + 0.5);

      const int min_row_shift = -min(r0 + u0, max_flow);
      const int max_row_shift = min(rows - 1 - (r0 + u0), max_flow);
      const int min_col_shift = -min(c0 + v0, max_flow);
      const int max_col_shift = min(cols - 1 - (c0 + v0), max_flow);

      double min_cost = DBL_MAX, best_u = u0, best_v = v0;

      Mat w_full_window;
      double w_full_window_sum;
      Mat diff_storage;

      if (r0 - averaging_radius >= 0 && 
          r0 + averaging_radius < rows &&
          c0 - averaging_radius >= 0 &&
          c0 + averaging_radius < cols &&
          mask.at<uchar>(r0, c0)) {
          w_full_window = wd(averaging_radius, 
                             averaging_radius, 
                             averaging_radius, 
                             averaging_radius, 
                             sigma_dist).mul(
                          wc(prev, r0, c0, 
                             averaging_radius, 
                             averaging_radius, 
                             averaging_radius, 
                             averaging_radius, 
                             sigma_color));

          w_full_window_sum = sum(w_full_window)[0];
          diff_storage = Mat::zeros(averaging_radius*2 + 1, averaging_radius*2 + 1, CV_64F);
      }

      bool first_flow_iteration = true;
      double sum_e, min_e;

      for (int u = min_row_shift; u <= max_row_shift; ++u) {
        for (int v = min_col_shift; v <= max_col_shift; ++v) {
          double e = dist(prev.at<Vec3b>(r0, c0), next.at<Vec3b>(r0 + u0 + u, c0 + v0 + v));
          if (first_flow_iteration) {
            sum_e = e;
            min_e = e;
            first_flow_iteration = false;
          } else {
            sum_e += e;
            min_e = std::min(min_e, e);
          }
          if (!mask.at<uchar>(r0, c0)) {
            continue;
          }

          const int window_top_shift = min(r0, r0 + u + u0, averaging_radius);
          const int window_bottom_shift = min(rows - 1 - r0, 
                                              rows - 1 - (r0 + u + u0), 
                                              averaging_radius);
          const int window_left_shift = min(c0, c0 + v + v0, averaging_radius);
          const int window_right_shift = min(cols - 1 - c0, 
                                             cols - 1 - (c0 + v + v0), 
                                             averaging_radius);

          const Range prev_row_range(r0 - window_top_shift, r0 + window_bottom_shift + 1);
          const Range prev_col_range(c0 - window_left_shift, c0 + window_right_shift + 1);

          const Range next_row_range(r0 + u0 + u - window_top_shift, 
                                     r0 + u0 + u + window_bottom_shift + 1);
          const Range next_col_range(c0 + v0 + v - window_left_shift, 
                                     c0 + v0 + v + window_right_shift + 1); 
          
          Mat diff2;
          Mat w;
          double w_sum;
          if (window_top_shift == averaging_radius &&
              window_bottom_shift == averaging_radius &&
              window_left_shift == averaging_radius &&
              window_right_shift == averaging_radius) {
            w = w_full_window;
            w_sum = w_full_window_sum;
            diff2 = diff_storage;

            dist(prev(prev_row_range, prev_col_range), next(next_row_range, next_col_range), diff2);
          } else {
            diff2 = Mat::zeros(window_bottom_shift + window_top_shift + 1,  
                                   window_right_shift + window_left_shift + 1, CV_64F);
            
            dist(prev(prev_row_range, prev_col_range), next(next_row_range, next_col_range), diff2);

            w = wd(window_top_shift, window_bottom_shift, window_left_shift, window_right_shift, sigma_dist).mul( 
                wc(prev, r0, c0, window_top_shift, window_bottom_shift, window_left_shift, window_right_shift, sigma_color));
            w_sum = sum(w)[0];
          }
          multiply(diff2, w, diff2);
      
          const double cost = sum(diff2)[0] / w_sum;
          if (cost < min_cost) {
            min_cost = cost;
            best_u = u + u0;
            best_v = v + v0;
          }
        }
      }
      int square = (max_row_shift - min_row_shift + 1) *
                   (max_col_shift - min_col_shift + 1);
      confidence.at<double>(r0, c0) = (square == 0) ? 0
                                                   : sum_e / square - min_e;
      if (mask.at<uchar>(r0, c0)) {
        flow.u.at<double>(r0, c0) = best_u;
        flow.v.at<double>(r0, c0) = best_v;
      }
    }
  }
}

static Flow upscaleOpticalFlow(int new_rows, 
                               int new_cols,
                               const Mat& image,
                               const Mat& confidence,
                               const Flow& flow,
                               int averaging_radius,
                               double sigma_dist,
                               double sigma_color) {
  const int rows = image.rows;
  const int cols = image.cols;
  Flow new_flow(new_rows, new_cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const int window_top_shift = min(r, averaging_radius);
      const int window_bottom_shift = min(rows - 1 - r, averaging_radius);
      const int window_left_shift = min(c, averaging_radius);
      const int window_right_shift = min(cols - 1 - c, averaging_radius);
      
      const Range row_range(r - window_top_shift, r + window_bottom_shift + 1);
      const Range col_range(c - window_left_shift, c + window_right_shift + 1);
      
      const Mat w = confidence(row_range, col_range).mul(
        wd(window_top_shift, window_bottom_shift, window_left_shift, window_right_shift, sigma_dist)).mul( 
        wc(image, r, c, window_top_shift, window_bottom_shift, window_left_shift, window_right_shift, sigma_color));

      const double w_sum = sum(w)[0];
      double new_u, new_v;
      if (fabs(w_sum) < 1e-9) {
        new_u = flow.u.at<double>(r, c);
        new_v = flow.v.at<double>(r, c);
      } else {
        new_u = sum(flow.u(row_range, col_range).mul(w))[0] / w_sum;
        new_v = sum(flow.v(row_range, col_range).mul(w))[0] / w_sum;
      }
      
      for (int dr = 0; dr <= 1; ++dr) {
        int nr = 2*r + dr;
        for (int dc = 0; dc <= 1; ++dc) {
          int nc = 2*c + dc;
          if (nr < new_rows && nc < new_cols) {
            new_flow.u.at<double>(nr, nc) = 2 * new_u;
            new_flow.v.at<double>(nr, nc) = 2 * new_v;
          }
        }
      }
    }
  }
  return new_flow;
}

static Mat calcIrregularityMat(const Flow& flow, int radius) {
  const int rows = flow.u.rows;
  const int cols = flow.v.cols;
  Mat irregularity = Mat::zeros(rows, cols, CV_64F);
  for (int r = 0; r < rows; ++r) {
    const int start_row = max(0, r - radius);
    const int end_row = min(rows - 1, r + radius);
    for (int c = 0; c < cols; ++c) {
      const int start_col = max(0, c - radius);
      const int end_col = min(cols - 1, c + radius);
      for (int dr = start_row; dr <= end_row; ++dr) {
        for (int dc = start_col; dc <= end_col; ++dc) {
          const double diff = dist(flow.u.at<double>(r, c), flow.v.at<double>(r, c), 
                                   flow.u.at<double>(dr, dc), flow.v.at<double>(dr, dc));
          if (diff > irregularity.at<double>(r, c)) {
            irregularity.at<double>(r, c) = diff;
          }
        }
      }
    }
  }
  return irregularity;
}

static void selectPointsToRecalcFlow(const Flow& flow, 
                                     int irregularity_metric_radius,
                                     int speed_up_thr,
                                     int curr_rows,
                                     int curr_cols,
                                     const Mat& prev_speed_up,
                                     Mat& speed_up,
                                     Mat& mask) {
  const int prev_rows = flow.u.rows;
  const int prev_cols = flow.v.cols;

  Mat is_flow_regular = calcIrregularityMat(flow, 
                                                irregularity_metric_radius)
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
                speed_up.at<uchar>(rr, cc) = speed_up_at_this_point + 1; 
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

static inline double extrapolateValueInRect(int height, int width,
                                            double v11, double v12,
                                            double v21, double v22,
                                            int r, int c) {
  if (r == 0 && c == 0) { return v11;}
  if (r == 0 && c == width) { return v12;}
  if (r == height && c == 0) { return v21;}
  if (r == height && c == width) { return v22;}
  
  double qr = double(r) / height;
  double pr = 1.0 - qr;
  double qc = double(c) / width;
  double pc = 1.0 - qc;

  return v11*pr*pc + v12*pr*qc + v21*qr*pc + v22*qc*qr; 
}
                                              
static void extrapolateFlow(Flow& flow,
                            const Mat& speed_up) {
  const int rows = flow.u.rows;
  const int cols = flow.u.cols;
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
            flow.u.at<double>(rr, cc) = extrapolateValueInRect(
                                          height, width, 
                                          flow.u.at<double>(top, left),
                                          flow.u.at<double>(top, right),
                                          flow.u.at<double>(bottom, left),
                                          flow.u.at<double>(bottom, right),
                                          rr-top, cc-left); 

            flow.v.at<double>(rr, cc) = extrapolateValueInRect(
                                          height, width, 
                                          flow.v.at<double>(top, left),
                                          flow.v.at<double>(top, right),
                                          flow.v.at<double>(bottom, left),
                                          flow.v.at<double>(bottom, right),
                                          rr-top, cc-left); 
          }
        }
      }
    }
  }
}

static void buildPyramidWithResizeMethod(Mat& src,
                                  vector<Mat>& pyramid,
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

static Flow calcOpticalFlowSF(Mat& from, 
                       Mat& to, 
                       int layers,
                       int averaging_block_size, 
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
                       double speed_up_thr) {
  vector<Mat> pyr_from_images;
  vector<Mat> pyr_to_images;

  buildPyramidWithResizeMethod(from, pyr_from_images, layers - 1, INTER_CUBIC);
  buildPyramidWithResizeMethod(to, pyr_to_images, layers - 1, INTER_CUBIC);
//  buildPyramid(from, pyr_from_images, layers - 1, BORDER_WRAP);
//  buildPyramid(to, pyr_to_images, layers - 1, BORDER_WRAP);

  if ((int)pyr_from_images.size() != layers) {
      exit(1);
  }

  if ((int)pyr_to_images.size() != layers) {
      exit(1);
  }

  Mat first_from_image = pyr_from_images[layers - 1];
  Mat first_to_image = pyr_to_images[layers - 1];

  Mat mask = Mat::ones(first_from_image.rows, first_from_image.cols, CV_8U);
  Mat mask_inv = Mat::ones(first_from_image.rows, first_from_image.cols, CV_8U);

  Flow flow(first_from_image.rows, first_from_image.cols);
  Flow flow_inv(first_to_image.rows, first_to_image.cols);

  Mat confidence;
  Mat confidence_inv;

  calcOpticalFlowSingleScaleSF(first_from_image, 
                               first_to_image, 
                               mask,
                               flow,
                               confidence,
                               averaging_block_size, 
                               max_flow, 
                               sigma_dist, 
                               sigma_color);

  calcOpticalFlowSingleScaleSF(first_to_image, 
                               first_from_image, 
                               mask_inv,
                               flow_inv,
                               confidence_inv,
                               averaging_block_size, 
                               max_flow, 
                               sigma_dist, 
                               sigma_color);

  removeOcclusions(flow, 
                   flow_inv,
                   occ_thr,
                   confidence);

  removeOcclusions(flow_inv, 
                   flow,
                   occ_thr,
                   confidence_inv);

  Mat speed_up = Mat::zeros(first_from_image.rows, first_from_image.cols, CV_8U);
  Mat speed_up_inv = Mat::zeros(first_from_image.rows, first_from_image.cols, CV_8U);

  for (int curr_layer = layers - 2; curr_layer >= 0; --curr_layer) {
    const Mat curr_from = pyr_from_images[curr_layer];
    const Mat curr_to = pyr_to_images[curr_layer];
    const Mat prev_from = pyr_from_images[curr_layer + 1];
    const Mat prev_to = pyr_to_images[curr_layer + 1];

    const int curr_rows = curr_from.rows;
    const int curr_cols = curr_from.cols;

    Mat new_speed_up, new_speed_up_inv;

    selectPointsToRecalcFlow(flow,
                             averaging_block_size,
                             speed_up_thr,
                             curr_rows,
                             curr_cols,
                             speed_up,
                             new_speed_up,
                             mask);

    int points_to_recalculate = sum(mask)[0] / MASK_TRUE_VALUE;

    selectPointsToRecalcFlow(flow_inv,
                             averaging_block_size,
                             speed_up_thr,
                             curr_rows,
                             curr_cols,
                             speed_up_inv,
                             new_speed_up_inv,
                             mask_inv);

    points_to_recalculate = sum(mask_inv)[0] / MASK_TRUE_VALUE;

    speed_up = new_speed_up;
    speed_up_inv = new_speed_up_inv;

    flow = upscaleOpticalFlow(curr_rows,
                              curr_cols,
                              prev_from,
                              confidence,
                              flow, 
                              upscale_averaging_radius,
                              upscale_sigma_dist,
                              upscale_sigma_color);

    flow_inv = upscaleOpticalFlow(curr_rows,
                                  curr_cols,
                                  prev_to,
                                  confidence_inv,
                                  flow_inv,
                                  upscale_averaging_radius,
                                  upscale_sigma_dist,
                                  upscale_sigma_color);

    calcOpticalFlowSingleScaleSF(curr_from, 
                                 curr_to, 
                                 mask,
                                 flow,
                                 confidence,
                                 averaging_block_size, 
                                 max_flow, 
                                 sigma_dist, 
                                 sigma_color);

    calcOpticalFlowSingleScaleSF(curr_to,
                                 curr_from, 
                                 mask_inv,
                                 flow_inv,
                                 confidence_inv,
                                 averaging_block_size, 
                                 max_flow, 
                                 sigma_dist, 
                                 sigma_color);

    extrapolateFlow(flow, speed_up);
    extrapolateFlow(flow_inv, speed_up_inv);

    removeOcclusions(flow, flow_inv, occ_thr, confidence);
    removeOcclusions(flow_inv, flow, occ_thr, confidence_inv);
  }

  WeightedCrossBilateralFilter filter_postprocess(pyr_from_images[0], 
                                                  postprocess_window,
                                                  sigma_dist_fix,
                                                  sigma_color_fix);

  flow.u = filter_postprocess.apply(flow.u, confidence);
  flow.v = filter_postprocess.apply(flow.v, confidence);

  Mat blured_u, blured_v;
  GaussianBlur(flow.u, blured_u, Size(3, 3), 5);
  GaussianBlur(flow.v, blured_v, Size(3, 3), 5);

  return Flow(blured_v, blured_u);
}

void calcOpticalFlowSF(Mat& from, 
                       Mat& to, 
                       Mat& flowX, 
                       Mat& flowY,
                       int layers,
                       int averaging_block_size, 
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
                       double speed_up_thr) {

  Flow flow = calcOpticalFlowSF(from, to, 
                                layers,
                                averaging_block_size,
                                max_flow,
                                sigma_dist,
                                sigma_color,
                                postprocess_window,
                                sigma_dist_fix,
                                sigma_color_fix,
                                occ_thr,
                                upscale_averaging_radius,
                                upscale_sigma_dist,
                                upscale_sigma_color,
                                speed_up_thr);
  flowX = flow.u;
  flowY = flow.v;
}

}

