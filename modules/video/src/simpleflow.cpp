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

static void removeOcclusions(const Mat& flow, 
                             const Mat& flow_inv,
                             float occ_thr,
                             Mat& confidence) {
  const int rows = flow.rows;
  const int cols = flow.cols;
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
  const float factor = 1.0 / (2.0 * sigma * sigma);
  for (int dr = -top_shift, r = 0; dr <= bottom_shift; ++dr, ++r) {
    for (int dc = -left_shift, c = 0; dc <= right_shift; ++dc, ++c) {
      d.at<float>(r, c) = -(dr*dr + dc*dc) * factor;
    }
  }
  exp(d, d);
}

static void wc(const Mat& image, Mat& d, int r0, int c0, 
               int top_shift, int bottom_shift, int left_shift, int right_shift, float sigma) {
  const float factor = 1.0 / (2.0 * sigma * sigma);
  const Vec3b centeral_point = image.at<Vec3b>(r0, c0);
  for (int dr = r0-top_shift, r = 0; dr <= r0+bottom_shift; ++dr, ++r) {
    const Vec3b *row = image.ptr<Vec3b>(dr); 
    float *d_row = d.ptr<float>(r);
    for (int dc = c0-left_shift, c = 0; dc <= c0+right_shift; ++dc, ++c) {
      d_row[c] = -dist(centeral_point, row[dc]) * factor;
    }
  }
  exp(d, d);
}

static void dist(const Mat& m1, const Mat& m2, Mat& result) {
  const int rows = m1.rows;
  const int cols = m1.cols;
  for (int r = 0; r < rows; ++r) {
    const Vec3b *m1_row = m1.ptr<Vec3b>(r);
    const Vec3b *m2_row = m2.ptr<Vec3b>(r);
    float* row = result.ptr<float>(r);
    for (int c = 0; c < cols; ++c) {
      row[c] = dist(m1_row[c], m2_row[c]);
    }
  }
}

static void crossBilateralFilter(const Mat& image, const Mat& edge_image, const Mat confidence, Mat& dst, int d, float sigma_color, float sigma_space, bool flag=false) {
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
  

  vector<Mat> image_extended_channels;
  split(image_extended, image_extended_channels);

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      wc(edge_image_extended, weights, row+d, col+d, d, d, d, d, sigma_color);

      Range window_rows(row,row+2*d+1);
      Range window_cols(col,col+2*d+1);

      multiply(weights, confidence_extended(window_rows, window_cols), weights);
      multiply(weights, weights_space, weights);
      float weights_sum = sum(weights)[0];

      for (int ch = 0; ch < 2; ++ch) {
        multiply(weights, image_extended_channels[ch](window_rows, window_cols), weighted_sum);
        float total_sum = sum(weighted_sum)[0];

        dst.at<Vec2f>(row, col)[ch] = (flag && fabs(weights_sum) < 1e-9) 
          ? image.at<float>(row, col) 
          : total_sum / weights_sum;
      }
    }
  }
}

static void calcOpticalFlowSingleScaleSF(const Mat& prev, 
                                         const Mat& next,
                                         const Mat& mask,
                                         Mat& flow,
                                         Mat& confidence,
                                         int averaging_radius, 
                                         int max_flow,
                                         float sigma_dist,
                                         float sigma_color) {
  const int rows = prev.rows;
  const int cols = prev.cols;
  confidence = Mat::zeros(rows, cols, CV_32F);
  
  Mat diff_storage(averaging_radius*2 + 1, averaging_radius*2 + 1, CV_32F);
  Mat w_full_window(averaging_radius*2 + 1, averaging_radius*2 + 1, CV_32F);
  Mat wd_full_window(averaging_radius*2 + 1, averaging_radius*2 + 1, CV_32F);
  float w_full_window_sum;

  Mat prev_extended;
  copyMakeBorder(prev, prev_extended, 
                 averaging_radius, averaging_radius, averaging_radius, averaging_radius,
                 BORDER_DEFAULT);

  wd(wd_full_window, averaging_radius, averaging_radius, averaging_radius, averaging_radius, sigma_dist);

  for (int r0 = 0; r0 < rows; ++r0) {
    for (int c0 = 0; c0 < cols; ++c0) {
      Vec2f flow_at_point = flow.at<Vec2f>(r0, c0);
      int u0 = floor(flow_at_point[0] + 0.5);
      if (r0 + u0 < 0) { u0 = -r0; }
      if (r0 + u0 >= rows) { u0 = rows - 1 - r0; }
      int v0 = floor(flow_at_point[1] + 0.5);
      if (c0 + v0 < 0) { v0 = -c0; }
      if (c0 + v0 >= cols) { v0 = cols - 1 - c0; }

      const int min_row_shift = -min(r0 + u0, max_flow);
      const int max_row_shift = min(rows - 1 - (r0 + u0), max_flow);
      const int min_col_shift = -min(c0 + v0, max_flow);
      const int max_col_shift = min(cols - 1 - (c0 + v0), max_flow);

      float min_cost = DBL_MAX, best_u = u0, best_v = v0;

      if (mask.at<uchar>(r0, c0)) {
          wc(prev_extended, w_full_window, r0 + averaging_radius, c0 + averaging_radius,
             averaging_radius, averaging_radius, averaging_radius, averaging_radius, sigma_color);
          multiply(w_full_window, wd_full_window, w_full_window);
          w_full_window_sum = sum(w_full_window)[0];
      }

      bool first_flow_iteration = true;
      float sum_e, min_e;

      for (int u = min_row_shift; u <= max_row_shift; ++u) {
        for (int v = min_col_shift; v <= max_col_shift; ++v) {
          float e = dist(prev.at<Vec3b>(r0, c0), next.at<Vec3b>(r0 + u0 + u, c0 + v0 + v));
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
          float w_sum;
          if (window_top_shift == averaging_radius &&
              window_bottom_shift == averaging_radius &&
              window_left_shift == averaging_radius &&
              window_right_shift == averaging_radius) {
            w = w_full_window;
            w_sum = w_full_window_sum;
            diff2 = diff_storage;
            dist(prev(prev_row_range, prev_col_range), next(next_row_range, next_col_range), diff2);
          } else {
            diff2 = diff_storage(Range(averaging_radius - window_top_shift, 
                                       averaging_radius + 1 + window_bottom_shift),
                                 Range(averaging_radius - window_left_shift,
                                       averaging_radius + 1 + window_right_shift));
            
            dist(prev(prev_row_range, prev_col_range), next(next_row_range, next_col_range), diff2);
            w = w_full_window(Range(averaging_radius - window_top_shift, 
                                    averaging_radius + 1 + window_bottom_shift),
                              Range(averaging_radius - window_left_shift,
                                    averaging_radius + 1 + window_right_shift));
            w_sum = sum(w)[0];
          }
          multiply(diff2, w, diff2);
      
          const float cost = sum(diff2)[0] / w_sum;
          if (cost < min_cost) {
            min_cost = cost;
            best_u = u + u0;
            best_v = v + v0;
          }
        }
      }
      int windows_square = (max_row_shift - min_row_shift + 1) *
                           (max_col_shift - min_col_shift + 1);
      confidence.at<float>(r0, c0) = (windows_square == 0) ? 0
                                                           : sum_e / windows_square - min_e;
      CV_Assert(confidence.at<float>(r0, c0) >= 0); // TODO: remove it after testing
      if (mask.at<uchar>(r0, c0)) {
        flow.at<Vec2f>(r0, c0) = Vec2f(best_u, best_v);
      }
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
  crossBilateralFilter(flow, image, confidence, flow, averaging_radius, sigma_color, sigma_dist, false);
  Mat new_flow;
  resize(flow, new_flow, Size(new_cols, new_rows), 0, 0, INTER_NEAREST);
  new_flow *= 2;
  return new_flow;
}

static Mat calcIrregularityMat(const Mat& flow, int radius) {
  const int rows = flow.rows;
  const int cols = flow.cols;
  Mat irregularity(rows, cols, CV_32F);
  for (int r = 0; r < rows; ++r) {
    const int start_row = max(0, r - radius);
    const int end_row = min(rows - 1, r + radius);
    for (int c = 0; c < cols; ++c) {
      const int start_col = max(0, c - radius);
      const int end_col = min(cols - 1, c + radius);
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
                                     int speed_up_thr,
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

static inline float extrapolateValueInRect(int height, int width,
                                            float v11, float v12,
                                            float v21, float v22,
                                            int r, int c) {
  if (r == 0 && c == 0) { return v11;}
  if (r == 0 && c == width) { return v12;}
  if (r == height && c == 0) { return v21;}
  if (r == height && c == width) { return v22;}
  
  float qr = float(r) / height;
  float pr = 1.0 - qr;
  float qc = float(c) / width;
  float pc = 1.0 - qc;

  return v11*pr*pc + v12*pr*qc + v21*qr*pc + v22*qc*qr; 
}
                                              
static void extrapolateFlow(Mat& flow,
                            const Mat& speed_up) {
  const int rows = flow.rows;
  const int cols = flow.cols;
  Mat done(rows, cols, CV_8U);
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

void calcOpticalFlowSF(Mat& from, 
                       Mat& to, 
                       Mat& resulted_flow,
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

  Mat flow(first_from_image.rows, first_from_image.cols, CV_32FC2);
  Mat flow_inv(first_to_image.rows, first_to_image.cols, CV_32FC2);

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

    selectPointsToRecalcFlow(flow_inv,
                             averaging_block_size,
                             speed_up_thr,
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

  crossBilateralFilter(flow, pyr_from_images[0], confidence, flow, 
                       postprocess_window, sigma_color_fix, sigma_dist_fix);

  GaussianBlur(flow, flow, Size(3, 3), 5);

  resulted_flow = Mat(flow.size(), CV_32FC2);
  int from_to[] = {0,1 , 1,0};
  mixChannels(&flow, 1, &resulted_flow, 1, from_to, 2);
}

}

