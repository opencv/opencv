// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"
#include "opencv2/gapi/cpu/gcpukernel.hpp"

namespace opencv_test
{

  namespace
  {
      G_TYPED_KERNEL(CustomResize, <cv::GMat(cv::GMat, cv::Size, double, double, int)>, "org.opencv.customk.resize")
      {
          static cv::GMatDesc outMeta(cv::GMatDesc in, cv::Size sz, double fx, double fy, int) {
              if (sz.width != 0 && sz.height != 0)
              {
                  return in.withSize(to_own(sz));
              }
              else
              {
                  GAPI_Assert(fx != 0. && fy != 0.);
                  return in.withSize
                    (cv::gapi::own::Size(static_cast<int>(std::round(in.size.width  * fx)),
                                         static_cast<int>(std::round(in.size.height * fy))));
              }
          }
      };

      GAPI_OCV_KERNEL(CustomResizeImpl, CustomResize)
      {
          static void run(const cv::Mat& in, cv::Size sz, double fx, double fy, int interp, cv::Mat &out)
          {
              cv::resize(in, out, sz, fx, fy, interp);
          }
      };

      struct GComputationApplyTest: public ::testing::Test
      {
          cv::GMat in;
          cv::Mat  in_mat;
          cv::Mat  out_mat;
          cv::GComputation m_c;

          GComputationApplyTest() : in_mat(300, 300, CV_8UC1),
                                    m_c(cv::GIn(in), cv::GOut(CustomResize::on(in, cv::Size(100, 100),
                                                                               0.0, 0.0, cv::INTER_LINEAR)))
          {
          }
      };
  }

  TEST_F(GComputationApplyTest, ThrowDontPassCustomKernel)
  {
      EXPECT_THROW(m_c.apply(in_mat, out_mat), std::logic_error);
  }

  TEST_F(GComputationApplyTest, NoThrowPassCustomKernel)
  {
      const auto pkg = cv::gapi::kernels<CustomResizeImpl>();

      ASSERT_NO_THROW(m_c.apply(in_mat, out_mat, cv::compile_args(pkg)));
  }

} // namespace opencv_test
