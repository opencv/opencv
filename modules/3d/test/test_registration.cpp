// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_WillowGarage.md file found in this module's directory

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class CV_RgbdDepthRegistrationTest: public cvtest::BaseTest
{
public:
  CV_RgbdDepthRegistrationTest()
  {
  }
  ~CV_RgbdDepthRegistrationTest()
  {
  }
protected:
  void
  run(int)
  {

      // Test all three input types for no-op registrations (where a depth image is registered to itself)

      int code = noOpRandomRegistrationTest<unsigned short>(100, 2500);
      if( code != cvtest::TS::OK )
      {
          ts->set_failed_test_info(code);
          return;
      }

      code = noOpRandomRegistrationTest<float>(0.1f, 2.5f);
      if( code != cvtest::TS::OK )
      {
          ts->set_failed_test_info(code);
          return;
      }

      code = noOpRandomRegistrationTest<double>(0.1, 2.5);
      if( code != cvtest::TS::OK )
      {
          ts->set_failed_test_info(code);
          return;
      }


      // Test sentinel value handling, occlusion, and dilation
      {

          // K from a VGA Kinect
          Mat K = (Mat_<float>(3, 3) << 525., 0., 319.5, 0., 525., 239.5, 0., 0., 1.);

          int width = 640, height = 480;

          // All elements are zero except for first two along the diagonal
          Mat_<unsigned short> vgaDepth(height, width, (unsigned short)0);
          vgaDepth(0,0) = 1001;
          vgaDepth(1,1) = 1000;

          Mat_<unsigned short> registeredDepth;
          registerDepth(K, K, Mat(), Matx44f::eye(), vgaDepth, Size(width, height), registeredDepth, true);

          // We expect the closer depth of 1000 to occlude the more distant depth and occupy the
          // upper four left pixels in the depth image because of dilation
          Mat_<unsigned short> expectedResult(height, width, (unsigned short)0);
          expectedResult(0,0) = 1000;
          expectedResult(0,1) = 1000;
          expectedResult(1,0) = 1000;
          expectedResult(1,1) = 1000;

          int cmpResult =  cvtest::cmpEps2( ts, registeredDepth, expectedResult, 0, true, "Dilation and occlusion");

          if( cmpResult != cvtest::TS::OK )
          {
              ts->set_failed_test_info(cmpResult);
              return;
          }

      }

      ts->set_failed_test_info(cvtest::TS::OK);

  }
private:

    template <class DepthDepth>
    int noOpRandomRegistrationTest(DepthDepth minDepth, DepthDepth maxDepth)
    {

        // K from a VGA Kinect
        Mat K = (Mat_<float>(3, 3) << 525., 0., 319.5, 0., 525., 239.5, 0., 0., 1.);

        // Create a random depth image
        RNG rng;
        Mat_<DepthDepth> randomVGADepth(480, 640);
        rng.fill(randomVGADepth, RNG::UNIFORM, minDepth, maxDepth);

        Mat registeredDepth;
        registerDepth(K, K, Mat(), Matx44f::eye(), randomVGADepth, Size(640, 480), registeredDepth);

        // See if registeredDepth == depth
        return cvtest::cmpEps2( ts, registeredDepth, randomVGADepth, 1e-5, true, "No-op registration");

    }

};

TEST(Rgbd_DepthRegistration, compute)
{
  CV_RgbdDepthRegistrationTest test;
  test.safe_run();
}

TEST(Rgbd_DepthRegistration, issue_2234)
{
    Matx33f intrinsicsDepth(100, 0,  50, 0, 100, 50, 0, 0, 1);
    Matx33f intrinsicsColor(100, 0, 200, 0, 100, 50, 0, 0, 1);

    Mat_<float> depthMat(100, 100, (float)0.);
    for(int i = 1; i <= 100; i++)
    {
        for(int j = 1; j <= 100; j++)
            depthMat(i-1,j-1) = (float)j;
    }

    Mat registeredDepth;
    registerDepth(intrinsicsDepth, intrinsicsColor, Mat(), Matx44f::eye(), depthMat, Size(400, 100), registeredDepth);

    Rect roi( 150, 0, 100, 100 );
    Mat subM(registeredDepth,roi);

    EXPECT_EQ(0, cvtest::norm(subM, depthMat, NORM_INF));
}


}} // namespace
