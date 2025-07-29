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

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<std::tuple<int, Size> > HasNonZeroAllZeros;

TEST_P(HasNonZeroAllZeros, hasNonZeroAllZeros)
{
    const int type = std::get<0>(GetParam());
    const Size size = std::get<1>(GetParam());

    Mat m = Mat::zeros(size, type);
    EXPECT_FALSE(hasNonZero(m));
}

INSTANTIATE_TEST_CASE_P(Core, HasNonZeroAllZeros,
    testing::Combine(
        testing::Values(CV_8UC1, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32FC1, CV_64FC1),
        testing::Values(Size(1, 1), Size(320, 240), Size(127, 113), Size(1, 113))
    )
);

typedef testing::TestWithParam<std::tuple<int, Size> > HasNonZeroNegZeros;

TEST_P(HasNonZeroNegZeros, hasNonZeroNegZeros)
{
    const int type = std::get<0>(GetParam());
    const Size size = std::get<1>(GetParam());

    Mat m = Mat(size, type);
    m.setTo(Scalar::all(-0.));
    EXPECT_FALSE(hasNonZero(m));
}

INSTANTIATE_TEST_CASE_P(Core, HasNonZeroNegZeros,
    testing::Combine(
        testing::Values(CV_32FC1, CV_64FC1),
        testing::Values(Size(1, 1), Size(320, 240), Size(127, 113), Size(1, 113))
    )
);

typedef testing::TestWithParam<std::tuple<int, Size> > HasNonZeroLimitValues;

TEST_P(HasNonZeroLimitValues, hasNonZeroLimitValues)
{
    const int type = std::get<0>(GetParam());
    const Size size = std::get<1>(GetParam());

    Mat m = Mat(size, type);

    m.setTo(Scalar::all(std::numeric_limits<double>::infinity()));
    EXPECT_TRUE(hasNonZero(m));

    m.setTo(Scalar::all(-std::numeric_limits<double>::infinity()));
    EXPECT_TRUE(hasNonZero(m));

    m.setTo(Scalar::all(std::numeric_limits<double>::quiet_NaN()));
    EXPECT_TRUE(hasNonZero(m));

    m.setTo((CV_MAT_DEPTH(type) == CV_64F) ? Scalar::all(std::numeric_limits<double>::epsilon()) : Scalar::all(std::numeric_limits<float>::epsilon()));
    EXPECT_TRUE(hasNonZero(m));

    m.setTo((CV_MAT_DEPTH(type) == CV_64F) ? Scalar::all(std::numeric_limits<double>::min()) : Scalar::all(std::numeric_limits<float>::min()));
    EXPECT_TRUE(hasNonZero(m));

    m.setTo((CV_MAT_DEPTH(type) == CV_64F) ? Scalar::all(std::numeric_limits<double>::denorm_min()) : Scalar::all(std::numeric_limits<float>::denorm_min()));
    EXPECT_TRUE(hasNonZero(m));
}

INSTANTIATE_TEST_CASE_P(Core, HasNonZeroLimitValues,
    testing::Combine(
        testing::Values(CV_32FC1, CV_64FC1),
        testing::Values(Size(1, 1), Size(320, 240), Size(127, 113), Size(1, 113))
    )
);

typedef testing::TestWithParam<std::tuple<int, Size> > HasNonZeroRandom;

TEST_P(HasNonZeroRandom, hasNonZeroRandom)
{
    const int type = std::get<0>(GetParam());
    const Size size = std::get<1>(GetParam());

    RNG& rng = theRNG();

    const size_t N = std::min(100, size.area());
    for(size_t i = 0 ; i<N ; ++i)
    {
      const int nz_pos_x = rng.uniform(0, size.width);
      const int nz_pos_y = rng.uniform(0, size.height);
      Mat m = Mat::zeros(size, type);
      Mat nzROI = Mat(m, Rect(nz_pos_x, nz_pos_y, 1, 1));
      nzROI.setTo(Scalar::all(1));
      EXPECT_TRUE(hasNonZero(m));
    }
}

INSTANTIATE_TEST_CASE_P(Core, HasNonZeroRandom,
    testing::Combine(
        testing::Values(CV_8UC1, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32FC1, CV_64FC1),
        testing::Values(Size(1, 1), Size(320, 240), Size(127, 113), Size(1, 113))
    )
);

typedef testing::TestWithParam<tuple<int, int, bool> > HasNonZeroNd;

TEST_P(HasNonZeroNd, hasNonZeroNd)
{
    const int type = get<0>(GetParam());
    const int ndims = get<1>(GetParam());
    const bool continuous = get<2>(GetParam());

    RNG& rng = theRNG();

    const size_t N = 10;
    for(size_t i = 0 ; i<N ; ++i)
    {
      std::vector<size_t> steps(ndims);
      std::vector<int> sizes(ndims);
      size_t totalBytes = 1;
      for(int dim = 0 ; dim<ndims ; ++dim)
      {
          const bool isFirstDim = (dim == 0);
          const bool isLastDim = (dim+1 == ndims);
          const int length = rng.uniform(1, 64);
          steps[dim] = (isLastDim ? 1 : static_cast<size_t>(length))*CV_ELEM_SIZE(type);
          sizes[dim] = (isFirstDim || continuous) ? length : rng.uniform(1, length);
          totalBytes *= steps[dim]*static_cast<size_t>(sizes[dim]);
      }

      std::vector<unsigned char> buffer(totalBytes);
      void* data = buffer.data();

      Mat m = Mat(ndims, sizes.data(), type, data, steps.data());

      std::vector<Range> nzRange(ndims);
      for(int dim = 0 ; dim<ndims ; ++dim)
      {
        const int pos = rng.uniform(0, sizes[dim]);
        nzRange[dim] = Range(pos, pos+1);
      }

      Mat nzROI = Mat(m, nzRange.data());
      nzROI.setTo(Scalar::all(1));

      const int nzCount = countNonZero(m);
      EXPECT_EQ((nzCount>0), hasNonZero(m));
    }
}

INSTANTIATE_TEST_CASE_P(Core, HasNonZeroNd,
    testing::Combine(
        testing::Values(CV_8UC1),
        testing::Values(2, 3),
        testing::Values(true, false)
    )
);

}} // namespace
