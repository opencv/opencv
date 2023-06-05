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

#include <functional>
#include <numeric>

namespace opencv_test { namespace {

static const int MAX_WIDTH = 640;
static const int MAX_HEIGHT = 480;

typedef testing::TestWithParam<int> HasNonZeroAllZeros;

TEST_P(HasNonZeroAllZeros, hasNonZeroAllZeros)
{
    const int type = GetParam();

    RNG& rng = theRNG();

    const size_t N = 100;
    for(size_t i = 0 ; i<N ; ++i)
    {
      const int width = std::max(1, static_cast<int>(rng.next())%MAX_WIDTH);
      const int height = std::max(1, static_cast<int>(rng.next())%MAX_HEIGHT);
      Mat m = Mat::zeros(Size(width, height), type);
      EXPECT_EQ(false, hasNonZero(m));
    }
}

INSTANTIATE_TEST_CASE_P(Core, HasNonZeroAllZeros,
  testing::Values(CV_8UC1, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32FC1, CV_64FC1)
);

typedef testing::TestWithParam<int> HasNonZeroNegZeros;

TEST_P(HasNonZeroNegZeros, hasNonZeroNegZeros)
{
    const int type = GetParam();

    RNG& rng = theRNG();

    const size_t N = 100;
    for(size_t i = 0 ; i<N ; ++i)
    {
      const int width = std::max(1, static_cast<int>(rng.next())%MAX_WIDTH);
      const int height = std::max(1, static_cast<int>(rng.next())%MAX_HEIGHT);
      Mat m = Mat(Size(width, height), type);
      m.setTo(Scalar::all(-0.));
      EXPECT_EQ(false, hasNonZero(m));
    }
}

INSTANTIATE_TEST_CASE_P(Core, HasNonZeroNegZeros,
  testing::Values(CV_32FC1, CV_64FC1)
);

typedef testing::TestWithParam<int> HasNonZeroRandom;

TEST_P(HasNonZeroRandom, hasNonZeroRandom)
{
    const int type = GetParam();

    RNG& rng = theRNG();

    const size_t N = 1000;
    for(size_t i = 0 ; i<N ; ++i)
    {
      const int width = rng.uniform(1, MAX_WIDTH);
      const int height = rng.uniform(1, MAX_HEIGHT);
      const int nz_pos_x = rng.uniform(0, width);
      const int nz_pos_y = rng.uniform(0, height);
      Mat m = Mat::zeros(Size(width, height), type);
      Mat nzROI = Mat(m, Rect(nz_pos_x, nz_pos_y, 1, 1));
      nzROI.setTo(Scalar::all(1));
      EXPECT_EQ(true, hasNonZero(m));
    }
}

INSTANTIATE_TEST_CASE_P(Core, HasNonZeroRandom,
  testing::Values(CV_8UC1, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32FC1, CV_64FC1)
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
      for(size_t dim = 0 ; dim<ndims ; ++dim)
      {
          const bool isFirstDim = (dim == 0);
          const bool isLastDim = (dim+1 == ndims);
          const int length = rng.uniform(1, 64);
          steps[dim] = (isLastDim ? 1 : static_cast<size_t>(length))*CV_ELEM_SIZE(type);
          sizes[dim] = (isFirstDim || continuous) ? length : rng.uniform(1, length);
          totalBytes *= steps[dim]*static_cast<size_t>(sizes[dim]);
      }

      void* data = fastMalloc(totalBytes);
      EXPECT_NE(nullptr, data);

      memset(data, 0, totalBytes);
      Mat m = Mat(ndims, sizes.data(), type, data, steps.data());

      std::vector<Range> nzRange(ndims);
      for(size_t dim = 0 ; dim<ndims ; ++dim)
      {
        const int pos = rng.uniform(0, sizes[dim]);
        nzRange[dim] = Range(pos, pos+1);
      }

      Mat nzROI = Mat(m, nzRange.data());
      nzROI.setTo(Scalar::all(1));

      const bool nzCount = countNonZero(m);
      EXPECT_EQ((nzCount>0), hasNonZero(m));
      fastFree(data);
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
