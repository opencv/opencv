// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include <iostream>

#include "opencv2/ts.hpp"
#include "opencv2/gapi.hpp"

namespace
{
    inline std::ostream& operator<<(std::ostream& o, const cv::GCompileArg& arg)
    {
        return o << (arg.tag.empty() ? "empty" : arg.tag);
    }
}

namespace opencv_test
{

class TestFunctional
{
public:
    cv::Mat in_mat1;
    cv::Mat in_mat2;
    cv::Mat out_mat_gapi;
    cv::Mat out_mat_ocv;

    cv::Scalar sc;

    void initMatsRandU(int type, cv::Size sz_in, int dtype, bool createOutputMatrices = true)
    {
        in_mat1 = cv::Mat(sz_in, type);
        in_mat2 = cv::Mat(sz_in, type);

        auto& rng = cv::theRNG();
        sc = cv::Scalar(rng(100),rng(100),rng(100),rng(100));
        cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
        cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));

        if (createOutputMatrices && dtype != -1)
        {
            out_mat_gapi = cv::Mat (sz_in, dtype);
            out_mat_ocv = cv::Mat (sz_in, dtype);
        }
    }

    void initMatrixRandU(int type, cv::Size sz_in, int dtype, bool createOutputMatrices = true)
    {
        in_mat1 = cv::Mat(sz_in, type);

        auto& rng = cv::theRNG();
        sc = cv::Scalar(rng(100),rng(100),rng(100),rng(100));

        cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));

        if (createOutputMatrices && dtype != -1)
        {
            out_mat_gapi = cv::Mat (sz_in, dtype);
            out_mat_ocv = cv::Mat (sz_in, dtype);
        }
    }

    void initMatsRandN(int type, cv::Size sz_in, int dtype, bool createOutputMatrices = true)
    {
        in_mat1  = cv::Mat(sz_in, type);
        cv::randn(in_mat1, cv::Scalar::all(127), cv::Scalar::all(40.f));

        if (createOutputMatrices  && dtype != -1)
        {
            out_mat_gapi = cv::Mat(sz_in, dtype);
            out_mat_ocv = cv::Mat(sz_in, dtype);
        }
    }

    static cv::Mat nonZeroPixels(const cv::Mat& mat)
    {
        int channels = mat.channels();
        std::vector<cv::Mat> split(channels);
        cv::split(mat, split);
        cv::Mat result;
        for (int c=0; c < channels; c++)
        {
            if (c == 0)
                result = split[c] != 0;
            else
                result = result | (split[c] != 0);
        }
        return result;
    }

    static int countNonZeroPixels(const cv::Mat& mat)
    {
        return cv::countNonZero( nonZeroPixels(mat) );
    }

};

template<class T>
class TestParams: public TestFunctional, public TestWithParam<T>{};

template<class T>
class TestPerfParams: public TestFunctional, public perf::TestBaseWithParam<T>{};

}
