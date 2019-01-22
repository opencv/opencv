/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 // This is a test file for the function decomposeHomography contributed to OpenCV
 // by Samson Yilma.
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
 // Copyright (C) 2014, Samson Yilma (samson_yilma@yahoo.com), all rights reserved.
 //
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

class CV_HomographyDecompTest: public cvtest::BaseTest {

public:
    CV_HomographyDecompTest()
    {
        buildTestDataSet();
    }

protected:
    void run(int)
    {
        vector<Mat> rotations;
        vector<Mat> translations;
        vector<Mat> normals;

        decomposeHomographyMat(_H, _K, rotations, translations, normals);

        //there should be at least 1 solution
        ASSERT_GT(static_cast<int>(rotations.size()), 0);
        ASSERT_GT(static_cast<int>(translations.size()), 0);
        ASSERT_GT(static_cast<int>(normals.size()), 0);

        ASSERT_EQ(rotations.size(), normals.size());
        ASSERT_EQ(translations.size(), normals.size());

        ASSERT_TRUE(containsValidMotion(rotations, translations, normals));

        decomposeHomographyMat(_H, _K, rotations, noArray(), noArray());
        ASSERT_GT(static_cast<int>(rotations.size()), 0);
    }

private:

    void buildTestDataSet()
    {
        _K = Matx33d(640, 0.0,  320,
                      0,    640, 240,
                      0,    0,   1);

         _H = Matx33d(2.649157564634028,  4.583875997496426,  70.694447785121326,
                     -1.072756858861583,  3.533262150437228,  1513.656999614321649,
                      0.001303887589576,  0.003042206876298,  1.000000000000000
                      );

        //expected solution for the given homography and intrinsic matrices
         _R = Matx33d(0.43307983549125, 0.545749113549648, -0.717356090899523,
                     -0.85630229674426, 0.497582023798831, -0.138414255706431,
                      0.281404038139784, 0.67421809131173, 0.682818960388909);

         _t = Vec3d(1.826751712278038,  1.264718492450820,  0.195080809998819);
         _n = Vec3d(0.244875830334816, 0.480857890778889, 0.841909446789566);
    }

    bool containsValidMotion(std::vector<Mat>& rotations,
                             std::vector<Mat>& translations,
                             std::vector<Mat>& normals
                             )
    {
        double max_error = 1.0e-3;

        vector<Mat>::iterator riter = rotations.begin();
        vector<Mat>::iterator titer = translations.begin();
        vector<Mat>::iterator niter = normals.begin();

        for (;
             riter != rotations.end() && titer != translations.end() && niter != normals.end();
             ++riter, ++titer, ++niter) {

            double rdist = cvtest::norm(*riter, _R, NORM_INF);
            double tdist = cvtest::norm(*titer, _t, NORM_INF);
            double ndist = cvtest::norm(*niter, _n, NORM_INF);

            if (   rdist < max_error
                && tdist < max_error
                && ndist < max_error )
                return true;
        }

        return false;
    }

    Matx33d _R, _K, _H;
    Vec3d _t, _n;
};

TEST(Calib3d_DecomposeHomography, regression) { CV_HomographyDecompTest test; test.safe_run(); }


TEST(Calib3d_DecomposeHomography, issue_4978)
{
    Matx33d K(
        1.0,   0.0,    0.0,
        0.0,   1.0,    0.0,
        0.0,   0.0,    1.0
    );

    Matx33d H(
        -0.102896, 0.270191,   -0.0031153,
        0.0406387, 1.19569,    -0.0120456,
        0.445351,  0.0410889,  1
    );

    vector<Mat> rotations;
    vector<Mat> translations;
    vector<Mat> normals;

    decomposeHomographyMat(H, K, rotations, translations, normals);

    ASSERT_GT(rotations.size(), (size_t)0u);
    for (size_t i = 0; i < rotations.size(); i++)
    {
        // check: det(R) = 1
        EXPECT_TRUE(std::fabs(cv::determinant(rotations[i]) - 1.0) < 0.01)
            << "R: det=" << cv::determinant(rotations[0]) << std::endl << rotations[i] << std::endl
            << "T:" << std::endl << translations[i] << std::endl;
    }
}


}} // namespace
