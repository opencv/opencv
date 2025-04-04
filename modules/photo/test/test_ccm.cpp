// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test
{
namespace
{

Mat s = (Mat_<Vec3d>(24, 1) <<
        Vec3d(214.11, 98.67, 37.97),
        Vec3d(231.94, 153.1, 85.27),
        Vec3d(204.08, 143.71, 78.46),
        Vec3d(190.58, 122.99, 30.84),
        Vec3d(230.93, 148.46, 100.84),
        Vec3d(228.64, 206.97, 97.5),
        Vec3d(229.09, 137.07, 55.29),
        Vec3d(189.21, 111.22, 92.66),
        Vec3d(223.5, 96.42, 75.45),
        Vec3d(201.82, 69.71, 50.9),
        Vec3d(240.52, 196.47, 59.3),
        Vec3d(235.73, 172.13, 54.),
        Vec3d(131.6, 75.04, 68.86),
        Vec3d(189.04, 170.43, 42.05),
        Vec3d(222.23, 74., 71.95),
        Vec3d(241.01, 199.1, 61.15),
        Vec3d(224.99, 101.4, 100.24),
        Vec3d(174.58, 152.63, 91.52),
        Vec3d(248.06, 227.69, 140.5),
        Vec3d(241.15, 201.38, 115.58),
        Vec3d(236.49, 175.87, 88.86),
        Vec3d(212.19, 133.49, 54.79),
        Vec3d(181.17, 102.94, 36.18),
        Vec3d(115.1, 53.77, 15.23));

TEST(Photo_ColorCorrection, test_model)
{

    ColorCorrectionModel model(s / 255, COLORCHECKER_Macbeth);
    model.computeCCM();
    Mat src_rgbl = (Mat_<Vec3d>(24, 1) <<
        Vec3d(0.68078957, 0.12382801, 0.01514889),
        Vec3d(0.81177942, 0.32550452, 0.089818),
        Vec3d(0.61259378, 0.2831933, 0.07478902),
        Vec3d(0.52696493, 0.20105976, 0.00958657),
        Vec3d(0.80402284, 0.30419523, 0.12989841),
        Vec3d(0.78658646, 0.63184111, 0.12062068),
        Vec3d(0.78999637, 0.25520249, 0.03462853),
        Vec3d(0.51866697, 0.16114393, 0.1078387),
        Vec3d(0.74820768, 0.11770076, 0.06862177),
        Vec3d(0.59776825, 0.05765816, 0.02886627),
        Vec3d(0.8793145, 0.56346033, 0.0403954),
        Vec3d(0.84124847, 0.42120746, 0.03287592),
        Vec3d(0.23333214, 0.06780408, 0.05612276),
        Vec3d(0.5176423, 0.41210976, 0.01896255),
        Vec3d(0.73888613, 0.06575388, 0.06181293),
        Vec3d(0.88326036, 0.58018751, 0.04321991),
        Vec3d(0.75922531, 0.13149072, 0.1282041),
        Vec3d(0.4345097, 0.32331019, 0.10494139),
        Vec3d(0.94110142, 0.77941419, 0.26946323),
        Vec3d(0.88438952, 0.5949049 , 0.17536928),
        Vec3d(0.84722687, 0.44160449, 0.09834799),
        Vec3d(0.66743106, 0.24076803, 0.03394333),
        Vec3d(0.47141286, 0.13592419, 0.01362205),
        Vec3d(0.17377101, 0.03256864, 0.00203026));
    EXPECT_MAT_NEAR(src_rgbl, model.get_src_rgbl(), 1e-4);

    Mat dst_rgbl = (Mat_<Vec3d>(24, 1) <<
        Vec3d(0.17303173, 0.08211037, 0.05672686),
        Vec3d(0.56832031, 0.29269488, 0.21835529),
        Vec3d(0.10365019, 0.19588357, 0.33140475),
        Vec3d(0.10159676, 0.14892193, 0.05188294),
        Vec3d(0.22159627, 0.21584476, 0.43461196),
        Vec3d(0.10806379, 0.51437196, 0.41264213),
        Vec3d(0.74736423, 0.20062878, 0.02807988),
        Vec3d(0.05757947, 0.10516793, 0.40296109),
        Vec3d(0.56676218, 0.08424805, 0.11969461),
        Vec3d(0.11099515, 0.04230796, 0.14292554),
        Vec3d(0.34546869, 0.50872001, 0.04944204),
        Vec3d(0.79461323, 0.35942459, 0.02051968),
        Vec3d(0.01710416, 0.05022043, 0.29220674),
        Vec3d(0.05598012, 0.30021149, 0.06871162),
        Vec3d(0.45585457, 0.03033727, 0.04085654),
        Vec3d(0.85737614, 0.56757335, 0.0068503),
        Vec3d(0.53348585, 0.08861148, 0.30750446),
        Vec3d(-0.0374061, 0.24699498, 0.40041217),
        Vec3d(0.91262695, 0.91493909, 0.89367049),
        Vec3d(0.57981916, 0.59200418, 0.59328881),
        Vec3d(0.35490581, 0.36544831, 0.36755375),
        Vec3d(0.19007357, 0.19186587, 0.19308397),
        Vec3d(0.08529188, 0.08887994, 0.09257601),
        Vec3d(0.0303193, 0.03113818, 0.03274845));
    EXPECT_MAT_NEAR(dst_rgbl, model.get_dst_rgbl(), 1e-4);

    Mat mask = Mat::ones(24, 1, CV_8U);
    EXPECT_MAT_NEAR(model.getMask(), mask, 0.0);


    Mat ccm = (Mat_<double>(3, 3) <<
    0.37406520, 0.02066507, 0.05804047,
    0.12719672, 0.77389268, -0.01569404,
    -0.27627010, 0.00603427, 2.74272981);
    EXPECT_MAT_NEAR(model.getCCM(), ccm, 1e-4);
}
TEST(Photo_ColorCorrection, test_masks_weights_1)
{
    Mat weights_list_ = (Mat_<double>(24, 1) <<
                            1.1, 0, 0, 1.2, 0, 0,
                            1.3, 0, 0, 1.4, 0, 0,
                            0.5, 0, 0, 0.6, 0, 0,
                            0.7, 0, 0, 0.8, 0, 0);
    ColorCorrectionModel model1(s / 255,COLORCHECKER_Macbeth);
    model1.setColorSpace(COLOR_SPACE_sRGB);
    model1.setCCMType(CCM_LINEAR);
    model1.setDistance(DISTANCE_CIE2000);
    model1.setLinear(LINEARIZATION_GAMMA);
    model1.setLinearGamma(2.2);
    model1.setLinearDegree(3);
    model1.setSaturatedThreshold(0, 0.98);
    model1.setWeightsList(weights_list_);
    model1.setWeightCoeff(1.5);
    model1.computeCCM();
    Mat weights = (Mat_<double>(8, 1) <<
                            1.15789474, 1.26315789, 1.36842105, 1.47368421,
                            0.52631579, 0.63157895, 0.73684211, 0.84210526);
    EXPECT_MAT_NEAR(model1.getWeights(), weights, 1e-4);

    Mat mask = (Mat_<uchar>(24, 1) <<
                            true, false, false, true, false, false,
                            true, false, false, true, false, false,
                            true, false, false, true, false, false,
                            true, false, false, true, false, false);
    EXPECT_MAT_NEAR(model1.getMask(), mask, 0.0);
}

TEST(Photo_ColorCorrection, test_masks_weights_2)
{
    ColorCorrectionModel model2(s / 255, COLORCHECKER_Macbeth);
    model2.setCCMType(CCM_LINEAR);
    model2.setDistance(DISTANCE_CIE2000);
    model2.setLinear(LINEARIZATION_GAMMA);
    model2.setLinearGamma(2.2);
    model2.setLinearDegree(3);
    model2.setSaturatedThreshold(0.05, 0.93);
    model2.setWeightsList(Mat());
    model2.setWeightCoeff(1.5);
    model2.computeCCM();
    Mat weights = (Mat_<double>(20, 1) <<
                            0.65554256, 1.49454705, 1.00499244, 0.79735434, 1.16327759,
                            1.68623868, 1.37973155, 0.73213388, 1.0169629, 0.47430246,
                            1.70312161, 0.45414218, 1.15910007, 0.7540434, 1.05049802,
                            1.04551645, 1.54082353, 1.02453421, 0.6015915, 0.26154558);
    EXPECT_MAT_NEAR(model2.getWeights(), weights, 1e-4);

    Mat mask = (Mat_<uchar>(24, 1) <<
                            true, true, true, true, true, true,
                            true, true, true, true, false, true,
                            true, true, true, false, true, true,
                            false, false, true, true, true, true);
    EXPECT_MAT_NEAR(model2.getMask(), mask, 0.0);
}

} // namespace
} // namespace opencv_test