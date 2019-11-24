// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

CV_ENUM(MatchTemplType, CV_TM_CCORR,  CV_TM_CCORR_NORMED,
                        CV_TM_SQDIFF, CV_TM_SQDIFF_NORMED,
                        CV_TM_CCOEFF, CV_TM_CCOEFF_NORMED)

class Imgproc_MatchTemplateWithMask : public TestWithParam<std::tuple<MatType,MatType>>
{
protected:
    // Member functions inherited from ::testing::Test
    void SetUp() override;

    // Matrices for test calculations (always CV_32)
    Mat img_;
    Mat templ_;
    Mat mask_;
    Mat templ_masked_;
    Mat img_roi_masked_;
    // Matrices for call to matchTemplate (have test type)
    Mat img_testtype_;
    Mat templ_testtype_;
    Mat mask_testtype_;
    Mat result_;

    // Constants
    static const Size IMG_SIZE;
    static const Size TEMPL_SIZE;
    static const Point TEST_POINT;
};

// Arbitraryly chosen test constants
const Size  Imgproc_MatchTemplateWithMask::IMG_SIZE(160, 100);
const Size  Imgproc_MatchTemplateWithMask::TEMPL_SIZE(21, 13);
const Point Imgproc_MatchTemplateWithMask::TEST_POINT(8, 9);

void Imgproc_MatchTemplateWithMask::SetUp()
{
    int type = std::get<0>(GetParam());
    int type_mask = std::get<1>(GetParam());

    // Matrices are created with the depth to test (for the call to matchTemplate()), but are also
    // converted to CV_32 for the test calculations, because matchTemplate() also only operates on
    // and returns CV_32.
    img_testtype_.create(IMG_SIZE, type);
    templ_testtype_.create(TEMPL_SIZE, type);
    mask_testtype_.create(TEMPL_SIZE, type_mask);

    randu(img_testtype_, 0, 10);
    randu(templ_testtype_, 0, 10);
    randu(mask_testtype_, 0, 5);

    img_testtype_.convertTo(img_, CV_32F);
    templ_testtype_.convertTo(templ_, CV_32F);
    mask_testtype_.convertTo(mask_, CV_32F);
    if (CV_MAT_DEPTH(type_mask) == CV_8U)
    {
        // CV_8U masks are interpreted as binary masks
        mask_.setTo(Scalar::all(1), mask_ != 0);
    }
    if (mask_.channels() != templ_.channels())
    {
        std::vector<Mat> mask_channels(templ_.channels(), mask_);
        merge(mask_channels.data(), templ_.channels(), mask_);
    }

    Rect roi(TEST_POINT, TEMPL_SIZE);
    img_roi_masked_ = img_(roi).mul(mask_);
    templ_masked_ = templ_.mul(mask_);
}

TEST_P(Imgproc_MatchTemplateWithMask, CompareNaiveImplSQDIFF)
{
    matchTemplate(img_testtype_, templ_testtype_, result_, CV_TM_SQDIFF, mask_testtype_);
    // Naive implementation for one point
    Mat temp = img_roi_masked_ - templ_masked_;
    Scalar temp_s = sum(temp.mul(temp));
    double val = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];

    EXPECT_NEAR(val, result_.at<float>(TEST_POINT), TEMPL_SIZE.area()*abs(val)*FLT_EPSILON);
}

TEST_P(Imgproc_MatchTemplateWithMask, CompareNaiveImplSQDIFF_NORMED)
{
    matchTemplate(img_testtype_, templ_testtype_, result_, CV_TM_SQDIFF_NORMED, mask_testtype_);
    // Naive implementation for one point
    Mat temp = img_roi_masked_ - templ_masked_;
    Scalar temp_s = sum(temp.mul(temp));
    double val = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];

    // Normalization
    temp_s = sum(templ_masked_.mul(templ_masked_));
    double norm = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];
    temp_s = sum(img_roi_masked_.mul(img_roi_masked_));
    norm *= temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];
    norm = sqrt(norm);
    val /= norm;

    EXPECT_NEAR(val, result_.at<float>(TEST_POINT), TEMPL_SIZE.area()*abs(val)*FLT_EPSILON);
}

TEST_P(Imgproc_MatchTemplateWithMask, CompareNaiveImplCCORR)
{
    matchTemplate(img_testtype_, templ_testtype_, result_, CV_TM_CCORR, mask_testtype_);
    // Naive implementation for one point
    Scalar temp_s = sum(templ_masked_.mul(img_roi_masked_));
    double val = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];

    EXPECT_NEAR(val, result_.at<float>(TEST_POINT), TEMPL_SIZE.area()*abs(val)*FLT_EPSILON);
}

TEST_P(Imgproc_MatchTemplateWithMask, CompareNaiveImplCCORR_NORMED)
{
    matchTemplate(img_testtype_, templ_testtype_, result_, CV_TM_CCORR_NORMED, mask_testtype_);
    // Naive implementation for one point
    Scalar temp_s = sum(templ_masked_.mul(img_roi_masked_));
    double val = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];

    // Normalization
    temp_s = sum(templ_masked_.mul(templ_masked_));
    double norm = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];
    temp_s = sum(img_roi_masked_.mul(img_roi_masked_));
    norm *= temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];
    norm = sqrt(norm);
    val /= norm;

    EXPECT_NEAR(val, result_.at<float>(TEST_POINT), TEMPL_SIZE.area()*abs(val)*FLT_EPSILON);
}

TEST_P(Imgproc_MatchTemplateWithMask, CompareNaiveImplCCOEFF)
{
    matchTemplate(img_testtype_, templ_testtype_, result_, CV_TM_CCOEFF, mask_testtype_);
    // Naive implementation for one point
    Scalar temp_s = sum(mask_);
    for (int i = 0; i < 4; i++)
    {
        if (temp_s[i] != 0.0)
            temp_s[i] = 1.0 / temp_s[i];
        else
            temp_s[i] = 1.0;
    }
    Mat temp = mask_.clone(); temp = temp_s; // Workaround to multiply Mat by Scalar
    Mat temp2 = mask_.clone(); temp2 = sum(templ_masked_); // Workaround to multiply Mat by Scalar
    Mat templx = templ_masked_ - mask_.mul(temp).mul(temp2);
    temp2 = sum(img_roi_masked_); // Workaround to multiply Mat by Scalar
    Mat imgx = img_roi_masked_ - mask_.mul(temp).mul(temp2);
    temp_s = sum(templx.mul(imgx));
    double val = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];

    EXPECT_NEAR(val, result_.at<float>(TEST_POINT), TEMPL_SIZE.area()*abs(val)*FLT_EPSILON);
}

TEST_P(Imgproc_MatchTemplateWithMask, CompareNaiveImplCCOEFF_NORMED)
{
    matchTemplate(img_testtype_, templ_testtype_, result_, CV_TM_CCOEFF_NORMED, mask_testtype_);
    // Naive implementation for one point
    Scalar temp_s = sum(mask_);
    for (int i = 0; i < 4; i++)
    {
        if (temp_s[i] != 0.0)
            temp_s[i] = 1.0 / temp_s[i];
        else
            temp_s[i] = 1.0;
    }
    Mat temp = mask_.clone(); temp = temp_s; // Workaround to multiply Mat by Scalar
    Mat temp2 = mask_.clone(); temp2 = sum(templ_masked_); // Workaround to multiply Mat by Scalar
    Mat templx = templ_masked_ - mask_.mul(temp).mul(temp2);
    temp2 = sum(img_roi_masked_); // Workaround to multiply Mat by Scalar
    Mat imgx = img_roi_masked_ - mask_.mul(temp).mul(temp2);
    temp_s = sum(templx.mul(imgx));
    double val = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];

    // Normalization
    temp_s = sum(templx.mul(templx));
    double norm = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];
    temp_s = sum(imgx.mul(imgx));
    norm *= temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];
    norm = sqrt(norm);
    val /= norm;

    EXPECT_NEAR(val, result_.at<float>(TEST_POINT), TEMPL_SIZE.area()*abs(val)*FLT_EPSILON);
}

INSTANTIATE_TEST_CASE_P(SingleChannelMask, Imgproc_MatchTemplateWithMask,
    Combine(
        Values(CV_32FC1, CV_32FC3, CV_8UC1, CV_8UC3),
        Values(CV_32FC1, CV_8UC1)));

INSTANTIATE_TEST_CASE_P(MultiChannelMask, Imgproc_MatchTemplateWithMask,
    Combine(
        Values(CV_32FC3, CV_8UC3),
        Values(CV_32FC3, CV_8UC3)));

class Imgproc_MatchTemplateWithMask2 : public TestWithParam<std::tuple<MatType,MatType,
                                                                       MatchTemplType>>
{
protected:
    // Member functions inherited from ::testing::Test
    void SetUp() override;

    // Data members
    Mat img_;
    Mat templ_;
    Mat mask_;
    Mat result_withoutmask_;
    Mat result_withmask_;

    // Constants
    static const Size IMG_SIZE;
    static const Size TEMPL_SIZE;
};

// Arbitraryly chosen test constants
const Size  Imgproc_MatchTemplateWithMask2::IMG_SIZE(160, 100);
const Size  Imgproc_MatchTemplateWithMask2::TEMPL_SIZE(21, 13);

void Imgproc_MatchTemplateWithMask2::SetUp()
{
    int type = std::get<0>(GetParam());
    int type_mask = std::get<1>(GetParam());

    img_.create(IMG_SIZE, type);
    templ_.create(TEMPL_SIZE, type);
    mask_.create(TEMPL_SIZE, type_mask);

    randu(img_, 0, 100);
    randu(templ_, 0, 100);

    if (CV_MAT_DEPTH(type_mask) == CV_8U)
    {
        // CV_8U implies binary mask, so all nonzero values should work
        randu(mask_, 1, 255);
    }
    else
    {
        mask_ = Scalar(1, 1, 1, 1);
    }
}

TEST_P(Imgproc_MatchTemplateWithMask2, CompareWithAndWithoutMask)
{
    int method = std::get<2>(GetParam());

    matchTemplate(img_, templ_, result_withmask_, method, mask_);
    matchTemplate(img_, templ_, result_withoutmask_, method);

    // Get maximum result for relative error calculation
    double min_val, max_val;
    minMaxLoc(abs(result_withmask_), &min_val, &max_val);

    // Get maximum of absolute diff for comparison
    double mindiff, maxdiff;
    minMaxLoc(abs(result_withmask_ - result_withoutmask_), &mindiff, &maxdiff);

    EXPECT_LT(maxdiff, max_val*TEMPL_SIZE.area()*FLT_EPSILON);
}


INSTANTIATE_TEST_CASE_P(SingleChannelMask, Imgproc_MatchTemplateWithMask2,
    Combine(
        Values(CV_32FC1, CV_32FC3, CV_8UC1, CV_8UC3),
        Values(CV_32FC1, CV_8UC1),
        Values(CV_TM_SQDIFF, CV_TM_SQDIFF_NORMED, CV_TM_CCORR, CV_TM_CCORR_NORMED,
               CV_TM_CCOEFF, CV_TM_CCOEFF_NORMED)));

INSTANTIATE_TEST_CASE_P(MultiChannelMask, Imgproc_MatchTemplateWithMask2,
    Combine(
        Values(CV_32FC3, CV_8UC3),
        Values(CV_32FC3, CV_8UC3),
        Values(CV_TM_SQDIFF, CV_TM_SQDIFF_NORMED, CV_TM_CCORR, CV_TM_CCORR_NORMED,
               CV_TM_CCOEFF, CV_TM_CCOEFF_NORMED)));


TEST(Imgproc_TemplMatch, accuracy_CrossCorr_OneChannel)
{
    float img_data[5][5] =
        { { 4.0, 5.0, 0.0, 3.0, 3.0 },
          { 3.0, 1.0, 3.0, 5.0, 2.0 },
          { 4.0, 0.0, 0.0, 4.0, 2.0 },
          { 1.0, 0.0, 1.0, 5.0, 1.0 },
          { 5.0, 0.0, 1.0, 4.0, 3.0 } };
    Mat img(5, 5, CV_32FC1, img_data);

    float templ_data[3][3] =
        { { 0.0, 3.0, 5.0 },
          { 0.0, 2.0, 3.0 },
          { 0.0, 1.0, 3.0 } };
    Mat templ(3, 3, CV_32FC1, templ_data);

    float expected_result_data[3][3] =
        { { 26.0, 48.0, 50.0 },
          { 21.0, 62.0, 47.0 },
          {  6.0, 50.0, 48.0 } };
    Mat expected_result(3, 3, CV_32FC1, expected_result_data);

    Mat result(3, 3, CV_32FC1);

    cv::crossCorr(img, templ, result, Point(0,0), 0, 0);

    // reshape for comparison, because minMaxLoc only supports single channel
    Mat expected_result_cmp = expected_result.reshape(1);
    Mat result_cmp = result.reshape(1);

    // Get maximum result for relative error calculation
    double min, max;
    minMaxLoc(abs(expected_result_cmp), &min, &max);

    // Get maximum of absolute diff for comparison
    double mindiff, maxdiff;
    minMaxLoc(abs(expected_result_cmp - result_cmp), &mindiff, &maxdiff);

    EXPECT_LT(maxdiff, max*100.0*FLT_EPSILON) << "result =\n" << result
                                              << "\nexpected result =\n" << expected_result;
}

TEST(Imgproc_TemplMatch, accuracy_CrossCorr_ThreeChannels)
{
    float img_data[5][5][3] =
        { { { 4.0, 5.0, 0.0 }, { 3.0, 3.0, 3.0 }, { 1.0, 3.0, 5.0 }, { 2.0, 4.0, 0.0 }, { 0.0, 4.0, 2.0 } },
          { { 1.0, 0.0, 1.0 }, { 5.0, 1.0, 5.0 }, { 0.0, 1.0, 4.0 }, { 3.0, 0.0, 3.0 }, { 5.0, 0.0, 2.0 } },
          { { 3.0, 0.0, 1.0 }, { 3.0, 5.0, 3.0 }, { 3.0, 0.0, 1.0 }, { 1.0, 1.0, 0.0 }, { 2.0, 4.0, 3.0 } },
          { { 3.0, 2.0, 4.0 }, { 2.0, 0.0, 0.0 }, { 4.0, 5.0, 5.0 }, { 0.0, 4.0, 1.0 }, { 4.0, 1.0, 2.0 } },
          { { 2.0, 0.0, 1.0 }, { 1.0, 1.0, 1.0 }, { 3.0, 3.0, 2.0 }, { 3.0, 0.0, 3.0 }, { 5.0, 4.0, 1.0 } } };
    Mat img(5, 5, CV_32FC3, img_data);

    float templ_data[3][3][3] =
        { { { 2.0, 4.0, 3.0 }, { 4.0, 4.0, 4.0 }, { 3.0, 4.0, 4.0 } },
          { { 4.0, 0.0, 4.0 }, { 3.0, 2.0, 5.0 }, { 5.0, 5.0, 0.0 } },
          { { 1.0, 5.0, 1.0 }, { 3.0, 0.0, 5.0 }, { 0.0, 1.0, 2.0 } } };
    Mat templ(3, 3, CV_32FC3, templ_data);

    float expected_result_data[3][3][3] =
        { { { 54.0, 51.0, 79.0 }, { 63.0, 68.0, 77.0 }, { 50.0, 48.0, 61.0 } },
          { { 67.0, 33.0, 72.0 }, { 59.0, 17.0, 87.0 }, { 56.0, 52.0, 50.0 } },
          { { 70.0, 48.0, 45.0 }, { 51.0, 59.0, 55.0 }, { 64.0, 52.0, 59.0 } } };
    Mat expected_result(3, 3, CV_32FC3, expected_result_data);

    Mat result(3, 3, CV_32FC3);

    cv::crossCorr(img, templ, result, Point(0,0), 0, 0);

    // reshape for comparison, because minMaxLoc only supports single channel
    Mat expected_result_cmp = expected_result.reshape(1);
    Mat result_cmp = result.reshape(1);

    // Get maximum result for relative error calculation
    double min, max;
    minMaxLoc(abs(expected_result_cmp), &min, &max);

    // Get maximum of absolute diff for comparison
    double mindiff, maxdiff;
    minMaxLoc(abs(expected_result_cmp - result_cmp), &mindiff, &maxdiff);

    EXPECT_LT(maxdiff, max*100.0*FLT_EPSILON) << "result =\n" << result
                                              << "\nexpected result =\n" << expected_result;
}

TEST(Imgproc_TemplMatch, accuracy_CrossCorr_ThreeChannelsTemplOnes)
{
    float img_data[5][5][3] =
        { { { 4.0, 5.0, 0.0 }, { 3.0, 3.0, 3.0 }, { 1.0, 3.0, 5.0 }, { 2.0, 4.0, 0.0 }, { 0.0, 4.0, 2.0 } },
          { { 1.0, 0.0, 1.0 }, { 5.0, 1.0, 5.0 }, { 0.0, 1.0, 4.0 }, { 3.0, 0.0, 3.0 }, { 5.0, 0.0, 2.0 } },
          { { 3.0, 0.0, 1.0 }, { 3.0, 5.0, 3.0 }, { 3.0, 0.0, 1.0 }, { 1.0, 1.0, 0.0 }, { 2.0, 4.0, 3.0 } },
          { { 3.0, 2.0, 4.0 }, { 2.0, 0.0, 0.0 }, { 4.0, 5.0, 5.0 }, { 0.0, 4.0, 1.0 }, { 4.0, 1.0, 2.0 } },
          { { 2.0, 0.0, 1.0 }, { 1.0, 1.0, 1.0 }, { 3.0, 3.0, 2.0 }, { 3.0, 0.0, 3.0 }, { 5.0, 4.0, 1.0 } } };
    Mat img(5, 5, CV_32FC3, img_data);

    float templ_data[3][3][3] =
        { { { 1.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0 } },
          { { 1.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0 } },
          { { 1.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0 } } };
    Mat templ(3, 3, CV_32FC3, templ_data);

    float expected_result_data[3][3][3] =
        { { { 23.0, 18.0, 23.0 }, { 21.0, 18.0, 24.0 }, { 17.0, 17.0, 20.0 } },
          { { 24.0, 14.0, 24.0 }, { 21.0, 17.0, 22.0 }, { 22.0, 16.0, 21.0 } },
          { { 24.0, 16.0, 18.0 }, { 20.0, 19.0, 16.0 }, { 25.0, 22.0, 18.0 } } };
    Mat expected_result(3, 3, CV_32FC3, expected_result_data);

    Mat result(3, 3, CV_32FC3);

    cv::crossCorr(img, templ, result, Point(0,0), 0, 0);

    // reshape for comparison, because minMaxLoc only supports single channel
    Mat expected_result_cmp = expected_result.reshape(1);
    Mat result_cmp = result.reshape(1);

    // Get maximum result for relative error calculation
    double min, max;
    minMaxLoc(abs(expected_result_cmp), &min, &max);

    // Get maximum of absolute diff for comparison
    double mindiff, maxdiff;
    minMaxLoc(abs(expected_result_cmp - result_cmp), &mindiff, &maxdiff);

    EXPECT_LT(maxdiff, max*100.0*FLT_EPSILON) << "result =\n" << result
                                              << "\nexpected result =\n" << expected_result;
}

}} // namespace
