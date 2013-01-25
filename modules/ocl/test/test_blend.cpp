#include "precomp.hpp"
#include <iomanip>

using namespace cv;
using namespace cv::ocl;
using namespace cvtest;
using namespace testing;
using namespace std;
#ifdef HAVE_OPENCL
template <typename T>
void blendLinearGold(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &weights1, const cv::Mat &weights2, cv::Mat &result_gold)
{
    result_gold.create(img1.size(), img1.type());

    int cn = img1.channels();

    for (int y = 0; y < img1.rows; ++y)
    {
        const float *weights1_row = weights1.ptr<float>(y);
        const float *weights2_row = weights2.ptr<float>(y);
        const T *img1_row = img1.ptr<T>(y);
        const T *img2_row = img2.ptr<T>(y);
        T *result_gold_row = result_gold.ptr<T>(y);

        for (int x = 0; x < img1.cols * cn; ++x)
        {
            float w1 = weights1_row[x / cn];
            float w2 = weights2_row[x / cn];
            result_gold_row[x] = static_cast<T>((img1_row[x] * w1 + img2_row[x] * w2) / (w1 + w2 + 1e-5f));
        }
    }
}

PARAM_TEST_CASE(Blend, cv::Size, MatType/*, UseRoi*/)
{
    //std::vector<cv::ocl::Info> oclinfo;
    cv::Size size;
    int type;
    bool useRoi;

    virtual void SetUp()
    {
        //devInfo = GET_PARAM(0);
        size = GET_PARAM(0);
        type = GET_PARAM(1);
        /*useRoi = GET_PARAM(3);*/

        //int devnums = getDevice(oclinfo, OPENCV_DEFAULT_OPENCL_DEVICE);
        //CV_Assert(devnums > 0);
    }
};

TEST_P(Blend, Accuracy)
{
    int depth = CV_MAT_DEPTH(type);

    cv::Mat img1 = randomMat(size, type, 0.0, depth == CV_8U ? 255.0 : 1.0);
    cv::Mat img2 = randomMat(size, type, 0.0, depth == CV_8U ? 255.0 : 1.0);
    cv::Mat weights1 = randomMat(size, CV_32F, 0, 1);
    cv::Mat weights2 = randomMat(size, CV_32F, 0, 1);

    cv::ocl::oclMat gimg1(size, type), gimg2(size, type), gweights1(size, CV_32F), gweights2(size, CV_32F);
    cv::ocl::oclMat dst(size, type);
    gimg1.upload(img1);
    gimg2.upload(img2);
    gweights1.upload(weights1);
    gweights2.upload(weights2);
    cv::ocl::blendLinear(gimg1, gimg2, gweights1, gweights2, dst);
    cv::Mat result;
    cv::Mat result_gold;
    dst.download(result);
    if (depth == CV_8U)
        blendLinearGold<uchar>(img1, img2, weights1, weights2, result_gold);
    else
        blendLinearGold<float>(img1, img2, weights1, weights2, result_gold);

    EXPECT_MAT_NEAR(result_gold, result, CV_MAT_DEPTH(type) == CV_8U ? 1.f : 1e-5f, 0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, Blend, Combine(
                            DIFFERENT_SIZES,
                            testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4), MatType(CV_32FC1), MatType(CV_32FC4))
                        ));
#endif