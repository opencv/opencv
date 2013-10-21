#include "test_precomp.hpp"
#include <iomanip>

#ifdef HAVE_OPENCL

using namespace cv;
using namespace cv::ocl;
using namespace cvtest;
using namespace testing;
PARAM_TEST_CASE(MomentsTest, MatType, bool)
{
    int type;
    cv::Mat mat1;
    bool test_contours;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        test_contours = GET_PARAM(1);
        cv::Size size(10*MWIDTH, 10*MHEIGHT);
        mat1 = randomMat(size, type, 5, 16, false);
    }

    void Compare(Moments& cpu_moments, Moments& gpu_moments)
    {
        Mat gpu_dst, cpu_dst;
        HuMoments(cpu_moments, cpu_dst);
        HuMoments(gpu_moments, gpu_dst);
        EXPECT_MAT_NEAR(gpu_dst, cpu_dst, .5);
    }

};


OCL_TEST_P(MomentsTest, Mat)
{
    bool binaryImage = 0;

    for(int j = 0; j < LOOP_TIMES; j++)
    {
        if(test_contours)
        {
            Mat src = readImage( "cv/shared/pic3.png", IMREAD_GRAYSCALE );
            ASSERT_FALSE(src.empty());
            Mat canny_output;
            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;
            Canny( src, canny_output, 100, 200, 3 );
            findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
            for( size_t i = 0; i < contours.size(); i++ )
            {
                Moments m = moments( contours[i], false );
                Moments dm = ocl::ocl_moments( contours[i], false );
                Compare(m, dm);
            }
        }
        cv::_InputArray _array(mat1);
        cv::Moments CvMom = cv::moments(_array, binaryImage);
        cv::Moments oclMom = cv::ocl::ocl_moments(_array, binaryImage);

        Compare(CvMom, oclMom);
    }
}
INSTANTIATE_TEST_CASE_P(OCL_ImgProc, MomentsTest, Combine(
                            Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_64FC1), Values(true,false)));

#endif // HAVE_OPENCL
