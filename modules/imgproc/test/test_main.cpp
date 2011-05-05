#include "test_precomp.hpp"

//CV_TEST_MAIN("cv")

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int
main(int argc, char *argv[])
{
    cv::Mat src_img = cv::imread("/Users/vp/Downloads/lenna.png", 1);
    if(!src_img.data) return -1;
    
    cv::Point2f pts1[] = {cv::Point2f(150,150.),cv::Point2f(150,300.),cv::Point2f(350,300.),cv::Point2f(350,150.)};
    cv::Point2f pts2[] = {cv::Point2f(200,200.),cv::Point2f(150,300.),cv::Point2f(350,300.),cv::Point2f(300,200.)};
    
    cv::Mat perspective_matrix = cv::getPerspectiveTransform(pts1, pts2);
    cv::Mat dst_img;
    dst_img = cv::Scalar::all(0);
    cv::warpPerspective(src_img, dst_img, perspective_matrix, src_img.size(), cv::INTER_LANCZOS4);
    
    cv::namedWindow("src", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
    cv::namedWindow("dst", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
    cv::imshow("src", src_img);
    cv::imshow("dst", dst_img);
    cv::waitKey(0);
}

