#include "test_precomp.hpp"

//CV_TEST_MAIN("cv")

int main(int, char**)
{
    cv::Matx33f m = cv::Matx33f::eye();
    cv::Matx33f::diag_type d = m.diag();
    std::cout << "diag: " << cv::Mat(d) << std::endl;
    cv::Matx33f n = cv::Matx33f::diag( d );
    std::cout << "diag matrix: " << cv::Mat(n) << std::endl;
    
    cv::Point2f     p( 1.f, 2.f );
    cv::Size2f      s( 3.f, 4.f );
    cv::RotatedRect rr( p, s, 5.f );
    cv::Point2f     pts[4];
    
    rr.points( pts );
    for( int i = 0; i < 4; i++ )
        std::cout << pts[i].x << " " << pts[i].y << std::endl;
    
    return 0;
}

