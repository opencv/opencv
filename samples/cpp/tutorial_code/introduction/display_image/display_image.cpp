//! [includes]
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#include <iostream>
//! [includes]

//! [namespace]
using namespace cv;
using namespace std;
//! [namespace]


/*int main( int argc, char** argv )
{
    //! [load]
    String imageName( "HappyFish.jpg" ); // by default
    if( argc > 1)
    {
        imageName = argv[1];
    }
    //! [load]

    //! [mat]
    Mat image;
    //! [mat]

    //! [imread]
    image = imread( samples::findFile( imageName ), IMREAD_COLOR ); // Read the file
    //! [imread]

    if( image.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    //! [window]
    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    //! [window]

    //! [imshow]
    imshow( "Display window", image );                // Show our image inside it.
    //! [imshow]

    //! [wait]
    waitKey(0); // Wait for a keystroke in the window
    //! [wait]
    return 0;
}*/

int main(int argc, char** argv)
{
    Mat src, src_gray, edges, standard_hough, pbb_hough;
    int min_threshold = 300;


    String imageName("building.jpg");
    if (argc > 1)
    {
        imageName = argv[1];
    }
 
    src = imread(samples::findFile(imageName), IMREAD_COLOR);

    if (src.empty())               
    {
        std::cerr << "Invalid input image\n";
        std::cout << "Usage : " << argv[0] << " <path_to_input_image>\n";;
        return -1;
    }

    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", src);
    waitKey(0);

    cvtColor(src, src_gray, COLOR_BGR2GRAY);
  
    namedWindow("cvtColor", WINDOW_AUTOSIZE); 
    imshow("cvtColor", src_gray);          
    waitKey(0);

    // Reduce the noise so we avoid false circle detection
    //GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);
      
    /*namedWindow("Blur", WINDOW_AUTOSIZE);
    imshow("Blur", src_gray);
    waitKey(0);*/

    Canny(src_gray, edges, 120, 200, 3);

    //namedWindow("Canny", WINDOW_AUTOSIZE); 
    imshow("Canny", edges);
    waitKey(0);

    vector<Vec2f> s_lines;
    standard_hough = src.clone();

    /// 1. Use Standard Hough Transform
    HoughLines(edges, s_lines, 1, CV_PI / 180, min_threshold+50, 0, 0);
    /// Show the result
    for (size_t i = 0; i < s_lines.size(); i++)
    {
        float r = s_lines[i][0], t = s_lines[i][1];
        double cos_t = cos(t), sin_t = sin(t);
        double x0 = r * cos_t, y0 = r * sin_t;
        double alpha = 1000;

        Point pt1(cvRound(x0 + alpha * (-sin_t)), cvRound(y0 + alpha * cos_t));
        Point pt2(cvRound(x0 - alpha * (-sin_t)), cvRound(y0 - alpha * cos_t));
        line(standard_hough, pt1, pt2, Scalar(100, 0, 155), 2, LINE_AA);
    }
    imshow("Standart_Hough", standard_hough);
    waitKey(0);

    vector<Vec4i> p_lines;
    pbb_hough = src.clone();
    /// 2. Use Probabilistic Hough Transform
    HoughLinesP(edges, p_lines, 1, CV_PI / 180, min_threshold, 30, 10);

    /// Show the result
    for (size_t i = 0; i < p_lines.size(); i++)
    {
        Vec4i l = p_lines[i];
        line(src_gray, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(155, 100, 0), 2, LINE_AA);
    }
    
    imshow("Pbb_Hough", src_gray);
    waitKey(0);
    return 0;
}
