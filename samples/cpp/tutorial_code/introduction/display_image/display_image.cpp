//! [includes]
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
//! [includes]

int main()
{
    //! [imread]
    cv::Mat img;
    img = cv::imread(cv::samples::findFile("starry_night.jpg"));
    //! [imread]

    //! [empty]
    if(img.empty())
    {
        std::cout << "Could not read the image." << std::endl ;
        return -1;
    }
    //! [empty]

    //! [imshow]
    cv::imshow("Display window", img);
    int k = cv::waitKey(0); // Wait for a keystroke in the window
    //! [imshow]

    //! [imsave]
    if(k == 's')
    {
        cv::imwrite("starry_night.png", img);
    }
    //! [imsave]

    return 0;
}
