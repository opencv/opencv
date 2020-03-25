//! [includes]
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;
//! [includes]

int main()
{
    //! [imread]
    Mat img;
    img = imread(samples::findFile("starry_night.jpg"));
    //! [imread]

    //! [empty]
    if(img.empty())
    {
        std::cout << "Could not read the image." << std::endl ;
        return -1;
    }
    //! [empty]

    //! [imshow]
    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    //! [imshow]

    //! [imsave]
    if(k == 's')
    {
        imwrite("starry_night.png", img);
    }
    //! [imsave]

    return 0;
}
