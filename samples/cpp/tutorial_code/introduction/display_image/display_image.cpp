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
    std::string image_path = samples::findFile("starry_night.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);
    //! [imread]

    //! [empty]
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    //! [empty]

    //! [imshow]
// Display the loaded image in a window on the screen

    imshow("Display window", img);    
    int k = waitKey(0); // Wait for a keystroke in the window
if (k < 0)
{
    std::cout << "No key was pressed. Exiting." << std::endl;
    return 0;
}


    //! [imshow]

    //! [imsave]
    if(k == 's')
    {
        imwrite("starry_night.png", img);
    }
    //! [imsave]

    return 0;
// End of display_image tutorial
}
