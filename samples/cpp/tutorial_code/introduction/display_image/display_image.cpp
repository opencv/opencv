//! [includes]
#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace std;
using namespace cv;
//! [includes]

int main()
{
    std::vector< String > resultfileNames;
    cv::glob("C:/Images/*.*", resultfileNames,true);
    for (auto name : resultfileNames)
    {
        if (name.find(".ZIP") == string::npos &&
            name.find(".zip") == string::npos &&
            name.find(".rar") == string::npos &&
            name.find(".RAR") == string::npos &&
            name.find(".7z") == string::npos &&
            name.find(".7Z") == string::npos &&
            name.find(".TXT") == string::npos &&
            name.find(".txt") == string::npos)
            try
            {
                Mat src = imread(samples::findFile(name));
                cout << name << "***********" << endl;
                cout << src.rows << ", " << src.cols << ", " << src.channels() << ", " << src.depth() << endl;
                cout << "*********************************************************" << endl;
                int m = std::max(src.rows, src.cols);
                Mat img;
                resize(src, img, Size((src.cols * 400) / m, (src.rows * 400) / m));
                imshow("Source", img);
                waitKey(10);

            }
            catch (Exception e)
            {
                cout << "CANNOT READ FILE -- >>  " << name << endl;
            }
    }
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
