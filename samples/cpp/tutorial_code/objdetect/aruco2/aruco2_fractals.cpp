#include <opencv2/objdetect/aruco2.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

int main()
{
    //! [create_fractal]
    Mat image;
    aruco2::getFractalImage(image, aruco2::FRACTAL_3L_6);
    imwrite("fractal.png", image);
    //! [create_fractal]
    //!
    //add some blur to make it more realistic. Otherwise inner corners will not be
    //properly detected due to the perfect black and white borders
    GaussianBlur(image, image, Size(3, 3), 0); //not needed in real scenarios, but helps in this synthetic example

    std::cout << "Fractal image size: " << image.cols << "x" << image.rows << std::endl;

    //! [detect_fractals]
    auto fractals = aruco2::detectFractals(image, aruco2::FRACTAL_3L_6);

    for (const auto &f : fractals) {
        std::cout << "Detected fractal marker ID: " << f.id << std::endl;
    }
    //! [detect_fractals]

    //! [draw_fractals]
    Mat colorImage;
    cvtColor(image, colorImage, COLOR_GRAY2BGR);
    aruco2::drawFractals(colorImage, fractals);
    imshow("Detected Fractals", colorImage);
    waitKey(0);
    //! [draw_fractals]

    return 0;
}
