#include <opencv2/objdetect/aruco2.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

int main()
{
    //! [create_fractal]
    Mat fractalImage;
    aruco2::getFractalImage(fractalImage, aruco2::FRACTAL_3L_6);
    imwrite("fractal.png", fractalImage);
    //! [create_fractal]

    std::cout << "Fractal image size: " << fractalImage.cols << "x" << fractalImage.rows << std::endl;

    // Place fractal on a white scene for detection
    Mat scene(fractalImage.rows + 100, fractalImage.cols + 100, CV_8UC1, Scalar(255));
    fractalImage.copyTo(scene(Rect(50, 50, fractalImage.cols, fractalImage.rows)));

    //! [detect_fractals]
    Mat image = scene.clone();
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
