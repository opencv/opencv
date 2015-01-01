#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <iostream>

using namespace cv;

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: ./lucid image.jpg\n";
        return 1;
    }

    Mat image, buf;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    if (!image.data) {
        std::cout << "Could not open or find the image\n";
        return 1;
    }

    cvtColor(image, buf, CV_BGR2GRAY);

    std::vector<KeyPoint> kpt;
    std::vector<std::vector<std::size_t> > dsc;

    FAST(buf, kpt, 9, 1);
    KeyPointsFilter::retainBest(kpt, 100);

    LUCID(image, kpt, dsc, 1, 2);

    for (std::size_t i = 0; i < dsc.size(); ++i) {
        for (std::size_t x = 0; x < dsc[i].size(); ++x) {
            std::cout << "Descriptor #" << i << ", Value #" << x << ": " << dsc[i][x] << "\n";
        }
    }

    separable_blur(image, buf, 6);

    namedWindow("Source image", WINDOW_AUTOSIZE);
    imshow("Source image", image);

    namedWindow("Blurred image", WINDOW_AUTOSIZE);
    imshow("Blurred image", buf);

    waitKey(0);

    return 0;
}
