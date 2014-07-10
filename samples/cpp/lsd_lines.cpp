#include <iostream>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    std::string in;
    if (argc != 2)
    {
        std::cout << "Usage: lsd_lines [input image]. Now loading building.jpg" << std::endl;
        in = "building.jpg";
    }
    else
    {
        in = argv[1];
    }

    Mat image = imread(in, IMREAD_GRAYSCALE);

#if 0
    Canny(image, image, 50, 200, 3); // Apply canny edge
#endif

    // Create and LSD detector with standard or no refinement.
#if 1
    Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
#else
    Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_NONE);
#endif

    double start = double(getTickCount());
    vector<Vec4i> lines_std;

    // Detect the lines
    ls->detect(image, lines_std);

    double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
    std::cout << "It took " << duration_ms << " ms." << std::endl;

    // Show found lines
    Mat drawnLines(image);
    ls->drawSegments(drawnLines, lines_std);
    imshow("Standard refinement", drawnLines);

    waitKey();
    return 0;
}
