#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv,
                                 "{input   i|building.jpg|input image}"
                                 "{refine  r|false|if true use LSD_REFINE_STD method, if false use LSD_REFINE_NONE method}"
                                 "{canny   c|false|use Canny edge detector}"
                                 "{overlay o|false|show result on input image}"
                                 "{output  o|result.jpg|output image}"
                                 "{help    h|false|show help message}");

    if (parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }

    parser.printMessage();

    String filename = samples::findFile(parser.get<String>("input"));
    bool useRefine = parser.get<bool>("refine");
    bool useCanny = parser.get<bool>("canny");
    bool overlay = parser.get<bool>("overlay");
    String outputFilename = parser.get<String>("output");

    Mat image = imread(filename, IMREAD_GRAYSCALE);

    if( image.empty() )
    {
        cout << "Unable to load " << filename << endl;
        return 1;
    }

    if (useCanny)
    {
        Canny(image, image, 50, 200, 3); // Apply Canny edge detector
    }

    // Create LSD detector with standard or no refinement.
    Ptr<LineSegmentDetector> ls = useRefine ? createLineSegmentDetector(LSD_REFINE_STD) : createLineSegmentDetector(LSD_REFINE_NONE);

    double start = double(getTickCount());
    vector<Vec4f> lines_std;

    // Detect the lines
    ls->detect(image, lines_std);

    double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
    std::cout << "It took " << duration_ms << " ms." << std::endl;

    // Show found lines
    if (!overlay || useCanny)
    {
        image = Scalar(0, 0, 0);
    }

    ls->drawSegments(image, lines_std);

    // Save the result image
    imwrite(outputFilename, image);
    cout << "Result saved to " << outputFilename << endl;

    return 0;
}

