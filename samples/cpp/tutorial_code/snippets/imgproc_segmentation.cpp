#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/segmentation.hpp"

using namespace cv;

static
void usage_example_intelligent_scissors()
{
    Mat image(Size(1920, 1080), CV_8UC3, Scalar::all(128));

    //! [usage_example_intelligent_scissors]
    segmentation::IntelligentScissorsMB tool;
    tool.setEdgeFeatureCannyParameters(16, 100)  // using Canny() as edge feature extractor
        .setGradientMagnitudeMaxLimit(200);

    // calculate image features
    tool.applyImage(image);

    // calculate map for specified source point
    Point source_point(200, 100);
    tool.buildMap(source_point);

    // fast fetching of contours
    // for specified target point and the pre-calculated map (stored internally)
    Point target_point(400, 300);
    std::vector<Point> pts;
    tool.getContour(target_point, pts);
    //! [usage_example_intelligent_scissors]
}

int main()
{
    usage_example_intelligent_scissors();
    return 0;
}
