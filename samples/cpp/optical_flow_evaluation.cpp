#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video.hpp"
#include <fstream>

using namespace std;
using namespace cv;


inline bool isFlowCorrect( const Point2f u )
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && (fabs(u.x) < 1e9) && (fabs(u.y) < 1e9);
}
inline bool isFlowCorrect( const Point3f u )
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && !cvIsNaN(u.z) && (fabs(u.x) < 1e9) && (fabs(u.y) < 1e9)
            && (fabs(u.z) < 1e9);
}
///based on TV-L1 test, needs improvement
static double averageEndpointError( const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2 )
{
    CV_Assert(flow1.rows == flow2.rows);
    CV_Assert(flow1.cols == flow2.cols);
    CV_Assert(flow1.channels() == 2 && flow2.channels() == 2);
    double sum = 0.0;
    int counter = 0;

    for ( int i = 0; i < flow1.rows; ++i )
    {
        for ( int j = 0; j < flow1.cols; ++j )
        {
            const Point2f u1 = flow1(i, j);
            const Point2f u2 = flow2(i, j);

            if ( isFlowCorrect(u1) && isFlowCorrect(u2) )
            {
                const Point2f diff = u1 - u2;
                sum += sqrt(diff.ddot(diff)); //distance
                ++counter;
            }
        }
    }
    return sum / (1e-9 + counter);
}
static double averageAngularError( const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2 )
{
    CV_Assert(flow1.rows == flow2.rows);
    CV_Assert(flow1.cols == flow2.cols);
    CV_Assert(flow1.channels() == 2 && flow2.channels() == 2);
    double sum = 0.0;
    int counter = 0;

    for ( int i = 0; i < flow1.rows; ++i )
    {
        for ( int j = 0; j < flow1.cols; ++j )
        {
            const Point2f u1_2d = flow1(i, j);
            const Point2f u2_2d = flow2(i, j);
            const Point3f u1(u1_2d.x, u1_2d.y, 1);
            const Point3f u2(u2_2d.x, u2_2d.y, 1);

            if ( isFlowCorrect(u1) && isFlowCorrect(u2) )
            {
                sum += acos((u1.ddot(u2)) / (norm(u1) * norm(u2)));
                ++counter;
            }
        }
    }
    return sum / (1e-9 + counter);
}

int main( int argc, char** argv )
{
    if ( argc < 4 )
    {
        printf("Not enough input arguments. Please specify 2 input images, "
                "the method [farneback, simpleflow, tvl1, deepflow], and optionally "
                "ground-truth flow (middlebury format)");
        return -1;
    }
    Mat i1, i2;
    Mat_<Point2f> flow, ground_truth;
    std::string method;
    i1 = imread(argv[1], 1);
    i2 = imread(argv[2], 1);
    method = argv[3];

    if ( !i1.data || !i2.data )
    {
        printf("No image data \n");
        return -1;
    }
    if ( i1.size() != i2.size() || i1.channels() != i2.channels())
    {
        printf("Dimension mismatch between input images\n");
        return -1;
    }
    // 8-bit images expected by all algorithms
    if ( i1.depth() != CV_8U) i1.convertTo(i1, CV_8U);
    if ( i2.depth() != CV_8U) i2.convertTo(i2, CV_8U);

    if ( ( method == "farneback" || method == "tvl1" || method =="deepflow") && i1.channels() == 3 )
    {   // 1-channel images are expected
        cvtColor(i1, i1, COLOR_BGR2GRAY);
        cvtColor(i2, i2, COLOR_BGR2GRAY);
    }else if (method == "simpleflow" && i1.channels() == 1)
    {   // 3-channel images expected
        cvtColor(i1, i1, COLOR_GRAY2BGR);
        cvtColor(i2, i2, COLOR_GRAY2BGR);
    }

    flow = Mat(i1.size[0], i1.size[1], CV_32FC2);
    Ptr<DenseOpticalFlow> algorithm;

    if ( method == "farneback" )
        algorithm = createOptFlow_Farnebacks();
    else if ( method == "simpleflow" )
        algorithm = createOptFlow_SimpleFlow();
    else if ( method == "tvl1" )
        algorithm = createOptFlow_DualTVL1();
    else if ( method == "deepflow" )
        algorithm = createOptFlow_DeepFlow();
    else
        printf("Wrong method!\n");


    double startTick, time;
    startTick = (double) getTickCount(); // measure time
    algorithm->calc(i1, i2, flow);
    time = ((double) getTickCount() - startTick) / getTickFrequency();
    printf("\nTime: %f.3\n", time);

    if ( argc == 5 )
    { // compare to ground truth
        string groundtruth_path(argv[4]);
        ground_truth = readOpticalFlow(groundtruth_path);
        double endpointError = averageEndpointError(flow, ground_truth);
        printf("Average endpoint error: %.2f\n", endpointError);
        double angularError = averageAngularError(flow, ground_truth);
        printf("Average angular error: %.2f\n", angularError);
    }
    return 0;
}
