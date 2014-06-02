#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video.hpp"
#include <fstream>

using namespace std;
using namespace cv;

static void farneback( InputArray i1, InputArray i2, InputOutputArray flow )
{
    calcOpticalFlowFarneback(i1, i2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
}
static void simpleflow( InputArray i1, InputArray i2, InputOutputArray flow )
{
    calcOpticalFlowSF(i1, i2, flow, 3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);
}
static void tvl1( InputArray i1, InputArray i2, InputOutputArray flow )
{
    Ptr<DenseOpticalFlow> tvl1 = createOptFlow_DualTVL1();
    tvl1->calc(i1, i2, flow);
}

// binary file format for flow data specified here:
// http://vision.middlebury.edu/flow/data/
// code duplicate from /modules/video/test/test_tvl1optflow.cpp
// TODO: Middlebury methods in one place
static void readOpticalFlowFromFile( Mat_<Point2f>& flow, const string& fileName )
{
    const float FLO_TAG_FLOAT = 202021.25f;
    ifstream file(fileName.c_str(), ios_base::binary);

    float tag;
    file.read((char*) &tag, sizeof(float));
    CV_Assert(tag == FLO_TAG_FLOAT);

    Size size;

    file.read((char*) &size.width, sizeof(int));
    file.read((char*) &size.height, sizeof(int));

    flow.create(size);

    for ( int i = 0; i < flow.rows; ++i )
    {
        for ( int j = 0; j < flow.cols; ++j )
        {
            Point2f u;

            file.read((char*) &u.x, sizeof(float));
            file.read((char*) &u.y, sizeof(float));

            flow(i, j) = u;
        }
    }
}
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
                "the method [farneback, simpleflow, tvl1], and optionally "
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

    if ( method == "farneback" || method == "tvl1" )
    {
        cvtColor(i1, i1, COLOR_BGR2GRAY);
        cvtColor(i2, i2, COLOR_BGR2GRAY);
    }

    flow = Mat(i1.size[0], i1.size[1], CV_32FC2);

    double startTick, time;
    startTick = (double) getTickCount();

    if ( method == "farneback" )
        farneback(i1, i2, flow);
    else if ( method == "simpleflow" )
        simpleflow(i1, i2, flow);
    else if ( method == "tvl1" )
        tvl1(i1, i2, flow);
    else
        printf("Wrong method!\n");
    time = ((double) getTickCount() - startTick) / getTickFrequency();
    printf("\nTime: %f.3\n", time);

    if ( argc == 5 )
    { // compare to ground truth
        string flow_file(argv[4]);
        readOpticalFlowFromFile(ground_truth, flow_file);
        double endpointError = averageEndpointError(flow, ground_truth);
        printf("Average endpoint error: %.2f\n", endpointError);
        double angularError = averageAngularError(flow, ground_truth);
        printf("Average angular error: %.2f\n", angularError);
    }
    return 0;
}
