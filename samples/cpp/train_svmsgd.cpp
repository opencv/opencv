#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace cv::ml;

#define WIDTH 841
#define HEIGHT 594

struct Data
{
    Mat img;
    Mat samples;
    Mat responses;
    RNG rng;
    //Point points[2];

    Data()
    {
        img = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
        imshow("Train svmsgd", img);
    }
};

bool doTrain(const Mat samples,const Mat responses, Mat &weights, float &shift);
bool findPointsForLine(const Mat &weights, float shift, Point (&points)[2]);
bool findCrossPoint(const Mat &weights, float shift, const std::pair<Point,Point> &segment, Point &crossPoint);
void fillSegments(std::vector<std::pair<Point,Point> > &segments);
void redraw(Data data, const Point points[2]);
void addPointsRetrainAndRedraw(Data &data, int x, int y);


bool doTrain( const Mat samples, const Mat responses, Mat &weights, float &shift)
{
    cv::Ptr<SVMSGD> svmsgd = SVMSGD::create();
    svmsgd->setOptimalParameters(SVMSGD::ASGD);
    svmsgd->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 50000, 0.0000001));
    svmsgd->setLambda(0.01);
    svmsgd->setGamma0(1);
   // svmsgd->setC(5);

    cv::Ptr<TrainData> train_data = TrainData::create( samples, cv::ml::ROW_SAMPLE, responses );
    svmsgd->train( train_data );

    if (svmsgd->isTrained())
    {
        weights = svmsgd->getWeights();
        shift = svmsgd->getShift();

        std::cout << weights << std::endl;
        std::cout << shift << std::endl;

        return true;
    }
    return false;
}


bool findCrossPoint(const Mat &weights, float shift, const std::pair<Point,Point> &segment, Point &crossPoint)
{
    int x = 0;
    int y = 0;
    //с (0,0) всё плохо
    if (segment.first.x == segment.second.x && weights.at<float>(1) != 0)
    {
        x = segment.first.x;
        y = -(weights.at<float>(0) * x + shift) / weights.at<float>(1);
        if (y >= 0 && y <= HEIGHT)
        {
            crossPoint.x = x;
            crossPoint.y = y;
            return true;
        }
    }
    else if (segment.first.y == segment.second.y && weights.at<float>(0) != 0)
    {
        y = segment.first.y;
        x = - (weights.at<float>(1) * y + shift) / weights.at<float>(0);
        if (x >= 0 && x <= WIDTH)
        {
            crossPoint.x = x;
            crossPoint.y = y;
            return true;
        }
    }
    return false;
}

bool findPointsForLine(const Mat &weights, float shift, Point (&points)[2])
{
    if (weights.empty())
    {
        return false;
    }

    int foundPointsCount = 0;
    std::vector<std::pair<Point,Point> > segments;
    fillSegments(segments);

    for (int i = 0; i < 4; i++)
    {
        if (findCrossPoint(weights, shift, segments[i], points[foundPointsCount]))
            foundPointsCount++;
        if (foundPointsCount > 2)
            break;
    }
    return true;
}

void fillSegments(std::vector<std::pair<Point,Point> > &segments)
{
    std::pair<Point,Point> curSegment;

    curSegment.first = Point(0,0);
    curSegment.second = Point(0,HEIGHT);
    segments.push_back(curSegment);

    curSegment.first = Point(0,0);
    curSegment.second = Point(WIDTH,0);
    segments.push_back(curSegment);

    curSegment.first = Point(WIDTH,0);
    curSegment.second = Point(WIDTH,HEIGHT);
    segments.push_back(curSegment);

    curSegment.first = Point(0,HEIGHT);
    curSegment.second = Point(WIDTH,HEIGHT);
    segments.push_back(curSegment);
}

void redraw(Data data, const Point points[2])
{
    data.img = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
    Point center;
    int radius = 3;
    Scalar color;
    for (int i = 0; i < data.samples.rows; i++)
    {
        center.x = data.samples.at<float>(i,0);
        center.y = data.samples.at<float>(i,1);
        color = (data.responses.at<float>(i) > 0) ? Scalar(128,128,0) : Scalar(0,128,128);
        circle(data.img, center, radius, color, 5);
    }
    line(data.img, points[0],points[1],cv::Scalar(1,255,1));

    imshow("Train svmsgd", data.img);
}

void addPointsRetrainAndRedraw(Data &data, int x, int y)
{

    Mat currentSample(1, 2, CV_32F);
    //start
/*
    Mat _weights;
    _weights.create(1, 2, CV_32FC1);
    _weights.at<float>(0) = 1;
    _weights.at<float>(1) = -1;

    int _x, _y;

    for (int i=0;i<199;i++)
    {
    _x = data.rng.uniform(0,800);
    _y = data.rng.uniform(0,500);*/
    currentSample.at<float>(0,0) = x;
    currentSample.at<float>(0,1) = y;
    //if (currentSample.dot(_weights) > 0)
        //data.responses.push_back(1);
   // else data.responses.push_back(-1);

    //finish
    data.samples.push_back(currentSample);



    Mat weights(1, 2, CV_32F);
    float shift = 0;

    if (doTrain(data.samples, data.responses, weights, shift))
    {
        Point points[2];
        shift = 0;

        findPointsForLine(weights, shift, points);

        redraw(data, points);
    }
}


static void onMouse( int event, int x, int y, int, void* pData)
{
    Data &data = *(Data*)pData;

    switch( event )
    {
    case CV_EVENT_LBUTTONUP:
        data.responses.push_back(1);
        addPointsRetrainAndRedraw(data, x, y);

        break;

    case CV_EVENT_RBUTTONDOWN:
        data.responses.push_back(-1);
        addPointsRetrainAndRedraw(data, x, y);
        break;
    }

}

int main()
{

    Data data;

    setMouseCallback( "Train svmsgd", onMouse, &data );
    waitKey();




    return 0;
}
