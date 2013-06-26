/*
 * Test (Temporal, just "Hello World"-like tests) 
 */
 

#include "test_precomp.hpp"

using namespace cv;
using namespace std;

class CV_ShapeTest : public cvtest::BaseTest
{
public:
    CV_ShapeTest();
    ~CV_ShapeTest();
protected:
    void run(int);
};

CV_ShapeTest::CV_ShapeTest()
{
}
CV_ShapeTest::~CV_ShapeTest()
{
}

void CV_ShapeTest::run( int /*start_from*/ )
{
    Mat shape1 = Mat::zeros(250, 250, CV_8UC1);
    Mat shape2 = Mat::zeros(250, 250, CV_8UC1);
    //Draw an Ellipse
    ellipse(shape1, Point(125, 125), Size(100,70), 0, 0, 360, 
             Scalar(255,255,255), -1, 8, 0);
    circle(shape2, Point(125, 125), 100, Scalar(255,255,255), -1, 8, 0);      
    imshow("image 1", shape1);
    imshow("image 2", shape2);
        
    //Extract the Contours
    vector<vector<Point> > contours1, contours2;
    findContours(shape1, contours1, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    findContours(shape2, contours2, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    cout<<"1. Number of points in the contour before simplification: "<<contours1[0].size()<<std::endl;
    cout<<"2. Number of points in the contour before simplification: "<<contours2[0].size()<<std::endl;

    approxPolyDP(Mat(contours1[0]), contours1[0], 0.5, true);
    approxPolyDP(Mat(contours2[0]), contours2[0], 0.5, true);
    
    cout<<"1. Number of points in the contour after simplification: "<<contours1[0].size()<<std::endl;
    cout<<"2. Number of points in the contour after simplification: "<<contours2[0].size()<<std::endl;  

    SCDMatcher sMatcher(0.01, DistanceSCDFlags::DEFAULT);
	while(1)
	{
        Mat scdesc1, scdesc2;
        int abins=9, rbins=5;
        SCD shapeDescriptor(abins,rbins,0.1,2);
        
        shapeDescriptor.extractSCD(contours1[0], scdesc1);
        shapeDescriptor.extractSCD(contours2[0], scdesc2);

        vector<DMatch> matches;
        sMatcher.matchDescriptors(scdesc1, scdesc2, matches);
        
        char key = (char)waitKey();
        if(key == 27 || key == 'q' || key == 'Q') // 'ESC'
            break;
    }

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Shape_SCD, regression) { CV_ShapeTest test; test.safe_run(); }
