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
    Mat img = Mat::zeros(250, 250, CV_8UC1);
    
    //Draw an Ellipse
    ellipse( img, Point(125, 125), Size(100,70), 0, 0, 360, 
             cv::Scalar(255,255,255), -1, 8, 0 );
    
    //Show the Ellipse
    namedWindow( "image", 1 );
    imshow( "image", img );
    
    //Extract the Contours
    vector<vector<Point> > contours;
    findContours( img, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    cout<<"Number of points in the contour before simplification: "<<contours[0].size()<<std::endl;

    approxPolyDP(Mat(contours[0]), contours[0], 0.5, true);
    cout<<"Number of points in the contour after simplification: "<<contours[0].size()<<std::endl;
    namedWindow( "contours", 1 );
	drawContours(img, contours, -1/*Draw all contours*/, Scalar(128,128,128),5);
    
    cv::Mat scdesc;
    SCD shapeDescriptor(6,3,0.1,0.2);
    shapeDescriptor.extractSCD(contours[0], scdesc);

    std::cout<<"Shape Descriptors: "<<std::endl;
    
    for (int i=0; i<contours[0].size(); i++){
        for (int j=0; j<shapeDescriptor.descriptorSize(); j++){
            std::cout<<scdesc.at<float>(i,j)<<"\t";
        }std::cout<<std::endl;
    }
    	
	while(1)
	{
		imshow( "contours", img );
        char key = (char)waitKey();
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;
    } 

    ts->set_failed_test_info( cvtest::TS::OK);
}

TEST(Shape_SCD, regression) { CV_ShapeTest test; test.safe_run(); }
