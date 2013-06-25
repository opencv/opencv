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
    /*ellipse(img, Point(125, 125), Size(100,70), 0, 0, 360, 
             Scalar(255,255,255), -1, 8, 0);*/
    circle(img, Point(125, 125), 100, Scalar(255,255,255), -1, 8, 0);      
    imshow("image", img);
        
    //Extract the Contours
    vector<vector<Point> > contours;
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    cout<<"Number of points in the contour before simplification: "<<contours[0].size()<<std::endl;

    approxPolyDP(Mat(contours[0]), contours[0], 0.5, true);
    cout<<"Number of points in the contour after simplification: "<<contours[0].size()<<std::endl;
    Mat img2=img.clone();
    cvtColor(img2,img2,8);
    
    int refPointIdx=0;   

    SCDMatcher sMatcher;
    //Mat costMatrix;
	while(1)
	{
        Mat scdesc;
        int abins=9, rbins=5;
        SCD shapeDescriptor(abins,rbins,0.1,2);
        drawContours(img2, contours, -1, Scalar(255,255,0), 5);
        shapeDescriptor.extractSCD(contours[0], scdesc);
        
        Mat descim;
        
        Point refP = (contours[0])[refPointIdx];
        
        for (size_t i=0; i<contours[0].size(); i++)
        {
            if (i==(size_t)refPointIdx) continue;
            Point P = (contours[0])[i];
            circle(img2, P, 2, Scalar(64,64,64),1);
        }
        circle(img2, refP, 2, Scalar(0,0,255),2);
        imshow("contours", img2);
        drawSCD(scdesc, abins, rbins, descim, refPointIdx, 25, DrawSCDFlags::DRAW_NORM_NEG);
		imshow("feature SCD", descim);
        
        char key = (char)waitKey();
        if(key == 27 || key == 'q' || key == 'Q') // 'ESC'
            break;
        if(key == ' ')
            refPointIdx++;
        if ((size_t)refPointIdx>=contours[0].size())
            refPointIdx=0;
        //sMatcher.buildCostMatrix(scdesc,  scdesc, costMatrix, DistanceSCDFlags::DEFAULT);
    } 

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Shape_SCD, regression) { CV_ShapeTest test; test.safe_run(); }
