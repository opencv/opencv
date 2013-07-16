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
private:
    void test1();
    void test2();
    void test3();
    void test4();
    void mpegTest();
    void listShapeNames(vector<string> &listHeaders);
    vector<Point2f> convertContourType( Mat& );
};

CV_ShapeTest::CV_ShapeTest()
{
}
CV_ShapeTest::~CV_ShapeTest()
{
}

void CV_ShapeTest::test1()
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
    vector<vector<Point> > _contours1, _contours2;
    findContours(shape1, _contours1, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    findContours(shape2, _contours2, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    cout<<"1. Number of points in the contour before simplification: "<<_contours1[0].size()<<std::endl;
    cout<<"2. Number of points in the contour before simplification: "<<_contours2[0].size()<<std::endl;

    approxPolyDP(Mat(_contours1[0]), _contours1[0], 0.5, true);
    approxPolyDP(Mat(_contours2[0]), _contours2[0], 0.5, true);

    cout<<"1. Number of points in the contour after simplification: "<<_contours1[0].size()<<std::endl;
    cout<<"2. Number of points in the contour after simplification: "<<_contours2[0].size()<<std::endl;

    vector<Point2f> contours1, contours2;

    Mat(_contours1[0]).convertTo(contours1, Mat(contours1).type());
    Mat(_contours2[0]).convertTo(contours2, Mat(contours2).type());

    std::cout<<"test x: "<<contours1[0].x<<std::endl;
    std::cout<<"test y: "<<contours1[0].y<<std::endl;

    SCDMatcher sMatcher(0.01, DistanceSCDFlags::DEFAULT);

    while(1)
    {
        Mat scdesc1, scdesc2;
        int abins=9, rbins=5;
        SCD shapeDescriptor(abins,rbins,0.1,2);

        shapeDescriptor.extractSCD(contours1, scdesc1);
        shapeDescriptor.extractSCD(contours2, scdesc2);

        vector<DMatch> matches;
        sMatcher.matchDescriptors(scdesc1, scdesc2, matches);

        char key = (char)waitKey();
        if(key == 27 || key == 'q' || key == 'Q') // 'ESC'
            break;
    }

}

void CV_ShapeTest::test2()
{
    vector<Point2f> shape1, shape2;

    shape1.push_back(Point(100,100));
    shape1.push_back(Point(90,110));
    shape1.push_back(Point(90,140));
    shape1.push_back(Point(100,150));
    shape1.push_back(Point(110,120));
    shape1.push_back(Point(110,130));
    shape1.push_back(Point(120,120.5));
    shape1.push_back(Point(130,100));
    shape1.push_back(Point(140,120));
    shape1.push_back(Point(150,120.5));
    shape1.push_back(Point(140,130));
    shape1.push_back(Point(130,150));
    //shape1.push_back(Point(5,5)); //outlier
    //shape1.push_back(Point(10,240)); //outlier
    //shape1.push_back(Point(240,240)); //outlier

    shape2.push_back(Point(101.2,119.1));
    shape2.push_back(Point(141.5,161.1));
    shape2.push_back(Point(109.1,112.2));
    shape2.push_back(Point(101.3,152.5));
    shape2.push_back(Point(123.2,128.3));
    shape2.push_back(Point(117.4,142.3));
    shape2.push_back(Point(134.6,112.2));
    shape2.push_back(Point(144.2,126.3));
    shape2.push_back(Point(162.1,132.5));
    shape2.push_back(Point(126.5,131.5));
    shape2.push_back(Point(149.4,141.2));
    shape2.push_back(Point(111,161.1));
    //shape2.push_back(Point(15.6,6.3)); //outlier

    /* compute SCD */
    SCD shapeDescriptor(12,10,0.1,5);
    SCDMatcher scdmatcher(1, DistanceSCDFlags::DIST_CHI);
    Mat scdesc1, scdesc2;
    vector<DMatch> matches;

    shapeDescriptor.extractSCD(shape1, scdesc1);
    shapeDescriptor.extractSCD(shape2, scdesc2);

    scdmatcher.matchDescriptors(scdesc1, scdesc2, matches);

    /* draw */
    int max=(shape1.size()>shape2.size())?shape1.size():shape2.size();

    Mat im=Mat::zeros(250, 250, CV_8UC3);
    Mat im1=Mat::zeros(250, 250, CV_8UC3);

    cout<<"The max is: "<<max<<". The size1="<<shape1.size()<<", and the size2="<<shape2.size()<<endl;
    for (int i=0; i<max; i++)
    {
        if ((int)shape1.size()>i)
        {
            circle(im, shape1[i], 3, Scalar(255,255,0), 1);
            circle(im1, shape1[i], 3, Scalar(255,255,0), 1);
        }

        if ((int)shape2.size()>i)
        {
            circle(im, shape2[i], 3, Scalar(0,0,255), 1);
            circle(im1, shape2[i], 3, Scalar(0,0,255), 1);
        }

        if ((size_t)matches[i].queryIdx<shape1.size() && (size_t)matches[i].trainIdx<shape2.size())
        {
            line(im, shape1[matches[i].queryIdx], shape2[matches[i].trainIdx], Scalar(0,255,255));
            cout<<"The Point "<<matches[i].queryIdx<<" in shape1, match with Point "<<
                  matches[i].trainIdx<<" in shape 2"<<endl;
        }
    }

    ThinPlateSplineTransform tpsTra(0.1);
    //AffineTransform tpsTra(true);
    vector<Point2f> transformed_shape;
    /* transformin shape1 to look like shape2 (saving it into transformed_shape) */
    /* cyan transforms in yellow accodring to red */
    tpsTra.applyTransformation(shape1, shape2, matches, transformed_shape);
    Mat im2=Mat::zeros(250, 250, CV_8UC3);
    for (size_t i=0; i<transformed_shape.size(); i++)
    {
        circle(im1, transformed_shape[i], 3, Scalar(0,255,255), 1);
    }

    while(1)
    {
        /* cyan transforms in yellow accodring to red */
        imshow("shapes (cyan=shape1, red=shape2, yellow=warped shape1)", im1);
        imshow("matches", im);

        char key = (char)waitKey();
        if(key == 27 || key == 'q' || key == 'Q') // 'ESC'
            break;
    }
}

void CV_ShapeTest::test3()
{
    vector<Point2f> shape1, shape2;

    shape1.push_back(Point2f(100,100));
    shape1.push_back(Point2f(90,110));
    shape1.push_back(Point2f(90,140));
    shape1.push_back(Point2f(100,150));
    shape1.push_back(Point2f(110,120));
    shape1.push_back(Point2f(110,130));
    shape1.push_back(Point2f(120,120.5));
    shape1.push_back(Point2f(130,100));
    shape1.push_back(Point2f(140,120));
    shape1.push_back(Point2f(150,120.5));
    shape1.push_back(Point2f(140,130));
    shape1.push_back(Point2f(130,150));
    //shape1.push_back(Point2f(5,5)); //outlier
    //shape1.push_back(Point2f(10,240)); //outlier
    //shape1.push_back(Point2f(240,240)); //outlier

    shape2.push_back(Point2f(145,106));
    shape2.push_back(Point2f(145,156));
    shape2.push_back(Point2f(115,106));
    shape2.push_back(Point2f(105,116));
    shape2.push_back(Point2f(155,126));
    shape2.push_back(Point2f(165,126.5));
    shape2.push_back(Point2f(155,136));
    shape2.push_back(Point2f(125,126));
    shape2.push_back(Point2f(125,136));
    shape2.push_back(Point2f(135,126.5));
    shape2.push_back(Point2f(105,146));
    shape2.push_back(Point2f(115,156));
    //shape2.push_back(Point2f(15.6,6.3)); //outlier

    SCD shapeDescriptor(12,10,0.1,5,true);
    SCDMatcher scdmatcher(1, DistanceSCDFlags::DIST_CHI);
    Mat scdesc1, scdesc2;
    vector<DMatch> matches;
    ThinPlateSplineTransform tpsTra(0.0);

    vector<Point2f> old_shape1=shape1;
    /* iterative process */
    for (int i=0;i<3;i++)
    {
        /* compute SCD and Match */
        shapeDescriptor.extractSCD(shape1, scdesc1);
        shapeDescriptor.extractSCD(shape2, scdesc2);
        scdmatcher.matchDescriptors(scdesc1, scdesc2, matches);

        /* transformin shape1 to look like shape2 (saving it into transformed_shape) */
        /* cyan transforms in yellow accodring to red */
        vector<Point2f> transformed_shape;
        tpsTra.applyTransformation(shape1, shape2, matches, transformed_shape);
        shape1=transformed_shape;
    }

    /* draw */
    int max=(shape1.size()>shape2.size())?shape1.size():shape2.size();

    Mat im=Mat::zeros(250, 250, CV_8UC3);
    Mat im1=Mat::zeros(250, 250, CV_8UC3);

    for (int i=0; i<max; i++)
    {
        if ((int)shape1.size()>i)
        {
            circle(im, shape1[i], 3, Scalar(255,255,0), 1);
            //circle(im1, shape1[i], 3, Scalar(255,255,0), 1);
            circle(im1, old_shape1[i], 3, Scalar(255,0,0), 1);
        }

        if ((int)shape2.size()>i)
        {
            circle(im, shape2[i], 3, Scalar(0,0,255), 1);
            circle(im1, shape2[i], 3, Scalar(0,0,255), 1);
        }

        if ((size_t)matches[i].queryIdx<shape1.size() && (size_t)matches[i].trainIdx<shape2.size())
        {
            line(im, shape1[matches[i].queryIdx], shape2[matches[i].trainIdx], Scalar(0,255,255));
        }
    }

    while(1)
    {
        /* cyan transforms in yellow accodring to red */
        imshow("matches", im);
        imshow("shapes (cyan=shape1, red=shape2, yellow=warped shape1)", im1);

        char key = (char)waitKey();
        if(key == 27 || key == 'q' || key == 'Q') // 'ESC'
            break;
    }
}

void CV_ShapeTest::test4()
{
    vector<Point2f> shape1, shape2;

    shape1.push_back(Point2f(55,40));
    shape1.push_back(Point2f(55,50));
    shape1.push_back(Point2f(50,60));
    shape1.push_back(Point2f(50,70));
    shape1.push_back(Point2f(50,80));
    shape1.push_back(Point2f(50,90));
    shape1.push_back(Point2f(50,100));
    shape1.push_back(Point2f(50,110));
    shape1.push_back(Point2f(60,60));
    shape1.push_back(Point2f(60,70));
    shape1.push_back(Point2f(60,80));
    shape1.push_back(Point2f(60,90));
    shape1.push_back(Point2f(60,100));
    shape1.push_back(Point2f(60,110));

    shape2.push_back(Point2f(2*40,2*55)-Point2f(16,14));
    shape2.push_back(Point2f(2*50,2*55)-Point2f(14.5,15.5));
    shape2.push_back(Point2f(2*60,2*50)-Point2f(14,15));
    shape2.push_back(Point2f(2*70,2*50)-Point2f(15,16));
    shape2.push_back(Point2f(2*80,2*50)-Point2f(16,14.5));
    shape2.push_back(Point2f(2*90,2*50)-Point2f(15.5,15.5));
    shape2.push_back(Point2f(2*100,2*50)-Point2f(15,15));
    shape2.push_back(Point2f(2*110,2*50)-Point2f(15,14.5));
    shape2.push_back(Point2f(2*60,2*60)-Point2f(15,16));
    shape2.push_back(Point2f(2*70,2*60)-Point2f(16,15));
    shape2.push_back(Point2f(2*80,2*60)-Point2f(14,15));
    shape2.push_back(Point2f(2*90,2*60)-Point2f(15,15));
    shape2.push_back(Point2f(2*100,2*60)-Point2f(15.5,15));
    shape2.push_back(Point2f(2*110,2*60)-Point2f(14.5,14.5));
    //shape2.push_back(Point2f(2*110,2*60)-Point2f(16,17));
    //shape2.push_back(Point2f(2,2));

    SCD shapeDescriptor(12,5,0.2,4,true);
    SCDMatcher scdmatcher(2.0, DistanceSCDFlags::DIST_CHI);
    Mat scdesc1, scdesc2;
    vector<DMatch> matches;
    //AffineTransform tpsTra(false);
    //AffineTransform tpsTra(true);
    ThinPlateSplineTransform tpsTra;
    float betai=0; //Regularization params
    float annRate=1;

    vector<Point2f> old_shape1=shape1;
    /* iterative process */
    float scdistance=0.0, benergy=0.0;
    for (int i=0;i<3;i++)
    {
        /* compute SCD and Match */
        shapeDescriptor.extractSCD(shape1, scdesc1);
        /* regularization parameter with annealing rate = 1 */
        betai=pow(shapeDescriptor.getMeanDistance(),2)*pow(annRate, i+1);
        shapeDescriptor.extractSCD(shape2, scdesc2);
        scdmatcher.matchDescriptors(scdesc1, scdesc2, matches);

        /* transformin shape1 to look like shape2 (saving it into transformed_shape) */
        /* cyan transforms in yellow accodring to red */
        vector<Point2f> transformed_shape;
        tpsTra.setRegularizationParam(betai);
        tpsTra.applyTransformation(shape1, shape2, matches, transformed_shape);
        shape1=transformed_shape;

        scdistance += scdmatcher.getMatchingCost();
        benergy += tpsTra.getTranformCost();
    }
    std::cout<<"Shape Context Distance: "<<scdistance<<std::endl;
    std::cout<<"Bending Energy: "<<benergy<<std::endl;

    /* draw */
    int max=(shape1.size()>shape2.size())?shape1.size():shape2.size();

    Mat im=Mat::zeros(250, 250, CV_8UC3);
    Mat im1=Mat::zeros(250, 250, CV_8UC3);

    for (int i=0; i<max; i++)
    {
        if ((int)shape1.size()>i)
        {
            circle(im, shape1[i], 3, Scalar(255,255,0), 1);
            //circle(im1, shape1[i], 3, Scalar(255,255,0), 1);
            circle(im1, old_shape1[i], 3, Scalar(255,0,0), 1);
        }

        if ((int)shape2.size()>i)
        {
            circle(im, shape2[i], 3, Scalar(0,0,255), 1);
            circle(im1, shape2[i], 3, Scalar(0,0,255), 1);
        }

        if ((size_t)matches[i].queryIdx<shape1.size() && (size_t)matches[i].trainIdx<shape2.size())
        {
            line(im, shape1[matches[i].queryIdx], shape2[matches[i].trainIdx], Scalar(0,255,255));
            line(im1, old_shape1[matches[i].queryIdx], shape2[matches[i].trainIdx], Scalar(0,255,255));
        }
    }

    while(1)
    {
        /* cyan transforms in yellow according to red */
        imshow("shapes (cyan=shape1, red=shape2, yellow=warped shape1)", im1);
        imshow("matches", im);

        char key = (char)waitKey();
        if(key == 27 || key == 'q' || key == 'Q') // 'ESC'
            break;
    }
}

vector <Point2f> CV_ShapeTest::convertContourType(Mat& currentQuery)
{
    vector<vector<Point> > _contoursQuery;
    vector <Point2f> contoursQuery;
    findContours(currentQuery, _contoursQuery, RETR_LIST, CHAIN_APPROX_NONE);
    for (size_t border=0; border<_contoursQuery.size(); border++)
        for (size_t p=0; p<_contoursQuery[border].size(); p++)
            contoursQuery.push_back(Point2f((float)_contoursQuery[border][p].x,
                                            (float)_contoursQuery[border][p].y));
    return contoursQuery;
}

void CV_ShapeTest::mpegTest()
{
    string baseTestFolder="shape/mpeg_test/";
    string path = cvtest::TS::ptr()->get_data_path() + baseTestFolder;
    vector<string> namesHeaders;
    listShapeNames(namesHeaders);

    const int angularBins=12;
    const int radialBins=4;
    const float minRad=0.2;
    const float maxRad=2;

    SCD shapeDescriptor(angularBins,radialBins, minRad, maxRad,true);
    SCDMatcher scdmatcher(2.0, DistanceSCDFlags::DIST_CHI);
    vector<Point2f> contoursQuery, contoursTesting;

    //Mat(_contours2[0]).convertTo(contours2, Mat(contours2).type());

    Mat querySCDMatrix;
    Mat testingSCDMatrix;

    //vector<Point2f>

    /* reading query and computing its properties */
    for (size_t n=0; n<namesHeaders.size(); n++)
    {
        for (int i=1; i<=20; i++)
        {
            /* read current image */
            stringstream thepathandname;
            thepathandname<<path+namesHeaders[n]<<"-"<<i<<".png";
            Mat currentQuery;
            std::cout<<"the current image: "<<(thepathandname.str());
            std::cout<<std::endl;
            currentQuery=imread(thepathandname.str().c_str(), 0);

            /* compute border and descriptor of the query */
            contoursQuery=convertContourType(currentQuery);

            //Mat(_contoursQuery[0]).convertTo(contoursQuery, Mat(contoursQuery).type());
            /* draw */
            Mat queryImage=Mat::zeros(currentQuery.rows, currentQuery.cols, CV_8UC3);
            for (size_t p=0; p<contoursQuery.size(); p++)
                circle(queryImage, contoursQuery[p], 3, Scalar(0,255,0), 1);
            imshow("Query Contour Points", queryImage);
            std::cout<<"Size of the contour: "<<contoursQuery.size()<<std::endl;
            char key = (char)waitKey();
            if(key == 27 || key == 'n' || key == ' ')
                continue;
        }
    }
}

void CV_ShapeTest::listShapeNames( vector<string> &listHeaders)
{
    listHeaders.push_back("apple");
    listHeaders.push_back("bat");
    listHeaders.push_back("beetle");
    listHeaders.push_back("bird");
    listHeaders.push_back("Bone");
    listHeaders.push_back("bottle");
    listHeaders.push_back("brick");
    listHeaders.push_back("butterfly");
    listHeaders.push_back("camel");
    listHeaders.push_back("car");
    listHeaders.push_back("carriage");
    listHeaders.push_back("butterfly");
    listHeaders.push_back("cattle");
    listHeaders.push_back("cellular_phone");
    listHeaders.push_back("chicken");
    listHeaders.push_back("children");
    listHeaders.push_back("chopper");
    listHeaders.push_back("classic");
    listHeaders.push_back("Comma");
    listHeaders.push_back("crown");
    listHeaders.push_back("cup");
    listHeaders.push_back("deer");
    listHeaders.push_back("device0");
    listHeaders.push_back("device1");
    listHeaders.push_back("device2");
    listHeaders.push_back("device3");
    listHeaders.push_back("device4");
    listHeaders.push_back("device5");
    listHeaders.push_back("device6");
    listHeaders.push_back("device7");
    listHeaders.push_back("device8");
    listHeaders.push_back("device9");
    listHeaders.push_back("dog");
    listHeaders.push_back("elephant");
    listHeaders.push_back("face");
    listHeaders.push_back("fish");
    listHeaders.push_back("flatfish");
    listHeaders.push_back("fly");
    listHeaders.push_back("fork");
    listHeaders.push_back("fountain");
    listHeaders.push_back("frog");
    listHeaders.push_back("Glas");
    listHeaders.push_back("guitar");
    listHeaders.push_back("hammer");
    listHeaders.push_back("hat");
    listHeaders.push_back("HCircle");
    listHeaders.push_back("Heart");
    listHeaders.push_back("horse");
    listHeaders.push_back("horseshoe");
    listHeaders.push_back("jar");
    listHeaders.push_back("key");
    listHeaders.push_back("lizzard");
    listHeaders.push_back("lmfish");
    listHeaders.push_back("Misk");
    listHeaders.push_back("octopus");
    listHeaders.push_back("pencil");
    listHeaders.push_back("personal_car");
    listHeaders.push_back("pocket");
    listHeaders.push_back("rat");
    listHeaders.push_back("ray");
    listHeaders.push_back("sea_snake");
    listHeaders.push_back("shoe");
    listHeaders.push_back("spoon");
    listHeaders.push_back("spring");
    listHeaders.push_back("stef");
    listHeaders.push_back("teddy");
    listHeaders.push_back("tree");
    listHeaders.push_back("truck");
    listHeaders.push_back("turtle");
    listHeaders.push_back("watch");
}

void CV_ShapeTest::run( int /*start_from*/ )
{
    //test1();
    mpegTest();
    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Shape_SCD, regression) { CV_ShapeTest test; test.safe_run(); }
