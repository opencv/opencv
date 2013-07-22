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
    void testSCD();
    void mpegTest();
    void listShapeNames(vector<string> &listHeaders);
    vector<Point2f> convertContourType( Mat&, int n=0 );
    float computeShapeDistance(vector <Point2f>& queryNormal,
                               vector <Point2f>& queryFlipped1,
                               vector <Point2f>& queryFlipped2,
                               vector<Point2f>& test, vector<DMatch> &);
    float point2PointEuclideanDistance(vector <Point2f>& query, vector <Point2f>& test, vector<DMatch>& matches);
    float distance(Point2f p, Point2f q);
    void displayMPEGResults();
};

CV_ShapeTest::CV_ShapeTest()
{
}
CV_ShapeTest::~CV_ShapeTest()
{
}

void CV_ShapeTest::testSCD()
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

vector <Point2f> CV_ShapeTest::convertContourType(Mat& currentQuery, int n)
{
    vector<vector<Point> > _contoursQuery;
    vector <Point2f> contoursQuery;
    findContours(currentQuery, _contoursQuery, RETR_LIST, CHAIN_APPROX_NONE);
    for (size_t border=0; border<_contoursQuery.size(); border++)
    {
        for (size_t p=0; p<_contoursQuery[border].size(); p++)
        {
            contoursQuery.push_back(Point2f((float)_contoursQuery[border][p].x,
                                            (float)_contoursQuery[border][p].y));
        }
    }
    //std::cout<<"Size Before simplification: "<<contoursQuery.size()<<std::endl;
    for (int add=contoursQuery.size(); add<=n; add++)
    {
        contoursQuery.push_back(Point2f(0,0)); //adding dummy values
    }
    if (n<=1)
    {
        return contoursQuery;
    }
    else
    {
        random_shuffle(contoursQuery.begin(), contoursQuery.end());
        vector<Point2f> cont;
        for (int i=0; i<n; i++)
        {
            cont.push_back(contoursQuery[i]);
        }
        return cont;
    }
}

void CV_ShapeTest::listShapeNames( vector<string> &listHeaders)
{
    //listHeaders.push_back("apple");
    //listHeaders.push_back("bat");
    /*listHeaders.push_back("beetle");
    listHeaders.push_back("bell");
    listHeaders.push_back("bird");
    listHeaders.push_back("Bone");
    listHeaders.push_back("bottle");
    listHeaders.push_back("brick");
    listHeaders.push_back("butterfly");
    listHeaders.push_back("camel");
    listHeaders.push_back("car");
    listHeaders.push_back("carriage");
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
    listHeaders.push_back("device0");*/
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

const int angularBins=12;
const int radialBins=4;
const float minRad=0.2;
const float maxRad=5;
const int NSN=20; //number of shapes per code (car, butterfly, etc)
const int NP=300; //number of points sympliying the contour

float CV_ShapeTest::computeShapeDistance(vector <Point2f>& query1, vector <Point2f>& query2,
                                         vector <Point2f>& query3, vector <Point2f>& test, vector<DMatch>& matches)
{
    /* executers */
    SCD shapeDescriptor1(angularBins,radialBins, minRad, maxRad,false);
    SCD shapeDescriptor2(angularBins,radialBins, minRad, maxRad,false);
    SCD shapeDescriptor3(angularBins,radialBins, minRad, maxRad,false);
    SCD shapeDescriptorT(angularBins,radialBins, minRad, maxRad,false);
    SCDMatcher scdmatcher1(2.0, DistanceSCDFlags::DIST_CHI);
    SCDMatcher scdmatcher2(2.0, DistanceSCDFlags::DIST_CHI);
    SCDMatcher scdmatcher3(2.0, DistanceSCDFlags::DIST_CHI);
    ThinPlateSplineTransform tpsTra1;
    ThinPlateSplineTransform tpsTra2;
    ThinPlateSplineTransform tpsTra3;
    /* SCD descriptors */
    Mat query1SCDMatrix, query2SCDMatrix, query3SCDMatrix;
    Mat testingSCDMatrix;
    vector<DMatch> matches1, matches2, matches3;
    /* Regularization params */
    float beta1=0, beta2=0, beta3=0;
    float annRate=0.5;

    /* Iterative process with NC cycles */
    int NC=5;//number of cycles
    float scdistance1=0.0, benergy1=0.0;// dist1=0.0;
    float scdistance2=0.0, benergy2=0.0;// dist2=0.0;
    float scdistance3=0.0, benergy3=0.0;// dist3=0.0;
    for (int i=0; i<NC; i++)
    {
        //std::cout<<"CICLO: "<<i<<std::endl;
        //std::cout<<"computing SCD "<<std::endl;
        // compute SCD //
        shapeDescriptor1.extractSCD(query1, query1SCDMatrix);
        shapeDescriptor2.extractSCD(query2, query2SCDMatrix);
        shapeDescriptor3.extractSCD(query3, query3SCDMatrix);

        // regularization parameter with annealing rate annRate //
        beta1=pow(shapeDescriptor1.getMeanDistance(),2)*pow(annRate, i+1);
        beta2=pow(shapeDescriptor2.getMeanDistance(),2)*pow(annRate, i+1);
        beta3=pow(shapeDescriptor3.getMeanDistance(),2)*pow(annRate, i+1);

        // compute SCD of the objective shape and match //
        shapeDescriptorT.extractSCD(test, testingSCDMatrix);
        //std::cout<<"Matching... "<<std::endl;
        scdmatcher1.matchDescriptors(query1SCDMatrix, testingSCDMatrix, matches1);
        scdmatcher2.matchDescriptors(query2SCDMatrix, testingSCDMatrix, matches2);
        scdmatcher3.matchDescriptors(query3SCDMatrix, testingSCDMatrix, matches3);

        vector<Point2f> transformed_shape;
        // transformin queries to look like test (saving it into transformed_shape) //
        //std::cout<<"Applying TPS "<<std::endl;
        tpsTra1.setRegularizationParam(beta1);
        tpsTra2.setRegularizationParam(beta2);
        tpsTra3.setRegularizationParam(beta3);
        tpsTra1.applyTransformation(query1, test, matches1, transformed_shape);
        query1=transformed_shape;
        tpsTra2.applyTransformation(query2, test, matches2, transformed_shape);
        query2=transformed_shape;
        tpsTra3.applyTransformation(query3, test, matches3, transformed_shape);
        query3=transformed_shape;

        // updating distances values //
        scdistance1 = scdmatcher1.getMatchingCost();
        scdistance2 = scdmatcher2.getMatchingCost();
        scdistance3 = scdmatcher3.getMatchingCost();
        benergy1 = tpsTra1.getTranformCost();
        benergy2 = tpsTra2.getTranformCost();
        benergy3 = tpsTra3.getTranformCost();
        //dist1 = point2PointEuclideanDistance(query1, test, matches1);
        //dist2 = point2PointEuclideanDistance(query2, test, matches2);
        //dist3 = point2PointEuclideanDistance(query3, test, matches3);
    }
    float distance1T=scdistance1+benergy1;//+dist1;
    float distance2T=scdistance2+benergy2;//+dist2;
    float distance3T=scdistance3+benergy3;//+dist3;

    //distance1T/=NC;
    //distance2T/=NC;
    //distance3T/=NC;

    if ( distance1T<=distance2T && distance1T<=distance3T )
    {
        matches=matches1;
        return distance1T;
    }
    if ( distance2T<=distance1T && distance2T<=distance3T )
    {
        matches=matches2;
        query1=query2;
        return distance2T;
    }
    if ( distance3T<=distance1T && distance3T<=distance2T )
    {
        matches=matches3;
        query1=query3;
        return distance3T;
    }
    matches=matches1;
    return 0.0;
}

float CV_ShapeTest::point2PointEuclideanDistance(vector <Point2f>& _query, vector <Point2f>& _test, vector<DMatch>& _matches)
{
    /* Use only valid matchings */
    std::vector<DMatch> matches;
    for (size_t i=0; i<_matches.size(); i++)
    {
        if (_matches[i].queryIdx<(int)_query.size() &&
            _matches[i].trainIdx<(int)_test.size())
        {
            matches.push_back(_matches[i]);
        }
    }

    /* Organizing the correspondent points in vector style */
    float dist=0.0;
    for (size_t i=0; i<matches.size(); i++)
    {
        Point2f pt1=_query[matches[i].queryIdx];
        Point2f pt2=_test[matches[i].trainIdx];
        dist+=distance(pt1,pt2);
    }

    return dist/matches.size();
}

float CV_ShapeTest::distance(Point2f p, Point2f q)
{
    Point2f diff = p - q;
    return (diff.x*diff.x + diff.y*diff.y)/2;
}

void CV_ShapeTest::mpegTest()
{
    string baseTestFolder="shape/mpeg_test/";
    string path = cvtest::TS::ptr()->get_data_path() + baseTestFolder;
    vector<string> namesHeaders;
    listShapeNames(namesHeaders);

    /* distance matrix */
    Mat distanceMat=Mat::zeros(NSN*namesHeaders.size(), NSN*namesHeaders.size(), CV_32F);

    /* query contours (normal v flipped, h flipped) and testing contour */
    vector<Point2f> contoursQuery1, contoursQuery2, contoursQuery3, contoursTesting;

    /* reading query and computing its properties */
    int counter=0;
    const int loops=NSN*namesHeaders.size()*NSN*namesHeaders.size();
    for (size_t n=0; n<namesHeaders.size(); n++)
    {
        for (int i=1; i<=NSN; i++)
        {
            /* read current image */
            stringstream thepathandname;
            thepathandname<<path+namesHeaders[n]<<"-"<<i<<".png";
            Mat currentQuery, auxQuery;
            currentQuery=imread(thepathandname.str().c_str(), 0);

            /* compute border of the query and its flipped versions */
            vector<Point2f> origContour;
            contoursQuery1=convertContourType(currentQuery, NP);
            origContour=contoursQuery1;
            flip(currentQuery, auxQuery, 0);
            contoursQuery2=convertContourType(auxQuery, NP);
            flip(currentQuery, auxQuery, 1);
            contoursQuery3=convertContourType(auxQuery, NP);

            /* compare with all the rest of the images: testing */
            for (size_t nt=0; nt<namesHeaders.size(); nt++)
            {
                for (int it=1; it<=NSN; it++)
                {
                    /* skip self-comparisson */
                    counter++;
                    if (nt==n && it==i)
                    {
                        distanceMat.at<float>(NSN*n+i-1,
                                              NSN*nt+it-1)=0;
                        continue;
                    }
                    /* read testing image */
                    stringstream thetestpathandname;
                    thetestpathandname<<path+namesHeaders[nt]<<"-"<<it<<".png";
                    Mat currentTest;
                    currentTest=imread(thetestpathandname.str().c_str(), 0);

                    /* compute border of the testing */
                    contoursTesting=convertContourType(currentTest, NP);

                    /* compute shape distance */
                    std::cout<<std::endl<<"Progress: "<<counter<<"/"<<loops<<": "<<100*double(counter)/loops<<"% *******"<<std::endl;
                    std::cout<<"Computing shape distance between "<<thepathandname.str()<<
                               " and "<<thetestpathandname.str()<<std::endl;
                    vector<DMatch> matches; //for drawing purposes
                    distanceMat.at<float>(NSN*n+i-1, NSN*nt+it-1)=
                            computeShapeDistance(contoursQuery1, contoursQuery2, contoursQuery3, contoursTesting, matches);
                    std::cout<<"The distance is: "<<distanceMat.at<float>(NSN*n+i-1, NSN*nt+it-1)<<std::endl;

                    /* draw */
                    /*Mat queryImage=Mat::zeros(currentQuery.rows, currentQuery.cols, CV_8UC3);
                    for (size_t p=0; p<contoursQuery1.size(); p++)
                    {
                        circle(queryImage, origContour[p], 4, Scalar(255,0,0), 1); //blue: query
                        circle(queryImage, contoursQuery1[p], 3, Scalar(0,255,0), 1); //green: modified query
                        circle(queryImage, contoursTesting[p], 2, Scalar(0,0,255), 1); //red: target
                    }
                    for (size_t l=0; l<matches.size(); l++)
                    {
                        line(queryImage, contoursTesting[matches[l].trainIdx],
                             contoursQuery1[matches[l].queryIdx], Scalar(160,230,189));
                    }
                    imshow("Query Contour Points", queryImage);
                    std::cout<<"Size of the contour and matches: "<<contoursQuery1.size()<<", matches: "<<
                            matches.size()<<std::endl;
                    char key=(char)waitKey();
                    if (key == ' ') continue;*/
                }
            }
        }
    }

    /* save distance matrix */
    FileStorage fs(cvtest::TS::ptr()->get_data_path() + baseTestFolder + "distanceMatrixMPEGTest.yml", FileStorage::WRITE);
    fs << "distanceMat" << distanceMat;
}

const int FIRST_MANY=40;
void CV_ShapeTest::displayMPEGResults()
{
    string baseTestFolder="shape/mpeg_test/";
    Mat distanceMat, sortedMat, actualSorted;
    FileStorage fs(cvtest::TS::ptr()->get_data_path() + baseTestFolder + "distanceMatrixMPEGTest.yml", FileStorage::READ);
    vector<string> namesHeaders;
    listShapeNames(namesHeaders);

    /* Read generated MAT */
    fs["distanceMat"]>>distanceMat;
    // sortIdx //
    cv::sortIdx(distanceMat, sortedMat, SORT_EVERY_ROW+SORT_ASCENDING);
    cv::sort(distanceMat, actualSorted, SORT_EVERY_ROW+SORT_ASCENDING);
    int corrects=0;
    int divi=0;

    for (int row=0; row<sortedMat.rows; row++)
    {
        if (row%NSN==0) //another group
        {
            divi+=NSN;
        }
        for (int col=0; col<sortedMat.cols; col++)
        {
            if (sortedMat.at<int>(row,col)<FIRST_MANY)
            {
                //there is an index belonging to the correct group
                if (col<divi)
                {
                    corrects++;
                }
            }
        }
    }
    //std::cout<<"#number of correct matches in the first 40 groups: "<<corrects<<std::endl;
    std::cout<<"% porcentage of succes: "<<100*float(corrects)/(NSN*sortedMat.rows)<<std::endl;
}

void CV_ShapeTest::run( int /*start_from*/ )
{
    mpegTest();
    displayMPEGResults();
    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Shape_SCD, regression) { CV_ShapeTest test; test.safe_run(); }
