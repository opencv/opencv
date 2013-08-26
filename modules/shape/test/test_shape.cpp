/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

using namespace cv;
using namespace std;

const int angularBins=12;
const int radialBins=4;
const float minRad=0.2;
const float maxRad=2;
const int NSN=20; //number of shapes per class
const int NP=300; //number of points sympliying the contour
const float outlierWeight=0.1;
const int numOutliers=25;

class CV_ShapeTest : public cvtest::BaseTest
{
public:
    CV_ShapeTest();
    ~CV_ShapeTest();
protected:
    void run(int);
private:
    void tempTests();
    void testSCD();
    void mpegTest();
    void listShapeNames(vector<string> &listHeaders);
    vector<Point2f> convertContourType(const Mat &, int n=0 );
    float computeShapeDistance(vector <Point2f>& queryNormal,
                               vector <Point2f>& queryFlipped1,
                               vector <Point2f>& queryFlipped2,
                               vector<Point2f>& testq, vector<DMatch> &);
    float distance(Point2f p, Point2f q);
    void displayMPEGResults();
    vector<float> getLocalTangentAngles(Mat image, vector<Point2f> pts);
    Mat buildTangentAngleDissimilarity(vector<float> angles1,vector<float> angles2);
};

CV_ShapeTest::CV_ShapeTest()
{
}
CV_ShapeTest::~CV_ShapeTest()
{
}

void CV_ShapeTest::tempTests()
{
    vector<Point2f> cont1, cont2;
    cont1.push_back(Point2f(1,1));
    cont1.push_back(Point2f(1,2));
    cont1.push_back(Point2f(1,3));
    cont1.push_back(Point2f(1,4));
    cont1.push_back(Point2f(1,5));
    cont1.push_back(Point2f(5,1));
    cont1.push_back(Point2f(5,2));
    cont1.push_back(Point2f(5,3));
    cont1.push_back(Point2f(5,4));
    cont1.push_back(Point2f(5,5));

    cont2.push_back(Point2f(5,5));
    cont2.push_back(Point2f(1,2));
    cont2.push_back(Point2f(1.1,3));
    cont2.push_back(Point2f(1,4.1));
    cont2.push_back(Point2f(1,5));
    cont2.push_back(Point2f(5.1,1));
    cont2.push_back(Point2f(5,2.1));
    cont2.push_back(Point2f(5,3));
    cont2.push_back(Point2f(5.1,4));
    cont2.push_back(Point2f(1.1,1.1));

    Mat scdesc1, scdesc2;
    SCD shapeDescriptor1(5, 3, 0.2, 2, false);
    SCD shapeDescriptor2(5, 3, 0.2, 2, false);

    shapeDescriptor1.extractSCD(cont1, scdesc1);
    shapeDescriptor2.extractSCD(cont2, scdesc2);

    for (int i=0; i<scdesc1.rows; i++)
    {
        for (int j=0; j<scdesc1.cols; j++)
        {
            std::cout<<scdesc1.at<float>(i,j)<<"\t";
        }
        std::cout<<std::endl;
    }
    std::cout<<shapeDescriptor1.getMeanDistance()<<std::endl;
    std::cout<<std::endl;
    for (int i=0; i<scdesc2.rows; i++)
    {
        for (int j=0; j<scdesc2.cols; j++)
        {
            std::cout<<scdesc2.at<float>(i,j)<<"\t";
        }
        std::cout<<std::endl;
    }
    std::cout<<shapeDescriptor2.getMeanDistance()<<std::endl;
    std::cout<<std::endl;
    SCDMatcher matcher(0, 0, DistanceSCDFlags::DIST_CHI);
    vector<DMatch> matches;
    vector<int> inliers1, inliers2;
    matcher.matchDescriptors(scdesc1, scdesc2, matches, inliers1, inliers2);

    for (size_t i=0; i<matches.size(); i++)
    {
        std::cout<<"scdesc1["<<matches[i].queryIdx<<"]->scdesc2["<<matches[i].trainIdx<<"]"<<std::endl;
    }
    std::cout<<"Cost: "<<matcher.getMatchingCost()<<std::endl;
}

void CV_ShapeTest::testSCD()
{
    // vars //
    Mat shape1, shape2, shapeBuf1, shapeBuf2;
    string baseTestFolder="shape/mpeg_test/";
    string path = cvtest::TS::ptr()->get_data_path() + baseTestFolder;
    vector<string> namesHeaders;
    listShapeNames(namesHeaders);
    stringstream thepathandname1;
    stringstream thepathandname2;

    // read //
    thepathandname1<<path+namesHeaders[5]<<"-"<<7<<".png";
    thepathandname2<<path+namesHeaders[5]<<"-"<<8<<".png";
    shape1=imread(thepathandname1.str(), IMREAD_GRAYSCALE);
    shape2=imread(thepathandname2.str(), IMREAD_GRAYSCALE);
    shapeBuf1=shape1.clone();
    shapeBuf2=shape2.clone();

    // contours //
    vector<Point2f> origcontours, contours1, contours2;
    contours1=convertContourType(shape1, NP);
    contours2=convertContourType(shape2, NP);
    origcontours=contours1;

    SCDMatcher scdmatcher(outlierWeight, numOutliers, DistanceSCDFlags::DEFAULT);
    ThinPlateSplineTransform tpsTra;
    vector<int> inliers1;
    vector<int> inliers2;

    Mat scdesc1, scdesc2;
    SCD shapeDescriptor1(angularBins, radialBins, minRad, maxRad, false);
    SCD shapeDescriptor2(angularBins, radialBins, minRad, maxRad, false);
    vector<DMatch> matches;
    float beta, annRate=2;

    // start loop //
    for (int j=0; j<1; j++)
    {
        // compute SCD //
        shapeDescriptor1.extractSCD(contours1, scdesc1, inliers1);
        shapeDescriptor2.extractSCD(contours2, scdesc2, inliers2, shapeDescriptor1.getMeanDistance());

        // regularization parameter with annealing rate annRate //
        beta=pow(shapeDescriptor1.getMeanDistance(),2)*pow(annRate, j+1);

        // match //
        scdmatcher.matchDescriptors(scdesc1, scdesc2, matches, inliers1, inliers2);
        // apply TPS transform //
        tpsTra.setRegularizationParam(beta);
        tpsTra.applyTransformation(contours1, contours2, matches, contours1);
    }

    while(1)
    {
        Mat imageTrans;
        tpsTra.warpImage(shapeBuf2, imageTrans);

        // show //
        Mat proc=Mat::zeros(shape1.rows, shape1.cols, CV_8UC3);
        for (size_t p=0; p<contours1.size(); p++)
        {
            circle(proc, origcontours[p], 4, Scalar(255,0,0), 1); //blue: query
            circle(proc, contours1[p], 3, Scalar(0,255,0), 1); //green: modified query
            circle(proc, contours2[p], 2, Scalar(0,0,255), 1); //red: target
        }
        for (size_t l=0; l<matches.size(); l++)
        {
            if (matches[l].trainIdx<(int)contours2.size() && matches[l].queryIdx<(int)contours1.size())
            {
                line(proc, contours2[matches[l].trainIdx],
                     contours1[matches[l].queryIdx], Scalar(160,230,189));
            }
        }
        imshow("proc", proc);
        imshow("image 1", shapeBuf1);
        imshow("image 2", shapeBuf2);
        imshow("warped image",imageTrans);
        char key = (char)waitKey();
        if(key == 27 || key == 'q' || key == 'Q') // 'ESC'
            break;
    }

}

vector <Point2f> CV_ShapeTest::convertContourType(const Mat& currentQuery, int n)
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

    // In case actual number of points is less than n
    for (int add=contoursQuery.size()-1; add<n; add++)
    {
        contoursQuery.push_back(contoursQuery[contoursQuery.size()-add+1]); //adding dummy values
    }

    // Uniformly sampling
    random_shuffle(contoursQuery.begin(), contoursQuery.end());
    int nStart=n;
    vector<Point2f> cont;
    for (int i=0; i<nStart; i++)
    {
        cont.push_back(contoursQuery[i]);
    }
    return cont;
}

void CV_ShapeTest::listShapeNames( vector<string> &listHeaders)
{
    listHeaders.push_back("apple"); //ok
    listHeaders.push_back("bat");
    listHeaders.push_back("beetle");
    listHeaders.push_back("bell"); // ~ok
    listHeaders.push_back("bird");
    listHeaders.push_back("Bone"); // ok
    listHeaders.push_back("bottle"); // ok
    listHeaders.push_back("brick"); // ok
    listHeaders.push_back("butterfly");
    listHeaders.push_back("camel");
    listHeaders.push_back("car"); // ok
    listHeaders.push_back("carriage"); // ok
    listHeaders.push_back("cattle");
    listHeaders.push_back("cellular_phone");
    listHeaders.push_back("chicken");
    listHeaders.push_back("children"); // ok
    listHeaders.push_back("chopper"); // ok
    listHeaders.push_back("classic"); // ~
    listHeaders.push_back("Comma"); // ~ok
    listHeaders.push_back("crown");
    listHeaders.push_back("cup"); // ~ok
    listHeaders.push_back("deer");
    listHeaders.push_back("device0"); // ~ok
    listHeaders.push_back("device1");
    listHeaders.push_back("device2");
    listHeaders.push_back("device3");
    listHeaders.push_back("device4");
    listHeaders.push_back("device5"); // ~ok
    listHeaders.push_back("device6");
    listHeaders.push_back("device7"); // ok
    listHeaders.push_back("device8");
    listHeaders.push_back("device9");
    listHeaders.push_back("dog");
    listHeaders.push_back("elephant");
    listHeaders.push_back("face"); // ok
    listHeaders.push_back("fish"); // ok
    listHeaders.push_back("flatfish"); // ok
    listHeaders.push_back("fly"); //~
    listHeaders.push_back("fork"); // ~ok
    listHeaders.push_back("fountain"); //ok
    listHeaders.push_back("frog");
    listHeaders.push_back("Glas"); // ~ok
    listHeaders.push_back("guitar");
    listHeaders.push_back("hammer");
    listHeaders.push_back("hat");
    listHeaders.push_back("HCircle"); // ok
    listHeaders.push_back("Heart"); // ok
    listHeaders.push_back("horse");
    listHeaders.push_back("horseshoe"); // ~ok
    listHeaders.push_back("jar");
    listHeaders.push_back("key"); // ok
    listHeaders.push_back("lizzard");
    listHeaders.push_back("lmfish"); //~
    listHeaders.push_back("Misk"); // ~ok
    listHeaders.push_back("octopus");
    listHeaders.push_back("pencil"); // ~
    listHeaders.push_back("personal_car"); // ~ok
    listHeaders.push_back("pocket");
    listHeaders.push_back("rat"); // ok
    listHeaders.push_back("ray");
    listHeaders.push_back("sea_snake");
    listHeaders.push_back("shoe"); // ~ok
    listHeaders.push_back("spoon");
    listHeaders.push_back("spring");
    listHeaders.push_back("stef"); // ~ok
    listHeaders.push_back("teddy"); // ok
    listHeaders.push_back("tree"); //~ok
    listHeaders.push_back("truck"); // ok
    listHeaders.push_back("turtle");
    listHeaders.push_back("watch"); // ok
}

float CV_ShapeTest::computeShapeDistance(vector <Point2f>& query1, vector <Point2f>& query2,
                                         vector <Point2f>& query3, vector <Point2f>& testq,
                                         vector<DMatch>& matches)
{
    // queries //
    vector< vector<Point2f> > query;
    query.push_back(query1);
    query.push_back(query2);
    query.push_back(query3);

    // executers //
    SCD shapeDescriptorT(angularBins,radialBins, minRad, maxRad,false);
    vector<SCD> shapeDescriptors;
    shapeDescriptors.push_back(SCD(angularBins,radialBins, minRad, maxRad,false));
    shapeDescriptors.push_back(SCD(angularBins,radialBins, minRad, maxRad,false));
    shapeDescriptors.push_back(SCD(angularBins,radialBins, minRad, maxRad,false));
    vector<SCDMatcher> scdmatchers;
    scdmatchers.push_back(SCDMatcher(outlierWeight, numOutliers, DistanceSCDFlags::DIST_EMDL1));
    scdmatchers.push_back(SCDMatcher(outlierWeight, numOutliers, DistanceSCDFlags::DIST_EMDL1));
    scdmatchers.push_back(SCDMatcher(outlierWeight, numOutliers, DistanceSCDFlags::DIST_EMDL1));
    vector<ThinPlateSplineTransform> tpsTra(3);

    // SCD descriptors //
    vector<Mat> querySCD(3);
    Mat testingSCDMatrix;
    vector< vector<DMatch> > matchesvec(3);

    // Regularization params //
    float beta;
    float annRate=1;
    const float BETA=1;
    // Iterative process with NC cycles //
    int NC=2;//number of cycles
    vector<float> scdistances(3), benergies(3);
    benergies[0]=0;
    benergies[1]=0;
    benergies[2]=0;

    // outliers vectors
    vector< vector<int> > inliers1(3), inliers2(3);
    shapeDescriptorT.extractSCD(testq, testingSCDMatrix);

    // Tangent angle
    //Mat tanMat; //this is the additional cost mat
    //vector<float> testLTA = getLocalTangentAngles(imtest, test); // this is the tangent angle vector for test

    // start loop //
    for (int i=0; i<3; i++)
    {
        //tanMat=buildTangentAngleDissimilarity(getLocalTangentAngles(ims[i], query[i]), testLTA);
        //scdmatchers[i].setAdditionalCostTerm(tanMat);

        for (int j=0; j<NC; j++)
        {
            // compute SCD //
            if (j==0)
                shapeDescriptors[i].extractSCD(query[i], querySCD[i]);//
            else
                shapeDescriptors[i].extractSCD(query[i], querySCD[i], inliers1[i]);

            // Extract SCD descriptor of the testing shape //
            shapeDescriptorT.extractSCD(testq, testingSCDMatrix, inliers2[i], shapeDescriptors[i].getMeanDistance());

            // regularization parameter with annealing rate annRate //
            beta=BETA*pow(shapeDescriptors[i].getMeanDistance(),2)*pow(annRate, j);

            // match //
            scdmatchers[i].matchDescriptors(querySCD[i], testingSCDMatrix, matchesvec[i], inliers1[i], inliers2[i]);

            // apply TPS transform //
            tpsTra[i].setRegularizationParam(beta);
            tpsTra[i].applyTransformation(query[i], testq, matchesvec[i], query[i]);

            // updating distances values //
            benergies[i] += tpsTra[i].getTranformCost();
        }
        // updating distances values //;
        scdistances[i] = scdmatchers[i].getMatchingCost();
    }

    float benergiesfactor=1;
    float scfactor=1;//2-benergiesfactor;
    float distance1T=scfactor*scdistances[0]+benergiesfactor*benergies[0];//+dist1;
    float distance2T=scfactor*scdistances[1]+benergiesfactor*benergies[1];//+dist2;
    float distance3T=scfactor*scdistances[2]+benergiesfactor*benergies[2];//+dist3;

    if ( distance1T<=distance2T && distance1T<=distance3T )
    {
        matches=matchesvec[0];
        query1=query[0];
        return distance1T;
    }
    if ( distance2T<=distance1T && distance2T<=distance3T )
    {
        matches=matchesvec[1];
        query1=query[1];
        return distance2T;
    }
    if ( distance3T<=distance1T && distance3T<=distance2T )
    {
        matches=matchesvec[2];
        query1=query[2];
        return distance3T;
    }
    matches=matchesvec[1];
    return 0.0;
}

vector<float> CV_ShapeTest::getLocalTangentAngles(Mat image, vector<Point2f> pts)
{
    Mat G1, G2;
    Sobel(image, G1, image.depth(), 1,0);
    Sobel(image, G2, image.depth(), 0,1);

    vector<float> output(pts.size());
    for (size_t i=0; i<pts.size(); i++)
    {
        int x=floor(pts[i].x+0.5);
        int y=floor(pts[i].y+0.5);
        if (y<G1.rows && x<G1.cols && x>=0 && y>=0 )
        {
            output[i]=atan2(G2.at<float>(y,x), G1.at<float>(y,x))+CV_PI/2;
        }
        else
        {
            output[i]=2*CV_PI;
        }
    }
    return output;
}

Mat CV_ShapeTest::buildTangentAngleDissimilarity(vector<float> angles1, vector<float> angles2)
{
    Mat output = Mat::zeros(angles1.size(), angles2.size(), CV_32F);
    for (int i=0; i<output.rows; i++)
    {
        for (int j=0; j<output.cols; j++)
        {
            output.at<float>(i,j)=0.5*(1-cos(angles1[i]-angles2[j]));
        }
    }
    return output;
}

float CV_ShapeTest::distance(Point2f p, Point2f q)
{
    Point2f diff = p - q;
    return (diff.x*diff.x + diff.y*diff.y);
}

void CV_ShapeTest::mpegTest()
{
    string baseTestFolder="shape/mpeg_test/";
    string path = cvtest::TS::ptr()->get_data_path() + baseTestFolder;
    vector<string> namesHeaders;
    listShapeNames(namesHeaders);

    // distance matrix //
    Mat distanceMat=Mat::zeros(NSN*namesHeaders.size(), NSN*namesHeaders.size(), CV_32F);

    // query contours (normal v flipped, h flipped) and testing contour //
    vector<Point2f> contoursQuery1, contoursQuery2, contoursQuery3, contoursTesting;

    // reading query and computing its properties //
    int counter=0;
    const int loops=NSN*namesHeaders.size()*NSN*namesHeaders.size();
    for (size_t n=0; n<namesHeaders.size(); n++)
    {
        for (int i=1; i<=NSN; i++)
        {
            // read current image //
            stringstream thepathandname;
            thepathandname<<path+namesHeaders[n]<<"-"<<i<<".png";
            Mat currentQuery, flippedHQuery, flippedVQuery;
            currentQuery=imread(thepathandname.str(), IMREAD_GRAYSCALE);
            flip(currentQuery, flippedHQuery, 0);
            flip(currentQuery, flippedVQuery, 1);
            // compute border of the query and its flipped versions //
            vector<Point2f> origContour;
            contoursQuery1=convertContourType(currentQuery, NP);
            origContour=contoursQuery1;
            contoursQuery2=convertContourType(flippedHQuery, NP);
            contoursQuery3=convertContourType(flippedVQuery, NP);

            // compare with all the rest of the images: testing //
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
                    // read testing image //
                    stringstream thetestpathandname;
                    thetestpathandname<<path+namesHeaders[nt]<<"-"<<it<<".png";
                    Mat currentTest;
                    currentTest=imread(thetestpathandname.str().c_str(), 0);

                    // compute border of the testing //
                    contoursTesting=convertContourType(currentTest, NP);

                    // compute shape distance //
                    std::cout<<std::endl<<"Progress: "<<counter<<"/"<<loops<<": "<<100*double(counter)/loops<<"% *******"<<std::endl;
                    std::cout<<"Computing shape distance between "<<namesHeaders[n]<<i<<
                               " and "<<namesHeaders[nt]<<it<<": ";
                    vector<DMatch> matches; //for drawing purposes
                    distanceMat.at<float>(NSN*n+i-1, NSN*nt+it-1)=
                            computeShapeDistance(contoursQuery1, contoursQuery2, contoursQuery3, contoursTesting, matches);
                    std::cout<<distanceMat.at<float>(NSN*n+i-1, NSN*nt+it-1)<<std::endl;

                    // draw //
                    #if (0)
                    Mat queryImage=Mat::zeros(500, 500, CV_8UC3);
                    for (size_t p=0; p<contoursQuery1.size(); p++)
                    {
                        circle(queryImage, origContour[p], 4, Scalar(255,0,0), 1); //blue: query
                        circle(queryImage, contoursQuery1[p], 3, Scalar(0,255,0), 1); //green: modified query
                        circle(queryImage, contoursTesting[p], 2, Scalar(0,0,255), 1); //red: target
                    }
                    for (size_t l=0; l<matches.size(); l++)
                    {
                        if (matches[l].trainIdx<(int)contoursTesting.size() && matches[l].queryIdx<(int)contoursQuery1.size())
                        {
                            line(queryImage, contoursTesting[matches[l].trainIdx],
                                 contoursQuery1[matches[l].queryIdx], Scalar(160,230,189));
                        }
                    }
                    stringstream text;
                    text<<"Shape distance: "<<distanceMat.at<float>(NSN*n+i-1, NSN*nt+it-1);
                    putText(queryImage, text.str(), Point(10,queryImage.rows-10),1,0.75,Scalar(255,255,0),1);
                    imshow("Query Contour Points", queryImage);
                    char key=(char)waitKey();
                    if (key == ' ') continue;
                    #endif
                }
            }
        }
    }
    // save distance matrix //
    FileStorage fs(cvtest::TS::ptr()->get_data_path() + baseTestFolder + "distanceMatrixMPEGTest.yml", FileStorage::WRITE);
    fs << "distanceMat" << distanceMat;
}

const int FIRST_MANY=40*NSN;
void CV_ShapeTest::displayMPEGResults()
{
    string baseTestFolder="shape/mpeg_test/";
    Mat distanceMat;
    FileStorage fs(cvtest::TS::ptr()->get_data_path() + baseTestFolder + "distanceMatrixMPEGTest.yml", FileStorage::READ);
    vector<string> namesHeaders;
    listShapeNames(namesHeaders);

    /* Read generated MAT */
    fs["distanceMat"]>>distanceMat;
    Mat draw;
    normalize(distanceMat, draw, 0, 255, NORM_MINMAX);
    draw.convertTo(draw, CV_8U);
    //imwrite("distanceMat.jpg",draw);

    int corrects=0;
    int divi=0;
    for (int row=0; row<distanceMat.rows; row++)
    {
        if (row%NSN==0) //another group
        {
            divi+=NSN;
        }
        for (int col=divi-NSN; col<divi; col++)
        {
            int nsmall=0;
            for (int i=0; i<distanceMat.cols; i++)
            {
                if (distanceMat.at<float>(row,col)>distanceMat.at<float>(row,i))
                {
                    nsmall++;
                }
            }
            if (nsmall<=FIRST_MANY)
            {
                corrects++;
            }
        }
    }

    std::cout<<"% porcentage of succes: "<<100*float(corrects)/(NSN*distanceMat.rows)<<std::endl;
}

void CV_ShapeTest::run( int /*start_from*/ )
{
    //tempTests();
    //mpegTest();
    //displayMPEGResults();
    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Shape_SCD, regression) { CV_ShapeTest test; test.safe_run(); }
