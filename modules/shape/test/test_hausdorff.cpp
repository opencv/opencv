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
#include <stdlib.h>

using namespace cv;
using namespace std;

const int NSN=5;//10;//20; //number of shapes per class
const float CURRENT_MAX_ACCUR=85; //90% and 91% reached in several tests, 85 is fixed as minimum boundary

class CV_HaussTest : public cvtest::BaseTest
{
public:
    CV_HaussTest();
    ~CV_HaussTest();
protected:
    void run(int);
private:
    float computeShapeDistance(vector<Point> &query1, vector<Point> &query2,
                               vector<Point> &query3, vector<Point> &testq);
    vector <Point> convertContourType(const Mat& currentQuery, int n=180);
    vector<Point2f> normalizeContour(const vector <Point>& contour);
    void listShapeNames( vector<string> &listHeaders);
    void mpegTest();
    void displayMPEGResults();
};

CV_HaussTest::CV_HaussTest()
{
}
CV_HaussTest::~CV_HaussTest()
{
}

vector<Point2f> CV_HaussTest::normalizeContour(const vector<Point> &contour)
{
    vector<Point2f> output(contour.size());
    Mat disMat((int)contour.size(),(int)contour.size(),CV_32F);
    Point2f meanpt(0,0);
    float meanVal=1;

    for (int ii=0, end1 = (int)contour.size(); ii<end1; ii++)
    {
        for (int jj=0, end2 = (int)contour.size(); end2; jj++)
        {
            if (ii==jj) disMat.at<float>(ii,jj)=0;
            else
            {
                disMat.at<float>(ii,jj)=
                    float(fabs(double(contour[ii].x*contour[jj].x)))+float(fabs(double(contour[ii].y*contour[jj].y)));
            }
        }
        meanpt.x+=contour[ii].x;
        meanpt.y+=contour[ii].y;
    }
    meanpt.x/=contour.size();
    meanpt.y/=contour.size();
    meanVal=float(cv::mean(disMat)[0]);
    for (size_t ii=0; ii<contour.size(); ii++)
    {
        output[ii].x = (contour[ii].x-meanpt.x)/meanVal;
        output[ii].y = (contour[ii].y-meanpt.y)/meanVal;
    }
    return output;
}

void CV_HaussTest::listShapeNames( vector<string> &listHeaders)
{
    listHeaders.push_back("apple"); //ok
    listHeaders.push_back("children"); // ok
    listHeaders.push_back("device7"); // ok
    listHeaders.push_back("Heart"); // ok
    listHeaders.push_back("teddy"); // ok
}


vector <Point> CV_HaussTest::convertContourType(const Mat& currentQuery, int n)
{
    vector<vector<Point> > _contoursQuery;
    vector <Point> contoursQuery;
    findContours(currentQuery, _contoursQuery, RETR_LIST, CHAIN_APPROX_NONE);
    for (size_t border=0; border<_contoursQuery.size(); border++)
    {
        for (size_t p=0; p<_contoursQuery[border].size(); p++)
        {
            contoursQuery.push_back(_contoursQuery[border][p]);
        }
    }

    // In case actual number of points is less than n
    for (int add=(int)contoursQuery.size()-1; add<n; add++)
    {
        contoursQuery.push_back(contoursQuery[contoursQuery.size()-add+1]); //adding dummy values
    }

    // Uniformly sampling
    random_shuffle(contoursQuery.begin(), contoursQuery.end());
    int nStart=n;
    vector<Point> cont;
    for (int i=0; i<nStart; i++)
    {
        cont.push_back(contoursQuery[i]);
    }
    return cont;
}

float CV_HaussTest::computeShapeDistance(vector <Point>& query1, vector <Point>& query2,
                                         vector <Point>& query3, vector <Point>& testq)
{
    Ptr <HausdorffDistanceExtractor> haus = createHausdorffDistanceExtractor();
    return std::min(haus->computeDistance(query1,testq), std::min(haus->computeDistance(query2,testq),
                             haus->computeDistance(query3,testq)));
}

void CV_HaussTest::mpegTest()
{
    string baseTestFolder="shape/mpeg_test/";
    string path = cvtest::TS::ptr()->get_data_path() + baseTestFolder;
    vector<string> namesHeaders;
    listShapeNames(namesHeaders);

    // distance matrix //
    Mat distanceMat=Mat::zeros(NSN*(int)namesHeaders.size(), NSN*(int)namesHeaders.size(), CV_32F);

    // query contours (normal v flipped, h flipped) and testing contour //
    vector<Point> contoursQuery1, contoursQuery2, contoursQuery3, contoursTesting;

    // reading query and computing its properties //
    int counter=0;
    const int loops=NSN*(int)namesHeaders.size()*NSN*(int)namesHeaders.size();
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
            vector<Point> origContour;
            contoursQuery1=convertContourType(currentQuery);
            origContour=contoursQuery1;
            contoursQuery2=convertContourType(flippedHQuery);
            contoursQuery3=convertContourType(flippedVQuery);

            // compare with all the rest of the images: testing //
            for (size_t nt=0; nt<namesHeaders.size(); nt++)
            {
                for (int it=1; it<=NSN; it++)
                {
                    /* skip self-comparisson */
                    counter++;
                    if (nt==n && it==i)
                    {
                        distanceMat.at<float>(NSN*(int)n+i-1,
                                              NSN*(int)nt+it-1)=0;
                        continue;
                    }
                    // read testing image //
                    stringstream thetestpathandname;
                    thetestpathandname<<path+namesHeaders[nt]<<"-"<<it<<".png";
                    Mat currentTest;
                    currentTest=imread(thetestpathandname.str().c_str(), 0);

                    // compute border of the testing //
                    contoursTesting=convertContourType(currentTest);

                    // compute shape distance //
                    std::cout<<std::endl<<"Progress: "<<counter<<"/"<<loops<<": "<<100*double(counter)/loops<<"% *******"<<std::endl;
                    std::cout<<"Computing shape distance between "<<namesHeaders[n]<<i<<
                               " and "<<namesHeaders[nt]<<it<<": ";
                    distanceMat.at<float>(NSN*(int)n+i-1, NSN*(int)nt+it-1)=
                            computeShapeDistance(contoursQuery1, contoursQuery2, contoursQuery3, contoursTesting);
                    std::cout<<distanceMat.at<float>(NSN*(int)n+i-1, NSN*(int)nt+it-1)<<std::endl;
                }
            }
        }
    }
    // save distance matrix //
    FileStorage fs(cvtest::TS::ptr()->get_data_path() + baseTestFolder + "distanceMatrixMPEGTest.yml", FileStorage::WRITE);
    fs << "distanceMat" << distanceMat;
}

const int FIRST_MANY=2*NSN;
void CV_HaussTest::displayMPEGResults()
{
    string baseTestFolder="shape/mpeg_test/";
    Mat distanceMat;
    FileStorage fs(cvtest::TS::ptr()->get_data_path() + baseTestFolder + "distanceMatrixMPEGTest.yml", FileStorage::READ);
    vector<string> namesHeaders;
    listShapeNames(namesHeaders);

    // Read generated MAT //
    fs["distanceMat"]>>distanceMat;

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
    float porc = 100*float(corrects)/(NSN*distanceMat.rows);
    std::cout<<"%="<<porc<<std::endl;
    if (porc >= CURRENT_MAX_ACCUR)
        ts->set_failed_test_info(cvtest::TS::OK);
    else
        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);

}


void CV_HaussTest::run(int /* */)
{
    mpegTest();
    displayMPEGResults();
    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Hauss, regression) { CV_HaussTest test; test.safe_run(); }
