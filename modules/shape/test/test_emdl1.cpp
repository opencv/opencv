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
const float minRad=0.2f;
const float maxRad=2;
const int NSN=5;//10;//20; //number of shapes per class
const int NP=100; //number of points sympliying the contour
const float CURRENT_MAX_ACCUR=95; //98% and 99% reached in several tests, 95 is fixed as minimum boundary

class CV_ShapeEMDTest : public cvtest::BaseTest
{
public:
    CV_ShapeEMDTest();
    ~CV_ShapeEMDTest();
protected:
    void run(int);

private:
    void mpegTest();
    void listShapeNames(vector<string> &listHeaders);
    vector<Point2f> convertContourType(const Mat &, int n=0 );
    float computeShapeDistance(vector <Point2f>& queryNormal,
                               vector <Point2f>& queryFlipped1,
                               vector <Point2f>& queryFlipped2,
                               vector<Point2f>& testq);
    void displayMPEGResults();
};

CV_ShapeEMDTest::CV_ShapeEMDTest()
{
}
CV_ShapeEMDTest::~CV_ShapeEMDTest()
{
}

vector <Point2f> CV_ShapeEMDTest::convertContourType(const Mat& currentQuery, int n)
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
    int dum=0;
    for (int add=(int)contoursQuery.size()-1; add<n; add++)
    {
        contoursQuery.push_back(contoursQuery[dum++]); //adding dummy values
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

void CV_ShapeEMDTest::listShapeNames( vector<string> &listHeaders)
{
    listHeaders.push_back("apple"); //ok
    listHeaders.push_back("children"); // ok
    listHeaders.push_back("device7"); // ok
    listHeaders.push_back("Heart"); // ok
    listHeaders.push_back("teddy"); // ok
}
float CV_ShapeEMDTest::computeShapeDistance(vector <Point2f>& query1, vector <Point2f>& query2,
                                         vector <Point2f>& query3, vector <Point2f>& testq)
{
    //waitKey(0);
    Ptr <ShapeContextDistanceExtractor> mysc = createShapeContextDistanceExtractor(angularBins, radialBins, minRad, maxRad);
    //Ptr <HistogramCostExtractor> cost = createNormHistogramCostExtractor(cv::DIST_L1);
    //Ptr <HistogramCostExtractor> cost = createChiHistogramCostExtractor(30,0.15);
    //Ptr <HistogramCostExtractor> cost = createEMDHistogramCostExtractor();
    // Ptr <HistogramCostExtractor> cost = createEMDL1HistogramCostExtractor();
    mysc->setIterations(1); //(3)
    mysc->setCostExtractor( createEMDL1HistogramCostExtractor() );
    //mysc->setTransformAlgorithm(createAffineTransformer(true));
    mysc->setTransformAlgorithm( createThinPlateSplineShapeTransformer() );
    //mysc->setImageAppearanceWeight(1.6);
    //mysc->setImageAppearanceWeight(0.0);
    //mysc->setImages(im1,imtest);
    return ( std::min( mysc->computeDistance(query1, testq),
                       std::min(mysc->computeDistance(query2, testq), mysc->computeDistance(query3, testq) )));
}

void CV_ShapeEMDTest::mpegTest()
{
    string baseTestFolder="shape/mpeg_test/";
    string path = cvtest::TS::ptr()->get_data_path() + baseTestFolder;
    vector<string> namesHeaders;
    listShapeNames(namesHeaders);

    // distance matrix //
    Mat distanceMat=Mat::zeros(NSN*(int)namesHeaders.size(), NSN*(int)namesHeaders.size(), CV_32F);

    // query contours (normal v flipped, h flipped) and testing contour //
    vector<Point2f> contoursQuery1, contoursQuery2, contoursQuery3, contoursTesting;

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
                    // skip self-comparisson //
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
                    contoursTesting=convertContourType(currentTest, NP);

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
void CV_ShapeEMDTest::displayMPEGResults()
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

void CV_ShapeEMDTest::run( int /*start_from*/ )
{
    mpegTest();
    displayMPEGResults();
}

TEST(ShapeEMD_SCD, regression) { CV_ShapeEMDTest test; test.safe_run(); }
