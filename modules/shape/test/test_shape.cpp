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

template <typename T, typename compute>
class ShapeBaseTest : public cvtest::BaseTest
{
public:
    typedef Point_<T> PointType;
    ShapeBaseTest(int _NSN, int _NP, float _CURRENT_MAX_ACCUR)
        : NSN(_NSN), NP(_NP), CURRENT_MAX_ACCUR(_CURRENT_MAX_ACCUR)
    {
        // generate file list
        vector<string> shapeNames;
        shapeNames.push_back("apple"); //ok
        shapeNames.push_back("children"); // ok
        shapeNames.push_back("device7"); // ok
        shapeNames.push_back("Heart"); // ok
        shapeNames.push_back("teddy"); // ok
        for (vector<string>::const_iterator i = shapeNames.begin(); i != shapeNames.end(); ++i)
        {
            for (int j = 0; j < NSN; ++j)
            {
                stringstream filename;
                filename << cvtest::TS::ptr()->get_data_path()
                         << "shape/mpeg_test/" << *i << "-" << j + 1 << ".png";
                filenames.push_back(filename.str());
            }
        }
        // distance matrix
        const int totalCount = (int)filenames.size();
        distanceMat = Mat::zeros(totalCount, totalCount, CV_32F);
    }

protected:
    void run(int)
    {
        mpegTest();
        displayMPEGResults();
    }

    vector<PointType> convertContourType(const Mat& currentQuery) const
    {
        vector<vector<Point> > _contoursQuery;
        findContours(currentQuery, _contoursQuery, RETR_LIST, CHAIN_APPROX_NONE);

        vector <PointType> contoursQuery;
        for (size_t border=0; border<_contoursQuery.size(); border++)
        {
            for (size_t p=0; p<_contoursQuery[border].size(); p++)
            {
                contoursQuery.push_back(PointType((T)_contoursQuery[border][p].x,
                                                  (T)_contoursQuery[border][p].y));
            }
        }

        // In case actual number of points is less than n
        for (int add=(int)contoursQuery.size()-1; add<NP; add++)
        {
            contoursQuery.push_back(contoursQuery[contoursQuery.size()-add+1]); //adding dummy values
        }

        // Uniformly sampling
        random_shuffle(contoursQuery.begin(), contoursQuery.end());
        int nStart=NP;
        vector<PointType> cont;
        for (int i=0; i<nStart; i++)
        {
            cont.push_back(contoursQuery[i]);
        }
        return cont;
    }

    void mpegTest()
    {
        // query contours (normal v flipped, h flipped) and testing contour
        vector<PointType> contoursQuery1, contoursQuery2, contoursQuery3, contoursTesting;
        // reading query and computing its properties
        for (vector<string>::const_iterator a = filenames.begin(); a != filenames.end(); ++a)
        {
            // read current image
            int aIndex = (int)(a - filenames.begin());
            Mat currentQuery = imread(*a, IMREAD_GRAYSCALE);
            Mat flippedHQuery, flippedVQuery;
            flip(currentQuery, flippedHQuery, 0);
            flip(currentQuery, flippedVQuery, 1);
            // compute border of the query and its flipped versions
            contoursQuery1=convertContourType(currentQuery);
            contoursQuery2=convertContourType(flippedHQuery);
            contoursQuery3=convertContourType(flippedVQuery);
            // compare with all the rest of the images: testing
            for (vector<string>::const_iterator b = filenames.begin(); b != filenames.end(); ++b)
            {
                int bIndex = (int)(b - filenames.begin());
                float distance = 0;
                // skip self-comparisson
                if (a != b)
                {
                    // read testing image
                    Mat currentTest = imread(*b, IMREAD_GRAYSCALE);
                    // compute border of the testing
                    contoursTesting=convertContourType(currentTest);
                    // compute shape distance
                    distance = cmp(contoursQuery1, contoursQuery2,
                                   contoursQuery3, contoursTesting);
                }
                distanceMat.at<float>(aIndex, bIndex) = distance;
            }
        }
    }

    void displayMPEGResults()
    {
        const int FIRST_MANY=2*NSN;

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
                    if (distanceMat.at<float>(row,col) > distanceMat.at<float>(row,i))
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
        std::cout << "Test result: " << porc << "%" << std::endl;
        if (porc >= CURRENT_MAX_ACCUR)
            ts->set_failed_test_info(cvtest::TS::OK);
        else
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
    }

protected:
    int NSN;
    int NP;
    float CURRENT_MAX_ACCUR;
    vector<string> filenames;
    Mat distanceMat;
    compute cmp;
};

//------------------------------------------------------------------------
//                       Test Shape_SCD.regression
//------------------------------------------------------------------------

class computeShapeDistance_Chi
{
    Ptr <ShapeContextDistanceExtractor> mysc;
public:
    computeShapeDistance_Chi()
    {
        const int angularBins=12;
        const int radialBins=4;
        const float minRad=0.2f;
        const float maxRad=2;
        mysc = createShapeContextDistanceExtractor(angularBins, radialBins, minRad, maxRad);
        mysc->setIterations(1);
        mysc->setCostExtractor(createChiHistogramCostExtractor(30,0.15f));
        mysc->setTransformAlgorithm( createThinPlateSplineShapeTransformer() );
    }
    float operator()(vector <Point2f>& query1, vector <Point2f>& query2,
                     vector <Point2f>& query3, vector <Point2f>& testq)
    {
        return std::min(mysc->computeDistance(query1, testq),
                        std::min(mysc->computeDistance(query2, testq),
                                 mysc->computeDistance(query3, testq)));
    }
};

TEST(Shape_SCD, regression)
{
    const int NSN_val=5;//10;//20; //number of shapes per class
    const int NP_val=120; //number of points simplifying the contour
    const float CURRENT_MAX_ACCUR_val=95; //99% and 100% reached in several tests, 95 is fixed as minimum boundary
    ShapeBaseTest<float, computeShapeDistance_Chi> test(NSN_val, NP_val, CURRENT_MAX_ACCUR_val);
    test.safe_run();
}

//------------------------------------------------------------------------
//                       Test ShapeEMD_SCD.regression
//------------------------------------------------------------------------

class computeShapeDistance_EMD
{
    Ptr <ShapeContextDistanceExtractor> mysc;
public:
    computeShapeDistance_EMD()
    {
        const int angularBins=12;
        const int radialBins=4;
        const float minRad=0.2f;
        const float maxRad=2;
        mysc = createShapeContextDistanceExtractor(angularBins, radialBins, minRad, maxRad);
        mysc->setIterations(1);
        mysc->setCostExtractor( createEMDL1HistogramCostExtractor() );
        mysc->setTransformAlgorithm( createThinPlateSplineShapeTransformer() );
    }
    float operator()(vector <Point2f>& query1, vector <Point2f>& query2,
                     vector <Point2f>& query3, vector <Point2f>& testq)
    {
        return std::min(mysc->computeDistance(query1, testq),
                        std::min(mysc->computeDistance(query2, testq),
                                 mysc->computeDistance(query3, testq)));
    }
};

TEST(ShapeEMD_SCD, regression)
{
    const int NSN_val=5;//10;//20; //number of shapes per class
    const int NP_val=100; //number of points simplifying the contour
    const float CURRENT_MAX_ACCUR_val=95; //98% and 99% reached in several tests, 95 is fixed as minimum boundary
    ShapeBaseTest<float, computeShapeDistance_EMD> test(NSN_val, NP_val, CURRENT_MAX_ACCUR_val);
    test.safe_run();
}

//------------------------------------------------------------------------
//                       Test Hauss.regression
//------------------------------------------------------------------------

class computeShapeDistance_Haussdorf
{
    Ptr <HausdorffDistanceExtractor> haus;
public:
    computeShapeDistance_Haussdorf()
    {
        haus = createHausdorffDistanceExtractor();
    }
    float operator()(vector<Point> &query1, vector<Point> &query2,
                     vector<Point> &query3, vector<Point> &testq)
    {
        return std::min(haus->computeDistance(query1,testq),
                        std::min(haus->computeDistance(query2,testq),
                                 haus->computeDistance(query3,testq)));
    }
};

TEST(Hauss, regression)
{
    const int NSN_val=5;//10;//20; //number of shapes per class
    const int NP_val = 180; //number of points simplifying the contour
    const float CURRENT_MAX_ACCUR_val=85; //90% and 91% reached in several tests, 85 is fixed as minimum boundary
    ShapeBaseTest<int, computeShapeDistance_Haussdorf> test(NSN_val, NP_val, CURRENT_MAX_ACCUR_val);
    test.safe_run();
}
