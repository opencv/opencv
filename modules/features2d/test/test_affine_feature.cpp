// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

// #define GENERATE_DATA // generate data in debug mode

namespace opencv_test { namespace {

#ifndef GENERATE_DATA
static bool isSimilarKeypoints( const KeyPoint& p1, const KeyPoint& p2 )
{
    const float maxPtDif = 1.f;
    const float maxSizeDif = 1.f;
    const float maxAngleDif = 2.f;
    const float maxResponseDif = 0.1f;

    float dist = (float)cv::norm( p1.pt - p2.pt );
    return (dist < maxPtDif &&
            fabs(p1.size - p2.size) < maxSizeDif &&
            abs(p1.angle - p2.angle) < maxAngleDif &&
            abs(p1.response - p2.response) < maxResponseDif &&
            (p1.octave & 0xffff) == (p2.octave & 0xffff)     // do not care about sublayers and class_id
            );
}
#endif

TEST(Features2d_AFFINE_FEATURE, regression)
{
    Mat image = imread(cvtest::findDataFile("features2d/tsukuba.png"));
    string xml = cvtest::TS::ptr()->get_data_path() + "asift/regression_cpp.xml.gz";
    ASSERT_FALSE(image.empty());

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Default ASIFT generates too large descriptors. This test uses small maxTilt to suppress the size of testdata.
    Ptr<AffineFeature> ext = AffineFeature::create(SIFT::create(), 2, 0, 1.4142135623730951f, 144.0f);
    Mat mpt, msize, mangle, mresponse, moctave, mclass_id;
#ifdef GENERATE_DATA
    // calculate
    vector<KeyPoint> calcKeypoints;
    Mat calcDescriptors;
    ext->detectAndCompute(gray, Mat(), calcKeypoints, calcDescriptors, false);

    // create keypoints XML
    FileStorage fs(xml, FileStorage::WRITE);
    ASSERT_TRUE(fs.isOpened()) << xml;
    std::cout << "Creating keypoints XML..." << std::endl;

    mpt = Mat(calcKeypoints.size(), 2, CV_32F);
    msize = Mat(calcKeypoints.size(), 1, CV_32F);
    mangle = Mat(calcKeypoints.size(), 1, CV_32F);
    mresponse = Mat(calcKeypoints.size(), 1, CV_32F);
    moctave = Mat(calcKeypoints.size(), 1, CV_32S);
    mclass_id = Mat(calcKeypoints.size(), 1, CV_32S);

    for( size_t i = 0; i < calcKeypoints.size(); i++ )
    {
        const KeyPoint& key = calcKeypoints[i];
        mpt.at<float>(i, 0) = key.pt.x;
        mpt.at<float>(i, 1) = key.pt.y;
        msize.at<float>(i, 0) = key.size;
        mangle.at<float>(i, 0) = key.angle;
        mresponse.at<float>(i, 0) = key.response;
        moctave.at<int>(i, 0) = key.octave;
        mclass_id.at<int>(i, 0) = key.class_id;
    }

    fs << "keypoints_pt" << mpt;
    fs << "keypoints_size" << msize;
    fs << "keypoints_angle" << mangle;
    fs << "keypoints_response" << mresponse;
    fs << "keypoints_octave" << moctave;
    fs << "keypoints_class_id" << mclass_id;

    // create descriptor XML
    fs << "descriptors" << calcDescriptors;
    fs.release();
#else
    const float badCountsRatio = 0.01f;
    const float badDescriptorDist = 1.0f;
    const float maxBadKeypointsRatio = 0.15f;
    const float maxBadDescriptorRatio = 0.15f;

    // read keypoints
    vector<KeyPoint> validKeypoints;
    Mat validDescriptors;
    FileStorage fs(xml, FileStorage::READ);
    ASSERT_TRUE(fs.isOpened()) << xml;

    fs["keypoints_pt"] >> mpt;
    ASSERT_EQ(mpt.type(), CV_32F);
    fs["keypoints_size"] >> msize;
    ASSERT_EQ(msize.type(), CV_32F);
    fs["keypoints_angle"] >> mangle;
    ASSERT_EQ(mangle.type(), CV_32F);
    fs["keypoints_response"] >> mresponse;
    ASSERT_EQ(mresponse.type(), CV_32F);
    fs["keypoints_octave"] >> moctave;
    ASSERT_EQ(moctave.type(), CV_32S);
    fs["keypoints_class_id"] >> mclass_id;
    ASSERT_EQ(mclass_id.type(), CV_32S);

    validKeypoints.resize(mpt.rows);
    for( int i = 0; i < (int)validKeypoints.size(); i++ )
    {
        validKeypoints[i].pt.x = mpt.at<float>(i, 0);
        validKeypoints[i].pt.y = mpt.at<float>(i, 1);
        validKeypoints[i].size = msize.at<float>(i, 0);
        validKeypoints[i].angle = mangle.at<float>(i, 0);
        validKeypoints[i].response = mresponse.at<float>(i, 0);
        validKeypoints[i].octave = moctave.at<int>(i, 0);
        validKeypoints[i].class_id = mclass_id.at<int>(i, 0);
    }

    // read descriptors
    fs["descriptors"] >> validDescriptors;
    fs.release();

    // calc and compare keypoints
    vector<KeyPoint> calcKeypoints;
    ext->detectAndCompute(gray, Mat(), calcKeypoints, noArray(), false);

    float countRatio = (float)validKeypoints.size() / (float)calcKeypoints.size();
    ASSERT_LT(countRatio, 1 + badCountsRatio) << "Bad keypoints count ratio.";
    ASSERT_GT(countRatio, 1 - badCountsRatio) << "Bad keypoints count ratio.";

    int badPointCount = 0, commonPointCount = max((int)validKeypoints.size(), (int)calcKeypoints.size());
    for( size_t v = 0; v < validKeypoints.size(); v++ )
    {
        int nearestIdx = -1;
        float minDist = std::numeric_limits<float>::max();
        float angleDistOfNearest = std::numeric_limits<float>::max();

        for( size_t c = 0; c < calcKeypoints.size(); c++ )
        {
            if( validKeypoints[v].class_id != calcKeypoints[c].class_id )
                continue;
            float curDist = (float)cv::norm( calcKeypoints[c].pt - validKeypoints[v].pt );
            if( curDist < minDist )
            {
                minDist = curDist;
                nearestIdx = (int)c;
                angleDistOfNearest = abs( calcKeypoints[c].angle - validKeypoints[v].angle );
            }
            else if( curDist == minDist ) // the keypoints whose positions are same but angles are different
            {
                float angleDist = abs( calcKeypoints[c].angle - validKeypoints[v].angle );
                if( angleDist < angleDistOfNearest )
                {
                    nearestIdx = (int)c;
                    angleDistOfNearest = angleDist;
                }
            }
        }
        if( nearestIdx == -1 || !isSimilarKeypoints( validKeypoints[v], calcKeypoints[nearestIdx] ) )
            badPointCount++;
    }
    float badKeypointsRatio = (float)badPointCount / (float)commonPointCount;
    std::cout << "badKeypointsRatio: " << badKeypointsRatio << std::endl;
    ASSERT_LT( badKeypointsRatio , maxBadKeypointsRatio ) << "Bad accuracy!";

    // Calc and compare descriptors. This uses validKeypoints for extraction.
    Mat calcDescriptors;
    ext->detectAndCompute(gray, Mat(), validKeypoints, calcDescriptors, true);

    int dim = validDescriptors.cols;
    int badDescriptorCount = 0;
    L1<float> distance;

    for( int i = 0; i < (int)validKeypoints.size(); i++ )
    {
        float dist = distance( validDescriptors.ptr<float>(i), calcDescriptors.ptr<float>(i), dim );
        if( dist > badDescriptorDist )
            badDescriptorCount++;
    }
    float badDescriptorRatio = (float)badDescriptorCount / (float)validKeypoints.size();
    std::cout << "badDescriptorRatio: " << badDescriptorRatio << std::endl;
    ASSERT_LT( badDescriptorRatio, maxBadDescriptorRatio ) << "Too many descriptors mismatched.";
#endif
}

}} // namespace
