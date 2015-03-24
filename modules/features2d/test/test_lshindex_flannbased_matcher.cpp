/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2015 Ippei Ito.  All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

/*
 For OpenCV2.4/OpenCV3.0

 Test for Pull Request # 3829
 https://github.com/Itseez/opencv/pull/3829

 This test code creates brute force matcher for accuracy of reference, and the test target matcher.
 Then, add() and train() transformed query image descriptors, and some outlier images descriptors to both matchers.
 Then, compared with the query image by match() and findHomography() to detect outlier and calculate accuracy.
 And each drawMatches() images are saved, if SAVE_DRAW_MATCHES_IMAGES is true.
 Finally, compare accuracies between the brute force matcher and the test target matcher.

 The lsh algorithm uses std::random_shuffle in lsh_index.h to make the random indexes table.
 So, in relation to default random seed value of the execution environment or by using "srand(time(0)) function",
 the match time and accuracy of the match results are different, each time the code ran.
 And the match time becomes late in relation to the number of the hash collision times.
*/

#include "test_precomp.hpp"
#include "opencv2/ts.hpp"    // for FilePath::CreateFolder()
#include <time.h> // for time()

// If defined, the match time and accuracy of the match results are a little different, each time the code ran.
#define INIT_RANDOM_SEED

// If defined, some outlier images descriptors add() the matcher.
#define TRAIN_WITH_OUTLIER_IMAGES

// If true, save drawMatches() images.
#define SAVE_DRAW_MATCHES_IMAGES false

// if true, verbose output
#define SHOW_DEBUG_LOG true

#if CV_MAJOR_VERSION==2
#define OrbCreate new ORB(4000)
#elif CV_MAJOR_VERSION==3
#define OrbCreate ORB::create(4000)
#define AKazeCreate AKAZE::create()
#endif

using namespace cv;
using namespace std;

int testno_for_make_filename = 0;

// --------------------------------------------------------------------------------------
// Parameter class to transform query image
// --------------------------------------------------------------------------------------
class testparam
{
public:
    string transname;
    void(*transfunc)(float, const Mat&, Mat&);
    float from, to, step;
    testparam(string _transname, void(*_transfunc)(float, const Mat&, Mat&), float _from, float _to, float _step) :
        transname(_transname),
        transfunc(_transfunc),
        from(_from),
        to(_to),
        step(_step)
    {}
};

// --------------------------------------------------------------------------------------
// from matching_to_many_images.cpp
// --------------------------------------------------------------------------------------
int maskMatchesByTrainImgIdx(const vector<DMatch>& matches, int trainImgIdx, vector<char>& mask)
{
    int matchcnt = 0;
    mask.resize(matches.size());
    fill(mask.begin(), mask.end(), 0);
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i].imgIdx == trainImgIdx)
        {
            mask[i] = 1;
            matchcnt++;
        }
    }
    return matchcnt;
}

int calcHomographyAndInlierCount(const vector<KeyPoint>& query_kp, const vector<KeyPoint>& train_kp, const vector<DMatch>& match, vector<char> &mask, Mat &homography)
{
    // make query and current train image keypoint pairs
    std::vector<cv::Point2f> srcPoints, dstPoints;
    for (unsigned int i = 0; i < match.size(); ++i)
    {
        if (mask[i] != 0) // is current train image ?
        {
            srcPoints.push_back(query_kp[match[i].queryIdx].pt);
            dstPoints.push_back(train_kp[match[i].trainIdx].pt);
        }
    }
    // calc homography
    vector<uchar> inlierMask;
    homography = findHomography(srcPoints, dstPoints, RANSAC, 3.0, inlierMask);

    // update outlier mask
    int j = 0;
    for (unsigned int i = 0; i < match.size(); ++i)
    {
        if (mask[i] != 0) // is current train image ?
        {
            if (inlierMask.size() == 0 || inlierMask[j] == 0) // is outlier ?
            {
                mask[i] = 0;
            }
            j++;
        }
    }

    // count inlier
    int inlierCnt = 0;
    for (unsigned int i = 0; i < mask.size(); ++i)
    {
        if (mask[i] != 0)
        {
            inlierCnt++;
        }
    }
    return inlierCnt;
}

void drawDetectedRectangle(Mat& imgResult, const Mat& homography, const Mat& imgQuery)
{
    std::vector<Point2f> query_corners(4);
    query_corners[0] = Point(0, 0);
    query_corners[1] = Point(imgQuery.cols, 0);
    query_corners[2] = Point(imgQuery.cols, imgQuery.rows);
    query_corners[3] = Point(0, imgQuery.rows);
    std::vector<Point2f> train_corners(4);
    perspectiveTransform(query_corners, train_corners, homography);
    line(imgResult, train_corners[0] + query_corners[1], train_corners[1] + query_corners[1], Scalar(0, 255, 0), 4);
    line(imgResult, train_corners[1] + query_corners[1], train_corners[2] + query_corners[1], Scalar(0, 255, 0), 4);
    line(imgResult, train_corners[2] + query_corners[1], train_corners[3] + query_corners[1], Scalar(0, 255, 0), 4);
    line(imgResult, train_corners[3] + query_corners[1], train_corners[0] + query_corners[1], Scalar(0, 255, 0), 4);
}

// --------------------------------------------------------------------------------------
// transform query image, extract&compute, train, matching and save result image function
// --------------------------------------------------------------------------------------
typedef struct tagTrainInfo
{
    int traindesccnt;
    double traintime;
    double matchtime;
    double accuracy;
}TrainInfo;

TrainInfo transImgAndTrain(
    Feature2D *fe,
    DescriptorMatcher *matcher,
    const string &matchername,
    const Mat& imgQuery, const vector<KeyPoint>& query_kp, const Mat& query_desc,
    const vector<Mat>& imgOutliers, const vector<vector<KeyPoint> >& outliers_kp, const vector<Mat>& outliers_desc, const int totalOutlierDescCnt,
    const float t, const testparam &tp,
    const int testno, const bool bVerboseOutput, const bool bSaveDrawMatches)
{
    TrainInfo ti;

    // transform query image
    Mat imgTransform;
    (tp.transfunc)(t, imgQuery, imgTransform);

    // extract kp and compute desc from transformed query image
    vector<KeyPoint> trans_query_kp;
    Mat trans_query_desc;
#if CV_MAJOR_VERSION==2
    (*fe)(imgTransform, Mat(), trans_query_kp, trans_query_desc);
#elif CV_MAJOR_VERSION==3
    fe->detectAndCompute(imgTransform, Mat(), trans_query_kp, trans_query_desc);
#endif
    // add&train transformed query desc and outlier desc
    matcher->clear();
    matcher->add(vector<Mat>(1, trans_query_desc));
    double s = (double)getTickCount();
    matcher->train();
    ti.traintime = 1000.0*((double)getTickCount() - s) / getTickFrequency();
    ti.traindesccnt = trans_query_desc.rows;
#if defined(TRAIN_WITH_OUTLIER_IMAGES)
    // same as matcher->add(outliers_desc); matcher->train();
    for (unsigned int i = 0; i < outliers_desc.size(); ++i)
    {
        matcher->add(vector<Mat>(1, outliers_desc[i]));
        s = (double)getTickCount();
        matcher->train();
        ti.traintime += 1000.0*((double)getTickCount() - s) / getTickFrequency();
    }
    ti.traindesccnt += totalOutlierDescCnt;
#endif
    // matching
    vector<DMatch> match;
    s = (double)getTickCount();
    matcher->match(query_desc, match);
    ti.matchtime = 1000.0*((double)getTickCount() - s) / getTickFrequency();

    // prepare a directory and variables for save matching images
    vector<char> mask;
    Mat imgResult;
    const char resultDir[] = "result";
    if (bSaveDrawMatches)
    {
        testing::internal::FilePath fp = testing::internal::FilePath(resultDir);
        fp.CreateFolder();
    }

    char buff[2048];
    int matchcnt;

    // save query vs transformed query matching image with detected rectangle
    matchcnt = maskMatchesByTrainImgIdx(match, (int)0, mask);
    // calc homography and inlier
    Mat homography;
    int inlierCnt = calcHomographyAndInlierCount(query_kp, trans_query_kp, match, mask, homography);
    ti.accuracy = (double)inlierCnt / (double)mask.size()*100.0;
    drawMatches(imgQuery, query_kp, imgTransform, trans_query_kp, match, imgResult, Scalar::all(-1), Scalar::all(128), mask, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    if (inlierCnt)
    {
        // draw detected rectangle
        drawDetectedRectangle(imgResult, homography, imgQuery);
    }
    // draw status
    sprintf(buff, "%s accuracy:%-3.2f%% %d descriptors training time:%-3.2fms matching :%-3.2fms", matchername.c_str(), ti.accuracy, ti.traindesccnt, ti.traintime, ti.matchtime);
    putText(imgResult, buff, Point(0, 12), FONT_HERSHEY_PLAIN, 0.8, Scalar(0., 0., 255.));
    sprintf(buff, "%s/res%03d_%s_%s%.1f_inlier.png", resultDir, testno, matchername.c_str(), tp.transname.c_str(), t);
    if (bSaveDrawMatches && !imwrite(buff, imgResult)) cout << "Image " << buff << " can not be saved (may be because directory " << resultDir << " does not exist)." << endl;

#if defined(TRAIN_WITH_OUTLIER_IMAGES)
    // save query vs outlier matching image(s)
    for (unsigned int i = 0; i <imgOutliers.size(); ++i)
    {
        matchcnt = maskMatchesByTrainImgIdx(match, (int)i + 1, mask);
        drawMatches(imgQuery, query_kp, imgOutliers[i], outliers_kp[i], match, imgResult, Scalar::all(-1), Scalar::all(128), mask);//  , DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        sprintf(buff, "query_num:%d train_num:%d matched:%d %d descriptors training time:%-3.2fms matching :%-3.2fms", (int)query_kp.size(), (int)outliers_kp[i].size(), matchcnt, ti.traindesccnt, ti.traintime, ti.matchtime);
        putText(imgResult, buff, Point(0, 12), FONT_HERSHEY_PLAIN, 0.8, Scalar(0., 0., 255.));
        sprintf(buff, "%s/res%03d_%s_%s%.1f_outlier%02d.png", resultDir, testno, matchername.c_str(), tp.transname.c_str(), t, i);
        if (bSaveDrawMatches && !imwrite(buff, imgResult)) cout << "Image " << buff << " can not be saved (may be because directory " << resultDir << " does not exist)." << endl;
    }
#endif
    if (bVerboseOutput)
    {
        cout << tp.transname <<" image matching accuracy:" << ti.accuracy << "% " << ti.traindesccnt << " train:" << ti.traintime << "ms match:" << ti.matchtime << "ms" << endl;
    }

    return ti;
}

// --------------------------------------------------------------------------------------
// Main Test Class
// --------------------------------------------------------------------------------------
class CV_FeatureDetectorMatcherBaseTest : public cvtest::BaseTest
{
private:

    Ptr<DescriptorMatcher> bfmatcher;   // brute force matcher for accuracy of reference
    Ptr<DescriptorMatcher> flmatcher;   // flann matcher to test
    Ptr<Feature2D> fe;                  // feature detector extractor
    Mat imgQuery;                       // query image
    vector<Mat> imgOutliers;            // outlier image
    vector<KeyPoint> query_kp;          // query key points detect from imgQuery
    Mat query_desc;                     // query descriptors extract from imgQuery
    vector<vector<KeyPoint> > outliers_kp;
    vector<Mat> outliers_desc;
    int totalOutlierDescCnt;

    string flmatchername;
    testparam tp;
    double target_accuracy_margin_from_bfmatcher;

public:

    //
    // constructor
    //
    CV_FeatureDetectorMatcherBaseTest(testparam _tp, double _accuracy_margin, Ptr<Feature2D> _fe, DescriptorMatcher *_flmatcher, string _flmatchername, int norm_type_for_bfmatcher) :
        tp(_tp),
        fe(_fe),
        flmatcher(_flmatcher),
        flmatchername(_flmatchername),
        target_accuracy_margin_from_bfmatcher(_accuracy_margin)
    {
#if defined(INIT_RANDOM_SEED)
        // from test/test_eigen.cpp
        srand((unsigned int)time(0));
#endif
        // create brute force matcher for accuracy of reference
        bfmatcher = makePtr<BFMatcher>(norm_type_for_bfmatcher);
    }

    //
    // Main Test method
    //
    virtual void run(int)
    {
        // load query image
        string strQueryFile = string(cvtest::TS::ptr()->get_data_path()) + "shared/lena.png";
        imgQuery = imread(strQueryFile, 0);
        if (imgQuery.empty())
        {
            ts->printf(cvtest::TS::LOG, "Image %s can not be read.\n", strQueryFile.c_str());
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        // load outlier images
        char* outliers[] = { (char*)"baboon.png", (char*)"fruits.png", (char*)"airplane.png" };
        for (unsigned int i = 0; i < sizeof(outliers) / sizeof(char*); i++)
        {
            string strOutlierFile = string(cvtest::TS::ptr()->get_data_path()) + "shared/" + outliers[i];
            Mat imgOutlier = imread(strOutlierFile, 0);
            if (imgQuery.empty())
            {
                ts->printf(cvtest::TS::LOG, "Image %s can not be read.\n", strOutlierFile.c_str());
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
                return;
            }
            imgOutliers.push_back(imgOutlier);
        }

        // extract and compute keypoints and descriptors from query image
#if CV_MAJOR_VERSION==2
        (*fe)(imgQuery, Mat(), query_kp, query_desc);
#elif CV_MAJOR_VERSION==3
        fe->detectAndCompute(imgQuery, Mat(), query_kp, query_desc);
#endif
        // extract and compute keypoints and descriptors from outlier images
        fe->detect(imgOutliers, outliers_kp);
        ((DescriptorExtractor*)fe)->compute(imgOutliers, outliers_kp, outliers_desc);
        totalOutlierDescCnt = 0;
        for (unsigned int i = 0; i < outliers_desc.size(); ++i) totalOutlierDescCnt += outliers_desc[i].rows;

        if (SHOW_DEBUG_LOG)
        {
            cout << query_kp.size() << " keypoints extracted from query image." << endl;
#if defined(TRAIN_WITH_OUTLIER_IMAGES)
            cout << totalOutlierDescCnt << " keypoints extracted from outlier image(s)." << endl;
#endif
        }
        // compute brute force matcher accuracy for reference
        double totalTrainTime = 0.;
        double totalMatchTime = 0.;
        double totalAccuracy = 0.;
        int cnt = 0;
        for (float t = tp.from; t <= tp.to; t += tp.step, ++testno_for_make_filename, ++cnt)
        {
            if (SHOW_DEBUG_LOG) cout << "Test No." << testno_for_make_filename << " BFMatcher " << t;

            TrainInfo ti = transImgAndTrain(fe, bfmatcher, "BFMatcher",
                imgQuery, query_kp, query_desc,
                imgOutliers, outliers_kp, outliers_desc,
                totalOutlierDescCnt,
                t, tp, testno_for_make_filename, SHOW_DEBUG_LOG, SAVE_DRAW_MATCHES_IMAGES);
            totalTrainTime += ti.traintime;
            totalMatchTime += ti.matchtime;
            totalAccuracy += ti.accuracy;
        }
        double bf_average_accuracy = totalAccuracy / cnt;
        if (SHOW_DEBUG_LOG)
        {
            cout << "total training time: " << totalTrainTime << "ms" << endl;
            cout << "total matching time: " << totalMatchTime << "ms" << endl;
            cout << "average accuracy:" << bf_average_accuracy << "%" << endl;
        }

        // test the target matcher
        totalTrainTime = 0.;
        totalMatchTime = 0.;
        totalAccuracy = 0.;
        cnt = 0;
        for (float t = tp.from; t <= tp.to; t += tp.step, ++testno_for_make_filename, ++cnt)
        {
            if (SHOW_DEBUG_LOG) cout << "Test No." << testno_for_make_filename << " " << flmatchername << " " << t;

            TrainInfo ti = transImgAndTrain(fe, flmatcher, flmatchername,
                imgQuery, query_kp, query_desc,
                imgOutliers, outliers_kp, outliers_desc,
                totalOutlierDescCnt,
                t, tp, testno_for_make_filename, SHOW_DEBUG_LOG, SAVE_DRAW_MATCHES_IMAGES);

            totalTrainTime += ti.traintime;
            totalMatchTime += ti.matchtime;
            totalAccuracy += ti.accuracy;
        }
        double average_accuracy = totalAccuracy / cnt;
        double target_average_accuracy = bf_average_accuracy * target_accuracy_margin_from_bfmatcher;

        if (SHOW_DEBUG_LOG)
        {
            cout << "total training time: " << totalTrainTime << "ms" << endl;
            cout << "total matching time: " << totalMatchTime << "ms" << endl;
            cout << "average accuracy:" << average_accuracy << "%" << endl;
            cout << "threshold of the target matcher average accuracy as error :" << target_average_accuracy << "%" << endl;
            cout << "accuracy degraded " << (100.0 - (average_accuracy / bf_average_accuracy *100.0)) << "% from BFMatcher.(lower percentage is better)" << endl;
        }
        // compare accuracies between the brute force matcher and the test target matcher
        if (average_accuracy < target_average_accuracy)
        {
            ts->printf(cvtest::TS::LOG, "Bad average accuracy %f < %f while test %s %s query\n", average_accuracy, target_average_accuracy, flmatchername.c_str(), tp.transname.c_str());
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        }
        return;
    }

};

// --------------------------------------------------------------------------------------
// Transform Functions
// --------------------------------------------------------------------------------------
static void rotate(float deg, const Mat& src, Mat& dst)
{
    warpAffine(src, dst, getRotationMatrix2D(Point2f(src.cols / 2.0f, src.rows / 2.0f), deg, 1), src.size(), INTER_CUBIC);
}
static void scale(float scale, const Mat& src, Mat& dst)
{
    resize(src, dst, Size((int)(src.cols*scale), (int)(src.rows*scale)), INTER_AREA);
}
static void blur(float k, const Mat& src, Mat& dst)
{
    GaussianBlur(src, dst, Size((int)k, (int)k), 0);
}

// --------------------------------------------------------------------------------------
// Tests Registrations
// --------------------------------------------------------------------------------------
#define SHORT_LSH_KEY_ACCURACY_MARGIN 0.72      // The margin for FlannBasedMatcher. 28% degraded from BFMatcher(Actually, about  10..24% measured.lower percentage is better.) for lsh key size=16.
#define MIDDLE_LSH_KEY_ACCURACY_MARGIN 0.72     // The margin for FlannBasedMatcher. 28% degraded from BFMatcher(Actually, about   7..24% measured.lower percentage is better.) for lsh key size=24.
#define LONG_LSH_KEY_ACCURACY_MARGIN 0.90       // The margin for FlannBasedMatcher. 10% degraded from BFMatcher(Actually, about -29...7% measured.lower percentage is better.) for lsh key size=31.

TEST(BlurredQueryFlannBasedLshShortKeyMatcherAdditionalTrainTest, accuracy)
{
    testparam tp("blurred", blur, 1.0f, 11.0f, 2.0f);
    CV_FeatureDetectorMatcherBaseTest test(tp, SHORT_LSH_KEY_ACCURACY_MARGIN, OrbCreate, new FlannBasedMatcher(makePtr<flann::LshIndexParams>(1, 16, 2)), "FlannLsh(1, 16, 2)", NORM_HAMMING);
    test.safe_run();
}
TEST(BlurredQueryFlannBasedLshMiddleKeyMatcherAdditionalTrainTest, accuracy)
{
    testparam tp("blurred", blur, 1.0f, 11.0f, 2.0f);
    CV_FeatureDetectorMatcherBaseTest test(tp, MIDDLE_LSH_KEY_ACCURACY_MARGIN, OrbCreate, new FlannBasedMatcher(makePtr<flann::LshIndexParams>(1, 24, 2)), "FlannLsh(1, 24, 2)", NORM_HAMMING);
    test.safe_run();
}
TEST(BlurredQueryFlannBasedLshLongKeyMatcherAdditionalTrainTest, accuracy)
{
    testparam tp("blurred", blur, 1.0f, 11.0f, 2.0f);
    CV_FeatureDetectorMatcherBaseTest test(tp, LONG_LSH_KEY_ACCURACY_MARGIN, OrbCreate, new FlannBasedMatcher(makePtr<flann::LshIndexParams>(1, 31, 2)), "FlannLsh(1, 31, 2)", NORM_HAMMING);
    test.safe_run();
}

TEST(ScaledQueryFlannBasedLshShortKeyMatcherAdditionalTrainTest, accuracy)
{
    testparam tp("scaled", scale, 0.5f, 1.5f, 0.1f);
    CV_FeatureDetectorMatcherBaseTest test(tp, SHORT_LSH_KEY_ACCURACY_MARGIN, OrbCreate, new FlannBasedMatcher(makePtr<flann::LshIndexParams>(1, 16, 2)), "FlannLsh(1, 16, 2)", NORM_HAMMING);
    test.safe_run();
}
TEST(ScaledQueryFlannBasedLshMiddleKeyMatcherAdditionalTrainTest, accuracy)
{
    testparam tp("scaled", scale, 0.5f, 1.5f, 0.1f);
    CV_FeatureDetectorMatcherBaseTest test(tp, MIDDLE_LSH_KEY_ACCURACY_MARGIN, OrbCreate, new FlannBasedMatcher(makePtr<flann::LshIndexParams>(1, 24, 2)), "FlannLsh(1, 24, 2)", NORM_HAMMING);
    test.safe_run();
}
TEST(ScaledQueryFlannBasedLshLongKeyMatcherAdditionalTrainTest, accuracy)
{
    testparam tp("scaled", scale, 0.5f, 1.5f, 0.1f);
    CV_FeatureDetectorMatcherBaseTest test(tp, LONG_LSH_KEY_ACCURACY_MARGIN, OrbCreate, new FlannBasedMatcher(makePtr<flann::LshIndexParams>(1, 31, 2)), "FlannLsh(1, 31, 2)", NORM_HAMMING);
    test.safe_run();
}

TEST(RotatedQueryFlannBasedLshShortKeyMatcherAdditionalTrainTest, accuracy)
{
    testparam tp("rotated", rotate, 0.0f, 359.0f, 30.0f);
    CV_FeatureDetectorMatcherBaseTest test(tp, SHORT_LSH_KEY_ACCURACY_MARGIN, OrbCreate, new FlannBasedMatcher(makePtr<flann::LshIndexParams>(1, 16, 2)), "FlannLsh(1, 16, 2)", NORM_HAMMING);
    test.safe_run();
}
TEST(RotatedQueryFlannBasedLshMiddleKeyMatcherAdditionalTrainTest, accuracy)
{
    testparam tp("rotated", rotate, 0.0f, 359.0f, 30.0f);
    CV_FeatureDetectorMatcherBaseTest test(tp, MIDDLE_LSH_KEY_ACCURACY_MARGIN, OrbCreate, new FlannBasedMatcher(makePtr<flann::LshIndexParams>(1, 24, 2)), "FlannLsh(1, 24, 2)", NORM_HAMMING);
    test.safe_run();
}
TEST(RotatedQueryFlannBasedLshLongKeyMatcherAdditionalTrainTest, accuracy)
{
    testparam tp("rotated", rotate, 0.0f, 359.0f, 30.0f);
    CV_FeatureDetectorMatcherBaseTest test(tp, LONG_LSH_KEY_ACCURACY_MARGIN, OrbCreate, new FlannBasedMatcher(makePtr<flann::LshIndexParams>(1, 31, 2)), "FlannLsh(1, 31, 2)", NORM_HAMMING);
    test.safe_run();
}
