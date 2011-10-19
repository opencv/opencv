#include "perf_precomp.hpp"

#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;
using namespace perf;

typedef TestBaseWithParam<String> stitch;

PERF_TEST_P( stitch, a123, testing::Values("surf", "orb"))
{
    Mat pano;
    
    vector<Mat> imgs;
    imgs.push_back( imread( getDataPath("stitching/a1.jpg") ) );
    imgs.push_back( imread( getDataPath("stitching/a2.jpg") ) );
    imgs.push_back( imread( getDataPath("stitching/a3.jpg") ) );

    Stitcher::Status status;
    Ptr<detail::FeaturesFinder> featuresFinder = GetParam() == "orb"
            ? (detail::FeaturesFinder*)new detail::OrbFeaturesFinder()
            : (detail::FeaturesFinder*)new detail::SurfFeaturesFinder();

    Ptr<detail::FeaturesMatcher> featuresMatcher = GetParam() == "orb"
            ? new detail::BestOf2NearestMatcher(false, 0.3f)
            : new detail::BestOf2NearestMatcher(false, 0.65f);

    declare.time(30 * 20).iterations(50);

    while(next())
    {
        Stitcher stitcher = Stitcher::createDefault();
        stitcher.setFeaturesFinder(featuresFinder);
        stitcher.setFeaturesMatcher(featuresMatcher);

        startTimer();
        status = stitcher.stitch(imgs, pano);
        stopTimer();
    }
}

PERF_TEST_P( stitch, b12, testing::Values("surf", "orb"))
{
    Mat pano;
    
    vector<Mat> imgs;
    imgs.push_back( imread( getDataPath("stitching/b1.jpg") ) );
    imgs.push_back( imread( getDataPath("stitching/b2.jpg") ) );

    Stitcher::Status status;
    Ptr<detail::FeaturesFinder> featuresFinder = GetParam() == "orb"
            ? (detail::FeaturesFinder*)new detail::OrbFeaturesFinder()
            : (detail::FeaturesFinder*)new detail::SurfFeaturesFinder();

    Ptr<detail::FeaturesMatcher> featuresMatcher = GetParam() == "orb"
            ? new detail::BestOf2NearestMatcher(false, 0.3f)
            : new detail::BestOf2NearestMatcher(false, 0.65f);

    declare.time(30 * 20).iterations(50);

    while(next())
    {
        Stitcher stitcher = Stitcher::createDefault();
        stitcher.setFeaturesFinder(featuresFinder);
        stitcher.setFeaturesMatcher(featuresMatcher);

        startTimer();
        status = stitcher.stitch(imgs, pano);
        stopTimer();
    }
}
