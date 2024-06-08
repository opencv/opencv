// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "opencv2/video/detail/tracking.detail.hpp"
#include "tracking_feature.hpp"

namespace cv {
namespace detail {
inline namespace tracking {
inline namespace internal {

class TrackerFeatureHAAR : public TrackerFeature
{
public:
    struct Params
    {
        Params();
        int numFeatures;  //!< # of rects
        Size rectSize;  //!< rect size
        bool isIntegral;  //!< true if input images are integral, false otherwise
    };

    TrackerFeatureHAAR(const TrackerFeatureHAAR::Params& parameters = TrackerFeatureHAAR::Params());

    virtual ~TrackerFeatureHAAR() CV_OVERRIDE {}

protected:
    bool computeImpl(const std::vector<Mat>& images, Mat& response) CV_OVERRIDE;

private:
    Params params;
    Ptr<CvHaarEvaluator> featureEvaluator;
};

/**
 * Parameters
 */

TrackerFeatureHAAR::Params::Params()
{
    numFeatures = 250;
    rectSize = Size(100, 100);
    isIntegral = false;
}

TrackerFeatureHAAR::TrackerFeatureHAAR(const TrackerFeatureHAAR::Params& parameters)
    : params(parameters)
{
    CvHaarFeatureParams haarParams;
    haarParams.numFeatures = params.numFeatures;
    haarParams.isIntegral = params.isIntegral;
    featureEvaluator = makePtr<CvHaarEvaluator>();
    featureEvaluator->init(&haarParams, 1, params.rectSize);
}

class Parallel_compute : public cv::ParallelLoopBody
{
private:
    Ptr<CvHaarEvaluator> featureEvaluator;
    std::vector<Mat> images;
    Mat response;
    //std::vector<CvHaarEvaluator::FeatureHaar> features;
public:
    Parallel_compute(Ptr<CvHaarEvaluator>& fe, const std::vector<Mat>& img, Mat& resp)
        : featureEvaluator(fe)
        , images(img)
        , response(resp)
    {

        //features = featureEvaluator->getFeatures();
    }

    virtual void operator()(const cv::Range& r) const CV_OVERRIDE
    {
        for (int jf = r.start; jf != r.end; ++jf)
        {
            int cols = images[jf].cols;
            int rows = images[jf].rows;
            for (int j = 0; j < featureEvaluator->getNumFeatures(); j++)
            {
                float res = 0;
                featureEvaluator->getFeatures()[j].eval(images[jf], Rect(0, 0, cols, rows), &res);
                (Mat_<float>(response))(j, jf) = res;
            }
        }
    }
};

bool TrackerFeatureHAAR::computeImpl(const std::vector<Mat>& images, Mat& response)
{
    if (images.empty())
    {
        return false;
    }

    int numFeatures = featureEvaluator->getNumFeatures();

    response = Mat_<float>(Size((int)images.size(), numFeatures));

    std::vector<CvHaarEvaluator::FeatureHaar> f = featureEvaluator->getFeatures();
    //for each sample compute #n_feature -> put each feature (n Rect) in response
    parallel_for_(Range(0, (int)images.size()), Parallel_compute(featureEvaluator, images, response));

    /*for ( size_t i = 0; i < images.size(); i++ )
  {
    int c = images[i].cols;
    int r = images[i].rows;
    for ( int j = 0; j < numFeatures; j++ )
    {
      float res = 0;
      featureEvaluator->getFeatures( j ).eval( images[i], Rect( 0, 0, c, r ), &res );
      ( Mat_<float>( response ) )( j, i ) = res;
    }
  }*/

    return true;
}

}}}}  // namespace cv::detail::tracking::internal
