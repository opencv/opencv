#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#include "opencv2/ts.hpp"
#include <opencv2/ts/cuda_test.hpp> // EXPECT_MAT_NEAR
#include "opencv2/ml.hpp"

#include <fstream>
using std::ifstream;

namespace opencv_test {

using namespace cv::ml;

#define CV_NBAYES   "nbayes"
#define CV_KNEAREST "knearest"
#define CV_SVM      "svm"
#define CV_EM       "em"
#define CV_ANN      "ann"
#define CV_DTREE    "dtree"
#define CV_BOOST    "boost"
#define CV_RTREES   "rtrees"
#define CV_ERTREES  "ertrees"
#define CV_SVMSGD   "svmsgd"

using cv::Ptr;
using cv::ml::StatModel;
using cv::ml::TrainData;
using cv::ml::NormalBayesClassifier;
using cv::ml::SVM;
using cv::ml::KNearest;
using cv::ml::ParamGrid;
using cv::ml::ANN_MLP;
using cv::ml::DTrees;
using cv::ml::Boost;
using cv::ml::RTrees;
using cv::ml::SVMSGD;

void defaultDistribs( Mat& means, vector<Mat>& covs, int type=CV_32FC1 );
void generateData( Mat& data, Mat& labels, const vector<int>& sizes, const Mat& _means, const vector<Mat>& covs, int dataType, int labelType );
int maxIdx( const vector<int>& count );
bool getLabelsMap( const Mat& labels, const vector<int>& sizes, vector<int>& labelsMap, bool checkClusterUniq=true );
bool calcErr( const Mat& labels, const Mat& origLabels, const vector<int>& sizes, float& err, bool labelsEquivalent = true, bool checkClusterUniq=true );

// used in LR test
bool calculateError( const Mat& _p_labels, const Mat& _o_labels, float& error);

} // namespace

#endif
