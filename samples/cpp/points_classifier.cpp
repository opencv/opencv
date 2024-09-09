#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::ml;

const Scalar WHITE_COLOR = Scalar(255,255,255);
const int testStep = 5;

Mat img, imgDst;
RNG rng;

vector<Point>  trainedPoints;
vector<int>    trainedPointsMarkers;
const int MAX_CLASSES = 2;
vector<Vec3b>  classColors(MAX_CLASSES);
int currentClass = 0;
vector<int> classCounters(MAX_CLASSES);

#define _NBC_ 1 // normal Bayessian classifier
#define _KNN_ 1 // k nearest neighbors classifier
#define _SVM_ 1 // support vectors machine
#define _DT_  1 // decision tree
#define _BT_  1 // ADA Boost
#define _GBT_ 0 // gradient boosted trees
#define _RF_  1 // random forest
#define _ANN_ 1 // artificial neural networks
#define _EM_  1 // expectation-maximization

static Mat prepare_train_samples(const vector<Point>& pts)
{
    Mat samples;
    Mat(pts).reshape(1, (int)pts.size()).convertTo(samples, CV_32F);
    return samples;
}

static Ptr<TrainData> prepare_train_data()
{
    Mat samples = prepare_train_samples(trainedPoints);
    return TrainData::create(samples, ROW_SAMPLE, Mat(trainedPointsMarkers));
}

static void predict_and_save(const Ptr<StatModel>& model, Mat& dst, const string& filename)
{
    Mat testSample( 1, 2, CV_32FC1 );
    for( int y = 0; y < img.rows; y += testStep )
    {
        for( int x = 0; x < img.cols; x += testStep )
        {
            testSample.at<float>(0) = (float)x;
            testSample.at<float>(1) = (float)y;

            int response = (int)model->predict( testSample );
            dst.at<Vec3b>(y, x) = classColors[response];
        }
    }
    imwrite(filename, dst);
    cout << "Output saved to: " << filename << endl;
}

#if _NBC_
static void find_decision_boundary_NBC(const string& filename)
{
    Ptr<NormalBayesClassifier> normalBayesClassifier = StatModel::train<NormalBayesClassifier>(prepare_train_data());
    predict_and_save(normalBayesClassifier, imgDst, filename);
}
#endif

#if _KNN_
static void find_decision_boundary_KNN(int K, const string& filename)
{
    Ptr<KNearest> knn = KNearest::create();
    knn->setDefaultK(K);
    knn->setIsClassifier(true);
    knn->train(prepare_train_data());
    predict_and_save(knn, imgDst, filename);
}
#endif

#if _SVM_
static void find_decision_boundary_SVM(double C, const string& filename)
{
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::POLY);
    svm->setDegree(0.5);
    svm->setGamma(1);
    svm->setCoef0(1);
    svm->setNu(0.5);
    svm->setP(0);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, 0.01));
    svm->setC(C);
    svm->train(prepare_train_data());
    predict_and_save(svm, imgDst, filename);
}
#endif

#if _DT_
static void find_decision_boundary_DT(const string& filename)
{
    Ptr<DTrees> dtree = DTrees::create();
    dtree->setMaxDepth(8);
    dtree->setMinSampleCount(2);
    dtree->setUseSurrogates(false);
    dtree->setCVFolds(0);
    dtree->setUse1SERule(false);
    dtree->setTruncatePrunedTree(false);
    dtree->train(prepare_train_data());
    predict_and_save(dtree, imgDst, filename);
}
#endif

#if _BT_
static void find_decision_boundary_BT(const string& filename)
{
    Ptr<Boost> boost = Boost::create();
    boost->setBoostType(Boost::DISCRETE);
    boost->setWeakCount(100);
    boost->setWeightTrimRate(0.95);
    boost->setMaxDepth(2);
    boost->setUseSurrogates(false);
    boost->setPriors(Mat());
    boost->train(prepare_train_data());
    predict_and_save(boost, imgDst, filename);
}
#endif

#if _RF_
static void find_decision_boundary_RF(const string& filename)
{
    Ptr<RTrees> rtrees = RTrees::create();
    rtrees->setMaxDepth(4);
    rtrees->setMinSampleCount(2);
    rtrees->setRegressionAccuracy(0.f);
    rtrees->setUseSurrogates(false);
    rtrees->setMaxCategories(16);
    rtrees->setPriors(Mat());
    rtrees->setCalculateVarImportance(false);
    rtrees->setActiveVarCount(1);
    rtrees->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 5, 0));
    rtrees->train(prepare_train_data());
    predict_and_save(rtrees, imgDst, filename);
}
#endif

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, "{help h||}{@output| |output image filename}");
    parser.about("This sample demonstrates various ML classifiers.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    string outputFileName = parser.get<string>("@output");
    if (outputFileName.empty())
    {
        cout << "Output file path is required." << endl;
        return 1;
    }

    img.create(480, 640, CV_8UC3);
    imgDst.create(480, 640, CV_8UC3);

    classColors[0] = Vec3b(0, 255, 0);
    classColors[1] = Vec3b(0, 0, 255);

    // For demonstration, we add some sample points to train
    // These points should be replaced by actual training data in a real scenario
    trainedPoints.push_back(Point(100, 100));
    trainedPointsMarkers.push_back(0);
    classCounters[0]++;
    trainedPoints.push_back(Point(500, 100));
    trainedPointsMarkers.push_back(1);
    classCounters[1]++;

    // Example usage of classifiers
#if _NBC_
    find_decision_boundary_NBC(outputFileName);
#endif
#if _KNN_
    find_decision_boundary_KNN(3, outputFileName);
#endif
#if _SVM_
    find_decision_boundary_SVM(1.0, outputFileName);
#endif
#if _DT_
    find_decision_boundary_DT(outputFileName);
#endif
#if _BT_
    find_decision_boundary_BT(outputFileName);
#endif
#if _RF_
    find_decision_boundary_RF(outputFileName);
#endif

    return 0;
}

