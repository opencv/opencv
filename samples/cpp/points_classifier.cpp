#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/highgui.hpp"
#ifdef HAVE_OPENCV_OCL
#define _OCL_KNN_ 1 // select whether using ocl::KNN method or not, default is using
#define _OCL_SVM_ 1 // select whether using ocl::svm method or not, default is using
#include "opencv2/ocl/ocl.hpp"
#endif

#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

const Scalar WHITE_COLOR = Scalar(255,255,255);
const string winName = "points";
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

static void on_mouse( int event, int x, int y, int /*flags*/, void* )
{
    if( img.empty() )
        return;

    int updateFlag = 0;

    if( event == EVENT_LBUTTONUP )
    {
        trainedPoints.push_back( Point(x,y) );
        trainedPointsMarkers.push_back( currentClass );
        classCounters[currentClass]++;
        updateFlag = true;
    }

    //draw
    if( updateFlag )
    {
        img = Scalar::all(0);

        // draw points
        for( size_t i = 0; i < trainedPoints.size(); i++ )
        {
            Vec3b c = classColors[trainedPointsMarkers[i]];
            circle( img, trainedPoints[i], 5, Scalar(c), -1 );
        }

        imshow( winName, img );
   }
}

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

static void predict_and_paint(const Ptr<StatModel>& model, Mat& dst)
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
}

#if _NBC_
static void find_decision_boundary_NBC()
{
    // learn classifier
    Ptr<NormalBayesClassifier> normalBayesClassifier = StatModel::train<NormalBayesClassifier>(prepare_train_data());

    predict_and_paint(normalBayesClassifier, imgDst);
}
#endif


#if _KNN_
static void find_decision_boundary_KNN( int K )
{

    Ptr<KNearest> knn = KNearest::create();
    knn->setDefaultK(K);
    knn->setIsClassifier(true);
    knn->train(prepare_train_data());
    predict_and_paint(knn, imgDst);
}
#endif

#if _SVM_
static void find_decision_boundary_SVM( double C )
{
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::POLY); //SVM::LINEAR;
    svm->setDegree(0.5);
    svm->setGamma(1);
    svm->setCoef0(1);
    svm->setNu(0.5);
    svm->setP(0);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, 0.01));
    svm->setC(C);
    svm->train(prepare_train_data());
    predict_and_paint(svm, imgDst);

    Mat sv = svm->getSupportVectors();
    for( int i = 0; i < sv.rows; i++ )
    {
        const float* supportVector = sv.ptr<float>(i);
        circle( imgDst, Point(saturate_cast<int>(supportVector[0]),saturate_cast<int>(supportVector[1])), 5, Scalar(255,255,255), -1 );
    }
}
#endif

#if _DT_
static void find_decision_boundary_DT()
{
    Ptr<DTrees> dtree = DTrees::create();
    dtree->setMaxDepth(8);
    dtree->setMinSampleCount(2);
    dtree->setUseSurrogates(false);
    dtree->setCVFolds(0); // the number of cross-validation folds
    dtree->setUse1SERule(false);
    dtree->setTruncatePrunedTree(false);
    dtree->train(prepare_train_data());
    predict_and_paint(dtree, imgDst);
}
#endif

#if _BT_
static void find_decision_boundary_BT()
{
    Ptr<Boost> boost = Boost::create();
    boost->setBoostType(Boost::DISCRETE);
    boost->setWeakCount(100);
    boost->setWeightTrimRate(0.95);
    boost->setMaxDepth(2);
    boost->setUseSurrogates(false);
    boost->setPriors(Mat());
    boost->train(prepare_train_data());
    predict_and_paint(boost, imgDst);
}

#endif

#if _GBT_
static void find_decision_boundary_GBT()
{
    GBTrees::Params params( GBTrees::DEVIANCE_LOSS, // loss_function_type
                         100, // weak_count
                         0.1f, // shrinkage
                         1.0f, // subsample_portion
                         2, // max_depth
                         false // use_surrogates )
                         );

    Ptr<GBTrees> gbtrees = StatModel::train<GBTrees>(prepare_train_data(), params);
    predict_and_paint(gbtrees, imgDst);
}
#endif

#if _RF_
static void find_decision_boundary_RF()
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
    predict_and_paint(rtrees, imgDst);
}

#endif

#if _ANN_
static void find_decision_boundary_ANN( const Mat&  layer_sizes )
{
    Mat trainClasses = Mat::zeros( (int)trainedPoints.size(), (int)classColors.size(), CV_32FC1 );
    for( int i = 0; i < trainClasses.rows; i++ )
    {
        trainClasses.at<float>(i, trainedPointsMarkers[i]) = 1.f;
    }

    Mat samples = prepare_train_samples(trainedPoints);
    Ptr<TrainData> tdata = TrainData::create(samples, ROW_SAMPLE, trainClasses);

    Ptr<ANN_MLP> ann = ANN_MLP::create();
    ann->setLayerSizes(layer_sizes);
    ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
    ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 300, FLT_EPSILON));
    ann->setTrainMethod(ANN_MLP::BACKPROP, 0.001);
    ann->train(tdata);
    predict_and_paint(ann, imgDst);
}
#endif

#if _EM_
static void find_decision_boundary_EM()
{
    img.copyTo( imgDst );

    Mat samples = prepare_train_samples(trainedPoints);

    int i, j, nmodels = (int)classColors.size();
    vector<Ptr<EM> > em_models(nmodels);
    Mat modelSamples;

    for( i = 0; i < nmodels; i++ )
    {
        const int componentCount = 3;

        modelSamples.release();
        for( j = 0; j < samples.rows; j++ )
        {
            if( trainedPointsMarkers[j] == i )
                modelSamples.push_back(samples.row(j));
        }

        // learn models
        if( !modelSamples.empty() )
        {
            Ptr<EM> em = EM::create();
            em->setClustersNumber(componentCount);
            em->setCovarianceMatrixType(EM::COV_MAT_DIAGONAL);
            em->trainEM(modelSamples, noArray(), noArray(), noArray());
            em_models[i] = em;
        }
    }

    // classify coordinate plane points using the bayes classifier, i.e.
    // y(x) = arg max_i=1_modelsCount likelihoods_i(x)
    Mat testSample(1, 2, CV_32FC1 );
    Mat logLikelihoods(1, nmodels, CV_64FC1, Scalar(-DBL_MAX));

    for( int y = 0; y < img.rows; y += testStep )
    {
        for( int x = 0; x < img.cols; x += testStep )
        {
            testSample.at<float>(0) = (float)x;
            testSample.at<float>(1) = (float)y;

            for( i = 0; i < nmodels; i++ )
            {
                if( !em_models[i].empty() )
                    logLikelihoods.at<double>(i) = em_models[i]->predict2(testSample, noArray())[0];
            }
            Point maxLoc;
            minMaxLoc(logLikelihoods, 0, 0, 0, &maxLoc);
            imgDst.at<Vec3b>(y, x) = classColors[maxLoc.x];
        }
    }
}
#endif

int main()
{
    cout << "Use:" << endl
         << "  key '0' .. '1' - switch to class #n" << endl
         << "  left mouse button - to add new point;" << endl
         << "  key 'r' - to run the ML model;" << endl
         << "  key 'i' - to init (clear) the data." << endl << endl;

    cv::namedWindow( "points", 1 );
    img.create( 480, 640, CV_8UC3 );
    imgDst.create( 480, 640, CV_8UC3 );

    imshow( "points", img );
    setMouseCallback( "points", on_mouse );

    classColors[0] = Vec3b(0, 255, 0);
    classColors[1] = Vec3b(0, 0, 255);

    for(;;)
    {
        uchar key = (uchar)waitKey();

        if( key == 27 ) break;

        if( key == 'i' ) // init
        {
            img = Scalar::all(0);

            trainedPoints.clear();
            trainedPointsMarkers.clear();
            classCounters.assign(MAX_CLASSES, 0);

            imshow( winName, img );
        }

        if( key == '0' || key == '1' )
        {
            currentClass = key - '0';
        }

        if( key == 'r' ) // run
        {
            double minVal = 0;
            minMaxLoc(classCounters, &minVal, 0, 0, 0);
            if( minVal == 0 )
            {
                printf("each class should have at least 1 point\n");
                continue;
            }
            img.copyTo( imgDst );
#if _NBC_
            find_decision_boundary_NBC();
            imshow( "NormalBayesClassifier", imgDst );
#endif
#if _KNN_
            find_decision_boundary_KNN( 3 );
            imshow( "kNN", imgDst );

            find_decision_boundary_KNN( 15 );
            imshow( "kNN2", imgDst );
#endif

#if _SVM_
            //(1)-(2)separable and not sets

            find_decision_boundary_SVM( 1 );
            imshow( "classificationSVM1", imgDst );

            find_decision_boundary_SVM( 10 );
            imshow( "classificationSVM2", imgDst );
#endif

#if _DT_
            find_decision_boundary_DT();
            imshow( "DT", imgDst );
#endif

#if _BT_
            find_decision_boundary_BT();
            imshow( "BT", imgDst);
#endif

#if _GBT_
            find_decision_boundary_GBT();
            imshow( "GBT", imgDst);
#endif

#if _RF_
            find_decision_boundary_RF();
            imshow( "RF", imgDst);
#endif

#if _ANN_
            Mat layer_sizes1( 1, 3, CV_32SC1 );
            layer_sizes1.at<int>(0) = 2;
            layer_sizes1.at<int>(1) = 5;
            layer_sizes1.at<int>(2) = (int)classColors.size();
            find_decision_boundary_ANN( layer_sizes1 );
            imshow( "ANN", imgDst );
#endif

#if _EM_
            find_decision_boundary_EM();
            imshow( "EM", imgDst );
#endif
        }
    }

    return 0;
}
