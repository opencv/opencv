#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>

using namespace std;
using namespace cv;

const Scalar WHITE_COLOR = CV_RGB(255,255,255);
const string winName = "points";
const int testStep = 5;

Mat img, imgDst;
RNG rng;

vector<Point>  trainedPoints;
vector<int>    trainedPointsMarkers;
vector<Scalar> classColors;

#define _NBC_ 0 // normal Bayessian classifier
#define _KNN_ 0 // k nearest neighbors classifier
#define _SVM_ 0 // support vectors machine
#define _DT_  1 // decision tree
#define _BT_  0 // ADA Boost
#define _GBT_ 0 // gradient boosted trees
#define _RF_  0 // random forest
#define _ERT_ 0 // extremely randomized trees
#define _ANN_ 0 // artificial neural networks
#define _EM_  0 // expectation-maximization

void on_mouse( int event, int x, int y, int /*flags*/, void* )
{
    if( img.empty() )
        return;

    int updateFlag = 0;

    if( event == CV_EVENT_LBUTTONUP )
    {
        if( classColors.empty() )
            return;

        trainedPoints.push_back( Point(x,y) );
        trainedPointsMarkers.push_back( (int)(classColors.size()-1) );
        updateFlag = true;
    }
    else if( event == CV_EVENT_RBUTTONUP )
    {
#if _BT_
        if( classColors.size() < 2 )
        {
#endif
            classColors.push_back( Scalar((uchar)rng(256), (uchar)rng(256), (uchar)rng(256)) );
            updateFlag = true;
#if _BT_
        }
        else
            cout << "New class can not be added, because CvBoost can only be used for 2-class classification" << endl;
#endif

    }

    //draw
    if( updateFlag )
    {
        img = Scalar::all(0);

        // put the text
        stringstream text;
        text << "current class " << classColors.size()-1;
        putText( img, text.str(), Point(10,25), CV_FONT_HERSHEY_SIMPLEX, 0.8f, WHITE_COLOR, 2 );

        text.str("");
        text << "total classes " << classColors.size();
        putText( img, text.str(), Point(10,50), CV_FONT_HERSHEY_SIMPLEX, 0.8f, WHITE_COLOR, 2 );

        text.str("");
        text << "total points " << trainedPoints.size();
        putText(img, text.str(), cvPoint(10,75), CV_FONT_HERSHEY_SIMPLEX, 0.8f, WHITE_COLOR, 2 );

        // draw points
        for( size_t i = 0; i < trainedPoints.size(); i++ )
            circle( img, trainedPoints[i], 5, classColors[trainedPointsMarkers[i]], -1 );

        imshow( winName, img );
   }
}

void prepare_train_data( Mat& samples, Mat& classes )
{
    Mat( trainedPoints ).copyTo( samples );
    Mat( trainedPointsMarkers ).copyTo( classes );

    // reshape trainData and change its type
    samples = samples.reshape( 1, samples.rows );
    samples.convertTo( samples, CV_32FC1 );
}

#if _NBC_
void find_decision_boundary_NBC()
{
    img.copyTo( imgDst );

    Mat trainSamples, trainClasses;
    prepare_train_data( trainSamples, trainClasses );

    // learn classifier
    CvNormalBayesClassifier normalBayesClassifier( trainSamples, trainClasses );

    Mat testSample( 1, 2, CV_32FC1 );
    for( int y = 0; y < img.rows; y += testStep )
    {
        for( int x = 0; x < img.cols; x += testStep )
        {
            testSample.at<float>(0) = (float)x;
            testSample.at<float>(1) = (float)y;

            int response = (int)normalBayesClassifier.predict( testSample );
            circle( imgDst, Point(x,y), 1, classColors[response] );
        }
    }
}
#endif


#if _KNN_
void find_decision_boundary_KNN( int K )
{
    img.copyTo( imgDst );

    Mat trainSamples, trainClasses;
    prepare_train_data( trainSamples, trainClasses );

    // learn classifier
    CvKNearest knnClassifier( trainSamples, trainClasses, Mat(), false, K );

    Mat testSample( 1, 2, CV_32FC1 );
    for( int y = 0; y < img.rows; y += testStep )
    {
        for( int x = 0; x < img.cols; x += testStep )
        {
            testSample.at<float>(0) = (float)x;
            testSample.at<float>(1) = (float)y;

            int response = (int)knnClassifier.find_nearest( testSample, K );
            circle( imgDst, Point(x,y), 1, classColors[response] );
        }
    }
}
#endif

#if _SVM_
void find_decision_boundary_SVM( CvSVMParams params )
{
    img.copyTo( imgDst );

    Mat trainSamples, trainClasses;
    prepare_train_data( trainSamples, trainClasses );

    // learn classifier
    CvSVM svmClassifier( trainSamples, trainClasses, Mat(), Mat(), params );

    Mat testSample( 1, 2, CV_32FC1 );
    for( int y = 0; y < img.rows; y += testStep )
    {
        for( int x = 0; x < img.cols; x += testStep )
        {
            testSample.at<float>(0) = (float)x;
            testSample.at<float>(1) = (float)y;

            int response = (int)svmClassifier.predict( testSample );
            circle( imgDst, Point(x,y), 2, classColors[response], 1 );
        }
    }


    for( int i = 0; i < svmClassifier.get_support_vector_count(); i++ )
    {
        const float* supportVector = svmClassifier.get_support_vector(i);
        circle( imgDst, Point(supportVector[0],supportVector[1]), 5, CV_RGB(255,255,255), -1 );
    }

}
#endif

#if _DT_
void find_decision_boundary_DT()
{
    img.copyTo( imgDst );

    Mat trainSamples, trainClasses;
    prepare_train_data( trainSamples, trainClasses );

    // learn classifier
    CvDTree  dtree;

    Mat var_types( 1, trainSamples.cols + 1, CV_8UC1, Scalar(CV_VAR_ORDERED) );
    var_types.at<uchar>( trainSamples.cols ) = CV_VAR_CATEGORICAL;

    CvDTreeParams params;
    params.max_depth = 8;
    params.min_sample_count = 2;
    params.use_surrogates = false;
    params.cv_folds = 0; // the number of cross-validation folds
    params.use_1se_rule = false;
    params.truncate_pruned_tree = false;

    dtree.train( trainSamples, CV_ROW_SAMPLE, trainClasses,
                 Mat(), Mat(), var_types, Mat(), params );

    Mat testSample(1, 2, CV_32FC1 );
    for( int y = 0; y < img.rows; y += testStep )
    {
        for( int x = 0; x < img.cols; x += testStep )
        {
            testSample.at<float>(0) = (float)x;
            testSample.at<float>(1) = (float)y;

            int response = (int)dtree.predict( testSample )->value;
            circle( imgDst, Point(x,y), 2, classColors[response], 1 );
        }
    }
}
#endif

#if _BT_
void find_decision_boundary_BT()
{
    img.copyTo( imgDst );

    Mat trainSamples, trainClasses;
    prepare_train_data( trainSamples, trainClasses );

    // learn classifier
    CvBoost  boost;

    Mat var_types( 1, trainSamples.cols + 1, CV_8UC1, Scalar(CV_VAR_ORDERED) );
    var_types.at<uchar>( trainSamples.cols ) = CV_VAR_CATEGORICAL;

    CvBoostParams  params( CvBoost::DISCRETE, // boost_type
                           100, // weak_count
                           0.95, // weight_trim_rate
                           2, // max_depth
                           false, //use_surrogates
                           0 // priors
                         );

    boost.train( trainSamples, CV_ROW_SAMPLE, trainClasses, Mat(), Mat(), var_types, Mat(), params );

    Mat testSample(1, 2, CV_32FC1 );
    for( int y = 0; y < img.rows; y += testStep )
    {
        for( int x = 0; x < img.cols; x += testStep )
        {
            testSample.at<float>(0) = (float)x;
            testSample.at<float>(1) = (float)y;

            int response = (int)boost.predict( testSample );
            circle( imgDst, Point(x,y), 2, classColors[response], 1 );
        }
    }
}

#endif

#if _GBT_
void find_decision_boundary_GBT()
{
    img.copyTo( imgDst );

    Mat trainSamples, trainClasses;
    prepare_train_data( trainSamples, trainClasses );

    // learn classifier
    CvGBTrees gbtrees;

    Mat var_types( 1, trainSamples.cols + 1, CV_8UC1, Scalar(CV_VAR_ORDERED) );
    var_types.at<uchar>( trainSamples.cols ) = CV_VAR_CATEGORICAL;

    CvGBTreesParams  params( CvGBTrees::DEVIANCE_LOSS, // loss_function_type
                             100, // weak_count
                             0.1f, // shrinkage
                             1.0f, // subsample_portion
                             2, // max_depth
                             false // use_surrogates )
                           );

    gbtrees.train( trainSamples, CV_ROW_SAMPLE, trainClasses, Mat(), Mat(), var_types, Mat(), params );

    Mat testSample(1, 2, CV_32FC1 );
    for( int y = 0; y < img.rows; y += testStep )
    {
        for( int x = 0; x < img.cols; x += testStep )
        {
            testSample.at<float>(0) = (float)x;
            testSample.at<float>(1) = (float)y;

            int response = (int)gbtrees.predict( testSample );
            circle( imgDst, Point(x,y), 2, classColors[response], 1 );
        }
    }
}

#endif

#if _RF_
void find_decision_boundary_RF()
{
    img.copyTo( imgDst );

    Mat trainSamples, trainClasses;
    prepare_train_data( trainSamples, trainClasses );

    // learn classifier
    CvRTrees  rtrees;
    CvRTParams  params( 4, // max_depth,
                        2, // min_sample_count,
                        0.f, // regression_accuracy,
                        false, // use_surrogates,
                        16, // max_categories,
                        0, // priors,
                        false, // calc_var_importance,
                        1, // nactive_vars,
                        5, // max_num_of_trees_in_the_forest,
                        0, // forest_accuracy,
                        CV_TERMCRIT_ITER // termcrit_type
                       );

    rtrees.train( trainSamples, CV_ROW_SAMPLE, trainClasses, Mat(), Mat(), Mat(), Mat(), params );

    Mat testSample(1, 2, CV_32FC1 );
    for( int y = 0; y < img.rows; y += testStep )
    {
        for( int x = 0; x < img.cols; x += testStep )
        {
            testSample.at<float>(0) = (float)x;
            testSample.at<float>(1) = (float)y;

            int response = (int)rtrees.predict( testSample );
            circle( imgDst, Point(x,y), 2, classColors[response], 1 );
        }
    }
}

#endif

#if _ERT_
void find_decision_boundary_ERT()
{
    img.copyTo( imgDst );

    Mat trainSamples, trainClasses;
    prepare_train_data( trainSamples, trainClasses );

    // learn classifier
    CvERTrees ertrees;

    Mat var_types( 1, trainSamples.cols + 1, CV_8UC1, Scalar(CV_VAR_ORDERED) );
    var_types.at<uchar>( trainSamples.cols ) = CV_VAR_CATEGORICAL;

    CvRTParams  params( 4, // max_depth,
                        2, // min_sample_count,
                        0.f, // regression_accuracy,
                        false, // use_surrogates,
                        16, // max_categories,
                        0, // priors,
                        false, // calc_var_importance,
                        1, // nactive_vars,
                        5, // max_num_of_trees_in_the_forest,
                        0, // forest_accuracy,
                        CV_TERMCRIT_ITER // termcrit_type
                       );

    ertrees.train( trainSamples, CV_ROW_SAMPLE, trainClasses, Mat(), Mat(), var_types, Mat(), params );

    Mat testSample(1, 2, CV_32FC1 );
    for( int y = 0; y < img.rows; y += testStep )
    {
        for( int x = 0; x < img.cols; x += testStep )
        {
            testSample.at<float>(0) = (float)x;
            testSample.at<float>(1) = (float)y;

            int response = (int)ertrees.predict( testSample );
            circle( imgDst, Point(x,y), 2, classColors[response], 1 );
        }
    }
}
#endif

#if _ANN_
void find_decision_boundary_ANN( const Mat&  layer_sizes )
{
    img.copyTo( imgDst );

    Mat trainSamples, trainClasses;
    prepare_train_data( trainSamples, trainClasses );

    // prerare trainClasses
    trainClasses.create( trainedPoints.size(), classColors.size(), CV_32FC1 );
    for( int i = 0; i <  trainClasses.rows; i++ )
    {
        for( int k = 0; k < trainClasses.cols; k++ )
        {
            if( k == trainedPointsMarkers[i] )
                trainClasses.at<float>(i,k) = 1;
            else
                trainClasses.at<float>(i,k) = 0;
        }
    }

    Mat weights( 1, trainedPoints.size(), CV_32FC1, Scalar::all(1) );

    // learn classifier
    CvANN_MLP  ann( layer_sizes, CvANN_MLP::SIGMOID_SYM, 1, 1 );
    ann.train( trainSamples, trainClasses, weights );

    Mat testSample( 1, 2, CV_32FC1 );
    for( int y = 0; y < img.rows; y += testStep )
    {
        for( int x = 0; x < img.cols; x += testStep )
        {
            testSample.at<float>(0) = (float)x;
            testSample.at<float>(1) = (float)y;

            Mat outputs( 1, classColors.size(), CV_32FC1, testSample.data );
            ann.predict( testSample, outputs );
            Point maxLoc;
            minMaxLoc( outputs, 0, 0, 0, &maxLoc );
            circle( imgDst, Point(x,y), 2, classColors[maxLoc.x], 1 );
        }
    }
}
#endif

#if _EM_
void find_decision_boundary_EM()
{
    img.copyTo( imgDst );

    Mat trainSamples, trainClasses;
    prepare_train_data( trainSamples, trainClasses );

    vector<cv::EM> em_models(classColors.size());

    CV_Assert((int)trainClasses.total() == trainSamples.rows);
    CV_Assert((int)trainClasses.type() == CV_32SC1);

    for(size_t modelIndex = 0; modelIndex < em_models.size(); modelIndex++)
    {
        const int componentCount = 3;
        em_models[modelIndex] = EM(componentCount, cv::EM::COV_MAT_DIAGONAL);

        Mat modelSamples;
        for(int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
        {
            if(trainClasses.at<int>(sampleIndex) == (int)modelIndex)
                modelSamples.push_back(trainSamples.row(sampleIndex));
        }

        // learn models
        if(!modelSamples.empty())
            em_models[modelIndex].train(modelSamples);
    }

    // classify coordinate plane points using the bayes classifier, i.e.
    // y(x) = arg max_i=1_modelsCount likelihoods_i(x)
    Mat testSample(1, 2, CV_32FC1 );
    for( int y = 0; y < img.rows; y += testStep )
    {
        for( int x = 0; x < img.cols; x += testStep )
        {
            testSample.at<float>(0) = (float)x;
            testSample.at<float>(1) = (float)y;

            Mat logLikelihoods(1, em_models.size(), CV_64FC1, Scalar(-DBL_MAX));
            for(size_t modelIndex = 0; modelIndex < em_models.size(); modelIndex++)
            {
                if(em_models[modelIndex].isTrained())
                    em_models[modelIndex].predict( testSample, noArray(), &logLikelihoods.at<double>(modelIndex) );
            }
            Point maxLoc;
            minMaxLoc(logLikelihoods, 0, 0, 0, &maxLoc);

            int response = maxLoc.x;
            circle( imgDst, Point(x,y), 2, classColors[response], 1 );
        }
    }
}
#endif

int main()
{
    cout << "Use:" << endl
         << "  right mouse button - to add new class;" << endl
         << "  left mouse button - to add new point;" << endl
         << "  key 'r' - to run the ML model;" << endl
         << "  key 'i' - to init (clear) the data." << endl << endl;

    cv::namedWindow( "points", 1 );
    img.create( 480, 640, CV_8UC3 );
    imgDst.create( 480, 640, CV_8UC3 );

    imshow( "points", img );
    cvSetMouseCallback( "points", on_mouse );

    for(;;)
    {
        uchar key = (uchar)waitKey();

        if( key == 27 ) break;

        if( key == 'i' ) // init
        {
            img = Scalar::all(0);

            classColors.clear();
            trainedPoints.clear();
            trainedPointsMarkers.clear();

            imshow( winName, img );
        }

        if( key == 'r' ) // run
        {
#if _NBC_
            find_decision_boundary_NBC();
            cvNamedWindow( "NormalBayesClassifier", WINDOW_AUTOSIZE );
            imshow( "NormalBayesClassifier", imgDst );
#endif
#if _KNN_
            int K = 3;
            find_decision_boundary_KNN( K );
            namedWindow( "kNN", WINDOW_AUTOSIZE );
            imshow( "kNN", imgDst );

            K = 15;
            find_decision_boundary_KNN( K );
            namedWindow( "kNN2", WINDOW_AUTOSIZE );
            imshow( "kNN2", imgDst );
#endif

#if _SVM_
            //(1)-(2)separable and not sets
            CvSVMParams params;
            params.svm_type = CvSVM::C_SVC;
            params.kernel_type = CvSVM::POLY; //CvSVM::LINEAR;
            params.degree = 0.5;
            params.gamma = 1;
            params.coef0 = 1;
            params.C = 1;
            params.nu = 0.5;
            params.p = 0;
            params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);

            find_decision_boundary_SVM( params );
            namedWindow( "classificationSVM1", WINDOW_AUTOSIZE );
            imshow( "classificationSVM1", imgDst );

            params.C = 10;
            find_decision_boundary_SVM( params );
            cvNamedWindow( "classificationSVM2", WINDOW_AUTOSIZE );
            imshow( "classificationSVM2", imgDst );
#endif

#if _DT_
            find_decision_boundary_DT();
            namedWindow( "DT", WINDOW_AUTOSIZE );
            imshow( "DT", imgDst );
#endif

#if _BT_
            find_decision_boundary_BT();
            namedWindow( "BT", WINDOW_AUTOSIZE );
            imshow( "BT", imgDst);
#endif

#if _GBT_
            find_decision_boundary_GBT();
            namedWindow( "GBT", WINDOW_AUTOSIZE );
            imshow( "GBT", imgDst);
#endif

#if _RF_
            find_decision_boundary_RF();
            namedWindow( "RF", WINDOW_AUTOSIZE );
            imshow( "RF", imgDst);
#endif

#if _ERT_
            find_decision_boundary_ERT();
            namedWindow( "ERT", WINDOW_AUTOSIZE );
            imshow( "ERT", imgDst);
#endif

#if _ANN_
            Mat layer_sizes1( 1, 3, CV_32SC1 );
            layer_sizes1.at<int>(0) = 2;
            layer_sizes1.at<int>(1) = 5;
            layer_sizes1.at<int>(2) = classColors.size();
            find_decision_boundary_ANN( layer_sizes1 );
            namedWindow( "ANN", WINDOW_AUTOSIZE );
            imshow( "ANN", imgDst );
#endif

#if _EM_
            find_decision_boundary_EM();
            namedWindow( "EM", WINDOW_AUTOSIZE );
            imshow( "EM", imgDst );
#endif
        }
    }

    return 1;
}
