#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <time.h>

using namespace cv;
using namespace cv::ml;
using namespace std;

vector< float > get_svm_detector( const Ptr< SVM >& svm );
void convert_to_ml( const std::vector< Mat > & train_samples, Mat& trainData );
void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages );
void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size );
void computeHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst, bool use_flip );
void test_trained_detector( String obj_det_filename, String test_dir, String videofilename );

vector< float > get_svm_detector( const Ptr< SVM >& svm )
{
    // get the support vectors
    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction( 0, alpha, svidx );

    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );

    vector< float > hog_detector( sv.cols + 1 );
    memcpy( &hog_detector[0], sv.ptr(), sv.cols*sizeof( hog_detector[0] ) );
    hog_detector[sv.cols] = (float)-rho;
    return hog_detector;
}

/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml( const vector< Mat > & train_samples, Mat& trainData )
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    Mat tmp( 1, cols, CV_32FC1 ); ///< used for transposition if needed
    trainData = Mat( rows, cols, CV_32FC1 );

    for( size_t i = 0 ; i < train_samples.size(); ++i )
    {
        CV_Assert( train_samples[i].cols == 1 || train_samples[i].rows == 1 );

        if( train_samples[i].cols == 1 )
        {
            transpose( train_samples[i], tmp );
            tmp.copyTo( trainData.row( (int)i ) );
        }
        else if( train_samples[i].rows == 1 )
        {
            train_samples[i].copyTo( trainData.row( (int)i ) );
        }
    }
}

void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages = false )
{
    vector< String > files;
    glob( dirname, files );

    for ( size_t i = 0; i < files.size(); ++i )
    {
        Mat img = imread( files[i] ); // load the image
        if ( img.empty() )
        {
            cout << files[i] << " is invalid!" << endl; // invalid image, skip it.
            continue;
        }

        if ( showImages )
        {
            imshow( "image", img );
            waitKey( 1 );
        }
        img_lst.push_back( img );
    }
}

void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size )
{
    Rect box;
    box.width = size.width;
    box.height = size.height;

    srand( (unsigned int)time( NULL ) );

    for ( size_t i = 0; i < full_neg_lst.size(); i++ )
        if ( full_neg_lst[i].cols > box.width && full_neg_lst[i].rows > box.height )
        {
            box.x = rand() % ( full_neg_lst[i].cols - box.width );
            box.y = rand() % ( full_neg_lst[i].rows - box.height );
            Mat roi = full_neg_lst[i]( box );
            neg_lst.push_back( roi.clone() );
        }
}

void computeHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst, bool use_flip )
{
    HOGDescriptor hog;
    hog.winSize = wsize;
    Mat gray;
    vector< float > descriptors;

    for( size_t i = 0 ; i < img_lst.size(); i++ )
    {
        if ( img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height )
        {
            Rect r = Rect(( img_lst[i].cols - wsize.width ) / 2,
                          ( img_lst[i].rows - wsize.height ) / 2,
                          wsize.width,
                          wsize.height);
            cvtColor( img_lst[i](r), gray, COLOR_BGR2GRAY );
            hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );
            gradient_lst.push_back( Mat( descriptors ).clone() );
            if ( use_flip )
            {
                flip( gray, gray, 1 );
                hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );
                gradient_lst.push_back( Mat( descriptors ).clone() );
            }
        }
    }
}

void test_trained_detector( String obj_det_filename, String test_dir, String videofilename )
{
    cout << "Testing trained detector..." << endl;
    HOGDescriptor hog;
    hog.load( obj_det_filename );

    vector< String > files;
    glob( test_dir, files );

    int delay = 0;
    VideoCapture cap;

    if ( videofilename != "" )
    {
        if ( videofilename.size() == 1 && isdigit( videofilename[0] ) )
            cap.open( videofilename[0] - '0' );
        else
            cap.open( videofilename );
    }

    obj_det_filename = "testing " + obj_det_filename;
    namedWindow( obj_det_filename, WINDOW_NORMAL );

    for( size_t i=0;; i++ )
    {
        Mat img;

        if ( cap.isOpened() )
        {
            cap >> img;
            delay = 1;
        }
        else if( i < files.size() )
        {
            img = imread( files[i] );
        }

        if ( img.empty() )
        {
            return;
        }

        vector< Rect > detections;
        vector< double > foundWeights;

        hog.detectMultiScale( img, detections, foundWeights );
        for ( size_t j = 0; j < detections.size(); j++ )
        {
            Scalar color = Scalar( 0, foundWeights[j] * foundWeights[j] * 200, 0 );
            rectangle( img, detections[j], color, img.cols / 400 + 1 );
        }

        imshow( obj_det_filename, img );

        if( waitKey( delay ) == 27 )
        {
            return;
        }
    }
}

int main( int argc, char** argv )
{
    const char* keys =
    {
        "{help h|     | show help message}"
        "{pd    |     | path of directory contains positive images}"
        "{nd    |     | path of directory contains negative images}"
        "{td    |     | path of directory contains test images}"
        "{tv    |     | test video file name}"
        "{dw    |     | width of the detector}"
        "{dh    |     | height of the detector}"
        "{f     |false| indicates if the program will generate and use mirrored samples or not}"
        "{d     |false| train twice}"
        "{t     |false| test a trained detector}"
        "{v     |false| visualize training steps}"
        "{fn    |my_detector.yml| file name of trained SVM}"
    };

    CommandLineParser parser( argc, argv, keys );

    if ( parser.has( "help" ) )
    {
        parser.printMessage();
        exit( 0 );
    }

    String pos_dir = parser.get< String >( "pd" );
    String neg_dir = parser.get< String >( "nd" );
    String test_dir = parser.get< String >( "td" );
    String obj_det_filename = parser.get< String >( "fn" );
    String videofilename = parser.get< String >( "tv" );
    int detector_width = parser.get< int >( "dw" );
    int detector_height = parser.get< int >( "dh" );
    bool test_detector = parser.get< bool >( "t" );
    bool train_twice = parser.get< bool >( "d" );
    bool visualization = parser.get< bool >( "v" );
    bool flip_samples = parser.get< bool >( "f" );

    if ( test_detector )
    {
        test_trained_detector( obj_det_filename, test_dir, videofilename );
        exit( 0 );
    }

    if( pos_dir.empty() || neg_dir.empty() )
    {
        parser.printMessage();
        cout << "Wrong number of parameters.\n\n"
             << "Example command line:\n" << argv[0] << " -dw=64 -dh=128 -pd=/INRIAPerson/96X160H96/Train/pos -nd=/INRIAPerson/neg -td=/INRIAPerson/Test/pos -fn=HOGpedestrian64x128.xml -d\n"
             << "\nExample command line for testing trained detector:\n" << argv[0] << " -t -fn=HOGpedestrian64x128.xml -td=/INRIAPerson/Test/pos";
        exit( 1 );
    }

    vector< Mat > pos_lst, full_neg_lst, neg_lst, gradient_lst;
    vector< int > labels;

    clog << "Positive images are being loaded..." ;
    load_images( pos_dir, pos_lst, visualization );
    if ( pos_lst.size() > 0 )
    {
        clog << "...[done] " << pos_lst.size() << " files." << endl;
    }
    else
    {
        clog << "no image in " << pos_dir <<endl;
        return 1;
    }

    Size pos_image_size = pos_lst[0].size();

    if ( detector_width && detector_height )
    {
        pos_image_size = Size( detector_width, detector_height );
    }
    else
    {
        for ( size_t i = 0; i < pos_lst.size(); ++i )
        {
            if( pos_lst[i].size() != pos_image_size )
            {
                cout << "All positive images should be same size!" << endl;
                exit( 1 );
            }
        }
        pos_image_size = pos_image_size / 8 * 8;
    }

    clog << "Negative images are being loaded...";
    load_images( neg_dir, full_neg_lst, visualization );
    clog << "...[done] " << full_neg_lst.size() << " files." << endl;

    clog << "Negative images are being processed...";
    sample_neg( full_neg_lst, neg_lst, pos_image_size );
    clog << "...[done] " << neg_lst.size() << " files." << endl;

    clog << "Histogram of Gradients are being calculated for positive images...";
    computeHOGs( pos_image_size, pos_lst, gradient_lst, flip_samples );
    size_t positive_count = gradient_lst.size();
    labels.assign( positive_count, +1 );
    clog << "...[done] ( positive images count : " << positive_count << " )" << endl;

    clog << "Histogram of Gradients are being calculated for negative images...";
    computeHOGs( pos_image_size, neg_lst, gradient_lst, flip_samples );
    size_t negative_count = gradient_lst.size() - positive_count;
    labels.insert( labels.end(), negative_count, -1 );
    CV_Assert( positive_count < labels.size() );
    clog << "...[done] ( negative images count : " << negative_count << " )" << endl;

    Mat train_data;
    convert_to_ml( gradient_lst, train_data );

    clog << "Training SVM...";
    Ptr< SVM > svm = SVM::create();
    /* Default values to train SVM */
    svm->setCoef0( 0.0 );
    svm->setDegree( 3 );
    svm->setTermCriteria( TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3 ) );
    svm->setGamma( 0 );
    svm->setKernel( SVM::LINEAR );
    svm->setNu( 0.5 );
    svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
    svm->setC( 0.01 ); // From paper, soft classifier
    svm->setType( SVM::EPS_SVR ); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
    svm->train( train_data, ROW_SAMPLE, labels );
    clog << "...[done]" << endl;

    if ( train_twice )
    {
        clog << "Testing trained detector on negative images. This might take a few minutes...";
        HOGDescriptor my_hog;
        my_hog.winSize = pos_image_size;

        // Set the trained svm to my_hog
        my_hog.setSVMDetector( get_svm_detector( svm ) );

        vector< Rect > detections;
        vector< double > foundWeights;

        for ( size_t i = 0; i < full_neg_lst.size(); i++ )
        {
            if ( full_neg_lst[i].cols >= pos_image_size.width && full_neg_lst[i].rows >= pos_image_size.height )
                my_hog.detectMultiScale( full_neg_lst[i], detections, foundWeights );
            else
                detections.clear();

            for ( size_t j = 0; j < detections.size(); j++ )
            {
                Mat detection = full_neg_lst[i]( detections[j] ).clone();
                resize( detection, detection, pos_image_size, 0, 0, INTER_LINEAR_EXACT);
                neg_lst.push_back( detection );
            }

            if ( visualization )
            {
                for ( size_t j = 0; j < detections.size(); j++ )
                {
                    rectangle( full_neg_lst[i], detections[j], Scalar( 0, 255, 0 ), 2 );
                }
                imshow( "testing trained detector on negative images", full_neg_lst[i] );
                waitKey( 5 );
            }
        }
        clog << "...[done]" << endl;

        gradient_lst.clear();
        clog << "Histogram of Gradients are being calculated for positive images...";
        computeHOGs( pos_image_size, pos_lst, gradient_lst, flip_samples );
        positive_count = gradient_lst.size();
        clog << "...[done] ( positive count : " << positive_count << " )" << endl;

        clog << "Histogram of Gradients are being calculated for negative images...";
        computeHOGs( pos_image_size, neg_lst, gradient_lst, flip_samples );
        negative_count = gradient_lst.size() - positive_count;
        clog << "...[done] ( negative count : " << negative_count << " )" << endl;

        labels.clear();
        labels.assign(positive_count, +1);
        labels.insert(labels.end(), negative_count, -1);

        clog << "Training SVM again...";
        convert_to_ml( gradient_lst, train_data );
        svm->train( train_data, ROW_SAMPLE, labels );
        clog << "...[done]" << endl;
    }

    HOGDescriptor hog;
    hog.winSize = pos_image_size;
    hog.setSVMDetector( get_svm_detector( svm ) );
    hog.save( obj_det_filename );

    test_trained_detector( obj_det_filename, test_dir, videofilename );

    return 0;
}
