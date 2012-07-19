// From OpenCV's samples/c directory
//   Example 13-3. Training snippet for boosted classifiers
//

/* License:
   July 20, 2011
   Standard BSD

   BOOK: It would be nice if you cited it:
   Learning OpenCV 2: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media
 
   AVAILABLE AT: 
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130    

   Main OpenCV site
   http://opencv.willowgarage.com/wiki/
   * An active user group is at:
     http://tech.groups.yahoo.com/group/OpenCV/
   * The minutes of weekly OpenCV development meetings are at:
     http://pr.willowgarage.com/wiki/OpenCV
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

/*
The sample demonstrates how to train Random Trees classifier
(or Boosting classifier, or MLP - see main()) using the provided dataset.

We use the sample database letter-recognition.data
from UCI Repository, here is the link:

Newman, D.J. & Hettich, S. & Blake, C.L. & Merz, C.J. (1998).
UCI Repository of machine learning databases
[http://www.ics.uci.edu/~mlearn/MLRepository.html].
Irvine, CA: University of California, Department of Information and Computer Science.

The dataset consists of 20000 feature vectors along with the
responses - capital latin letters A..Z.
The first 16000 (10000 for boosting)) samples are used for training
and the remaining 4000 (10000 for boosting) - to test the classifier.
*/

void help()
{
    cout << "This is letter recognition sample.\n"
    "Usage: ./ch13_ex13_3 [-data <path to letter-recognition.data>] \\\n"
    "  [-save <output XML file for the classifier>] \\\n"
    "  [-load <XML file with the pre-trained classifier>] \\\n"
    "  [-boost|-mlp] # to use boost/mlp classifier instead of default Random Trees\n"
    "    where [] means optional\n" << endl;
}

// This function reads data and responses from the file <filename>
static int
read_num_class_data( const char* filename, int var_count,
                     Mat& data, Mat& responses )
{
    const int M = 1024;
    FILE* f = fopen( filename, "rt" );
    char buf[M+2];
    int i;

    if( !f )
        return false;

    vector<float> rowdata(var_count+1);
    vector<float> alldata;

    for(;;)
    {
        char* ptr;
        if( !fgets( buf, M, f ) || !strchr( buf, ',' ) )
            break;
        rowdata[0] = buf[0];
        ptr = buf+2;
        for( i = 1; i <= var_count; i++ )
        {
            int n = 0;
            sscanf( ptr, "%f%n", &rowdata[i], &n );
            ptr += n + 1;
        }
        if( i <= var_count )
            break;
        copy(rowdata.begin(), rowdata.end(), back_inserter(alldata));
    }
    fclose(f);

    Mat alldatam((int)alldata.size()/rowdata.size(), (int)rowdata.size(), CV_32F, &alldata[0]);
    alldatam.colRange(1, alldatam.cols).copyTo(data);
    alldatam.col(0).copyTo(responses);
    
    return true;
}

static
int build_rtrees_classifier( Mat& data, Mat& responses,
    char* filename_to_save, char* filename_to_load )
{
    RandomTrees forest;
    
    int i, nsamples_all = data.rows;
    int ntrain_samples = (int)(nsamples_all*0.8);

    // Create or load Random Trees classifier
    if( filename_to_load )
    {
        // load classifier from the specified file
        forest.load( filename_to_load );
        ntrain_samples = 0;
        if( forest.get_tree_count() == 0 )
        {
            cout << "Could not read the classifier " << filename_to_load << endl;
            return -1;
        }
        cout << "The classifier " << filename_to_load << " is loaded.\n";
    }
    else
    {
        // create classifier by using <data> and <responses>
        cout << "Training the classifier ...";
        cout.flush();

        // 1. create type mask
        Mat var_type( data.cols + 1, 1, CV_8U );
        var_type = Scalar::all(CV_VAR_ORDERED);
        var_type.at<uchar>(data.cols) = CV_VAR_CATEGORICAL;

        // 2. create sample_idx
        Mat sample_idx( 1, nsamples_all, CV_8UC1 );
        Mat mat = sample_idx.colRange(0, ntrain_samples);
        mat = Scalar::all(1);
        mat = sample_idx.colRange(ntrain_samples, nsamples_all);
        mat = Scalar::all(0);

        // 3. train classifier
        forest.train( data, CV_ROW_SAMPLE, responses, Mat(), sample_idx, var_type, Mat(),
            RandomTreeParams(10,10,0,false,15,0,true,4,100,0.01f,CV_TERMCRIT_ITER));
        cout << "\nDone\n";
    }

    // compute prediction error on train and test data
    double train_hr = 0, test_hr = 0;
    for( i = 0; i < nsamples_all; i++ )
    {
        Mat sample = data.row(i);

        double r = forest.predict( sample );
        r = fabs((double)r - responses.at<float>(i)) <= FLT_EPSILON ? 1 : 0;

        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= (double)(nsamples_all-ntrain_samples);
    train_hr /= (double)ntrain_samples;
    cout << "Recognition rate: train = " << train_hr*100 << "%, " <<
            "test = " << test_hr*100. << "%\n";

    cout << "Number of trees: " << forest.get_tree_count() << endl;

    // Print variable importance
    Mat var_importance(forest.get_var_importance());
    if( !var_importance.empty() )
    {
        double rt_imp_sum = sum( var_importance )[0];
        printf("var importance (in %%):\n");
        for( i = 0; i < var_importance.cols; i++ )
            printf( "%-2d\t%-4.1f\n", i,
                   100.f*var_importance.at<float>(i)/rt_imp_sum);
    }

    //Print some proximitites
    cout << "Proximities between some samples corresponding to the letter 'T':\n";
    const int pairs[][2] = {{0,103}, {0,106}, {106,103}, {-1,-1}};

    for( i = 0; pairs[i][0] >= 0; i++ )
    {
        Mat sample1 = data.row(pairs[i][0]);
        Mat sample2 = data.row(pairs[i][1]);
        CvMat c_sample1 = sample1, c_sample2 = sample2;
        printf( "proximity(%d,%d) = %.1f%%\n", pairs[i][0], pairs[i][1],
            forest.get_proximity( &c_sample1, &c_sample2 )*100. );
    }

    // Save Random Trees classifier to file if needed
    if( filename_to_save )
        forest.save( filename_to_save );

    return 0;
}


static
int build_boost_classifier( Mat& data, Mat& responses,
    char* filename_to_save, char* filename_to_load )
{
    const int class_count = 26;
    int j, k;    
    Boost boost;

    int i, nsamples_all = data.rows;
    int ntrain_samples = (int)(nsamples_all*0.8);
    int var_count = data.cols;

    // Create or load Boosted Tree classifier
    if( filename_to_load )
    {
        // load classifier from the specified file
        boost.load( filename_to_load );
        ntrain_samples = 0;
        if( !boost.get_weak_predictors() )
        {
            cout << "Could not read the classifier " << filename_to_load << endl;
            return -1;
        }
        cout << "The classifier " << filename_to_load << " is loaded.\n";
    }
    else
    {
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //
        // As currently boosted tree classifier in MLL can only be trained
        // for 2-class problems, we transform the training database by
        // "unrolling" each training sample as many times as the number of
        // classes (26) that we have.
        //
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        Mat new_data( ntrain_samples*class_count, var_count + 1, CV_32F );
        Mat new_responses( ntrain_samples*class_count, 1, CV_32S );

        // 1. unroll the database type mask
        cout << "Unrolling the database...\n";
        for( i = 0; i < ntrain_samples; i++ )
        {
            const float* data_row = (float*)data.ptr<float>(i);
            for( j = 0; j < class_count; j++ )
            {
                float* new_data_row = (float*)new_data.ptr<float>(i*class_count+j);
                for( k = 0; k < var_count; k++ )
                    new_data_row[k] = data_row[k];
                new_data_row[var_count] = (float)j;
                new_responses.at<int>(i*class_count + j) = responses.at<float>(i) == j+'A';
            }
        }

        // 2. create type mask
        Mat var_type( var_count + 2, 1, CV_8U );
        var_type = Scalar::all(CV_VAR_ORDERED);
        // the last indicator variable, as well
        // as the new (binary) response are categorical
        var_type.at<uchar>(var_count) = var_type.at<uchar>(var_count+1) = CV_VAR_CATEGORICAL;

        // 3. train classifier
        cout << "Training the classifier (may take a few minutes)...";
        cout.flush();
        boost.train( new_data, CV_ROW_SAMPLE, new_responses, Mat(), Mat(), var_type, Mat(),
            BoostParams(CvBoost::REAL, 100, 0.95, 5, false, 0 ));
        cout << "\nDone\n";
    }

    Mat temp_sample( 1, var_count + 1, CV_32F );
    CvMat c_temp_sample = temp_sample;
    Mat weak_responses( 1, boost.get_weak_predictors()->total, CV_32F );
    CvMat c_weak_responses = weak_responses;

    // compute prediction error on train and test data
    double train_hr = 0, test_hr = 0;
    for( i = 0; i < nsamples_all; i++ )
    {
        int best_class = 0;
        double max_sum = -DBL_MAX;
        double r;
        Mat sample = data.row(i);
        
        for( k = 0; k < var_count; k++ )
            temp_sample.at<float>(k) = sample.at<float>(k);

        for( j = 0; j < class_count; j++ )
        {
            temp_sample.at<float>(var_count) = (float)j;
            
            boost.predict( &c_temp_sample, 0, &c_weak_responses );
            double s = sum( weak_responses )[0];
            if( max_sum < s )
            {
                max_sum = s;
                best_class = j + 'A';
            }
        }

        r = fabs(best_class - responses.at<float>(i)) < FLT_EPSILON ? 1 : 0;

        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= (double)(nsamples_all-ntrain_samples);
    train_hr /= (double)ntrain_samples;
    printf( "Recognition rate: train = %.1f%%, test = %.1f%%\n",
            train_hr*100., test_hr*100. );

    cout << "Number of trees: " << boost.get_weak_predictors()->total << endl;

    // Save classifier to file if needed
    if( filename_to_save )
        boost.save( filename_to_save );

    return 0;
}


static
int build_mlp_classifier( Mat& data, Mat& responses,
    char* filename_to_save, char* filename_to_load )
{
    const int class_count = 26;
    CvANN_MLP mlp;

    int i, nsamples_all = data.rows;
    int ntrain_samples = (int)(nsamples_all*0.8);

    // Create or load MLP classifier
    if( filename_to_load )
    {
        // load classifier from the specified file
        mlp.load( filename_to_load );
        ntrain_samples = 0;
        if( !mlp.get_layer_count() )
        {
            cout << "Could not read the classifier " << filename_to_load << endl;
            return -1;
        }
        cout << "The classifier " << filename_to_load << " is loaded.\n";
    }
    else
    {
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //
        // MLP does not support categorical variables by explicitly.
        // So, instead of the output class label, we will use
        // a binary vector of <class_count> components for training and,
        // therefore, MLP will give us a vector of "probabilities" at the
        // prediction stage
        //
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        Mat new_responses( ntrain_samples, class_count, CV_32F, Scalar::all(0) );

        // 1. unroll the responses
        cout << "Unrolling the responses...\n";
        for( i = 0; i < ntrain_samples; i++ )
        {
            int cls_label = cvRound(responses.at<float>(i)) - 'A';
            new_responses.at<float>(i, cls_label) = 1.f;
        }
        Mat train_data = data.rowRange(0, ntrain_samples);

        // 2. train classifier
        int layer_sz[] = { data.cols, 100, 100, class_count };
        Mat layer_sizes( 1, (int)(sizeof(layer_sz)/sizeof(layer_sz[0])), CV_32S, layer_sz );
        mlp.create( layer_sizes );
        cout << "Training the classifier (may take a few minutes)...";
        cout.flush();
        mlp.train( train_data, new_responses, Mat(), Mat(),
            ANN_MLP_TrainParams(TermCriteria(CV_TERMCRIT_ITER,300,0.01),
            ANN_MLP_TrainParams::RPROP,0.01));
        cout << "\nDone\n";
    }

    Mat mlp_response( 1, class_count, CV_32F );

    // compute prediction error on train and test data
    double train_hr = 0, test_hr = 0;
    for( i = 0; i < nsamples_all; i++ )
    {
        int best_class;
        Mat sample = data.row(i);
        mlp.predict( sample, mlp_response );
        Point max_loc;
        minMaxLoc( mlp_response, 0, 0, 0, &max_loc );
        best_class = max_loc.x + 'A';

        int r = fabs((double)best_class - responses.at<float>(i)) < FLT_EPSILON ? 1 : 0;

        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= (double)(nsamples_all-ntrain_samples);
    train_hr /= (double)ntrain_samples;
    printf( "Recognition rate: train = %.1f%%, test = %.1f%%\n",
            train_hr*100., test_hr*100. );

    // Save classifier to file if needed
    if( filename_to_save )
        mlp.save( filename_to_save );

    return 0;
}


int main( int argc, char *argv[] )
{
	help();
    char* filename_to_save = 0;
    char* filename_to_load = 0;
    char default_data_filename[] = "./letter-recognition.data";
    char* data_filename = default_data_filename;
    int method = 0;

    int i;
    for( i = 1; i < argc; i++ )
    {
        if( strcmp(argv[i],"-data") == 0 ) // flag "-data letter_recognition.xml"
        {
            i++;
            data_filename = argv[i];
        }
        else if( strcmp(argv[i],"-save") == 0 ) // flag "-save filename.xml"
        {
            i++;
            filename_to_save = argv[i];
        }
        else if( strcmp(argv[i],"-load") == 0) // flag "-load filename.xml"
        {
            i++;
            filename_to_load = argv[i];
        }
        else if( strcmp(argv[i],"-boost") == 0)
        {
            method = 1;
        }
        else if( strcmp(argv[i],"-mlp") == 0 )
        {
            method = 2;
        }
        else
        {
            cout << "invalid option " << argv[i] << endl;
            help();
            return -1;
        }
    }

    Mat data, responses;
    bool ok = read_num_class_data( data_filename, 16, data, responses );
    RandomTrees forest;
    
    if( !ok )
    {
        cout << "Could not read the database " << data_filename << endl;
        help();
        return -1;
    }
    
    cout << "The database " << data_filename << " is loaded.\n";
    
    if( method == 0 )
        build_rtrees_classifier( data, responses, filename_to_save, filename_to_load );
    else if( method == 1 )
        build_boost_classifier( data, responses, filename_to_save, filename_to_load );
    else if( method == 2 )
        build_mlp_classifier( data, responses, filename_to_save, filename_to_load );
    else
        assert(0);
    return 0;
}
