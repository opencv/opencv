#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

#include <cstdio>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;

static void help()
{
    printf("\nThe sample demonstrates how to train Random Trees classifier\n"
    "(or Boosting classifier, or MLP, or Knearest, or Nbayes, or Support Vector Machines - see main()) using the provided dataset.\n"
    "\n"
    "We use the sample database letter-recognition.data\n"
    "from UCI Repository, here is the link:\n"
    "\n"
    "Newman, D.J. & Hettich, S. & Blake, C.L. & Merz, C.J. (1998).\n"
    "UCI Repository of machine learning databases\n"
    "[http://www.ics.uci.edu/~mlearn/MLRepository.html].\n"
    "Irvine, CA: University of California, Department of Information and Computer Science.\n"
    "\n"
    "The dataset consists of 20000 feature vectors along with the\n"
    "responses - capital latin letters A..Z.\n"
    "The first 16000 (10000 for boosting)) samples are used for training\n"
    "and the remaining 4000 (10000 for boosting) - to test the classifier.\n"
    "======================================================\n");
    printf("\nThis is letter recognition sample.\n"
            "The usage: letter_recog [-data <path to letter-recognition.data>] \\\n"
            "  [-save <output XML file for the classifier>] \\\n"
            "  [-load <XML file with the pre-trained classifier>] \\\n"
            "  [-boost|-mlp|-knearest|-nbayes|-svm] # to use boost/mlp/knearest/SVM classifier instead of default Random Trees\n" );
}

// This function reads data and responses from the file <filename>
static bool
read_num_class_data( const string& filename, int var_count,
                     Mat* _data, Mat* _responses )
{
    const int M = 1024;
    char buf[M+2];

    Mat el_ptr(1, var_count, CV_32F);
    int i;
    vector<int> responses;

    _data->release();
    _responses->release();

    FILE* f = fopen( filename.c_str(), "rt" );
    if( !f )
    {
        cout << "Could not read the database " << filename << endl;
        return false;
    }

    for(;;)
    {
        char* ptr;
        if( !fgets( buf, M, f ) || !strchr( buf, ',' ) )
            break;
        responses.push_back((int)buf[0]);
        ptr = buf+2;
        for( i = 0; i < var_count; i++ )
        {
            int n = 0;
            sscanf( ptr, "%f%n", &el_ptr.at<float>(i), &n );
            ptr += n + 1;
        }
        if( i < var_count )
            break;
        _data->push_back(el_ptr);
    }
    fclose(f);
    Mat(responses).copyTo(*_responses);

    cout << "The database " << filename << " is loaded.\n";

    return true;
}

template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load)
{
    // load classifier from the specified file
    Ptr<T> model = StatModel::load<T>( filename_to_load );
    if( model.empty() )
        cout << "Could not read the classifier " << filename_to_load << endl;
    else
        cout << "The classifier " << filename_to_load << " is loaded.\n";

    return model;
}

static Ptr<TrainData>
prepare_train_data(const Mat& data, const Mat& responses, int ntrain_samples)
{
    Mat sample_idx = Mat::zeros( 1, data.rows, CV_8U );
    Mat train_samples = sample_idx.colRange(0, ntrain_samples);
    train_samples.setTo(Scalar::all(1));

    int nvars = data.cols;
    Mat var_type( nvars + 1, 1, CV_8U );
    var_type.setTo(Scalar::all(VAR_ORDERED));
    var_type.at<uchar>(nvars) = VAR_CATEGORICAL;

    return TrainData::create(data, ROW_SAMPLE, responses,
                             noArray(), sample_idx, noArray(), var_type);
}

inline TermCriteria TC(int iters, double eps)
{
    return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

static void test_and_save_classifier(const Ptr<StatModel>& model,
                                     const Mat& data, const Mat& responses,
                                     int ntrain_samples, int rdelta,
                                     const string& filename_to_save)
{
    int i, nsamples_all = data.rows;
    double train_hr = 0, test_hr = 0;

    // compute prediction error on train and test data
    for( i = 0; i < nsamples_all; i++ )
    {
        Mat sample = data.row(i);

        float r = model->predict( sample );
        r = std::abs(r + rdelta - responses.at<int>(i)) <= FLT_EPSILON ? 1 : 0;

        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= nsamples_all - ntrain_samples;
    train_hr = ntrain_samples > 0 ? train_hr/ntrain_samples : 1.;

    printf( "Recognition rate: train = %.1f%%, test = %.1f%%\n",
            train_hr*100., test_hr*100. );

    if( !filename_to_save.empty() )
    {
        model->save( filename_to_save );
    }
}


static bool
build_rtrees_classifier( const string& data_filename,
                         const string& filename_to_save,
                         const string& filename_to_load )
{
    Mat data;
    Mat responses;
    bool ok = read_num_class_data( data_filename, 16, &data, &responses );
    if( !ok )
        return ok;

    Ptr<RTrees> model;

    int nsamples_all = data.rows;
    int ntrain_samples = (int)(nsamples_all*0.8);

    // Create or load Random Trees classifier
    if( !filename_to_load.empty() )
    {
        model = load_classifier<RTrees>(filename_to_load);
        if( model.empty() )
            return false;
        ntrain_samples = 0;
    }
    else
    {
        // create classifier by using <data> and <responses>
        cout << "Training the classifier ...\n";
        Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);

        // 3. train classifier
        model = RTrees::create(RTrees::Params(10,10,0,false,15,Mat(),true,4,TC(100,0.01f)));
        model->train( tdata );
        cout << endl;
    }

    test_and_save_classifier(model, data, responses, ntrain_samples, 0, filename_to_save);
    cout << "Number of trees: " << model->getRoots().size() << endl;

    // Print variable importance
    Mat var_importance = model->getVarImportance();
    if( !var_importance.empty() )
    {
        double rt_imp_sum = sum( var_importance )[0];
        printf("var#\timportance (in %%):\n");
        int i, n = (int)var_importance.total();
        for( i = 0; i < n; i++ )
            printf( "%-2d\t%-4.1f\n", i, 100.f*var_importance.at<float>(i)/rt_imp_sum);
    }

    return true;
}


static bool
build_boost_classifier( const string& data_filename,
                        const string& filename_to_save,
                        const string& filename_to_load )
{
    const int class_count = 26;
    Mat data;
    Mat responses;
    Mat weak_responses;

    bool ok = read_num_class_data( data_filename, 16, &data, &responses );
    if( !ok )
        return ok;

    int i, j, k;
    Ptr<Boost> model;

    int nsamples_all = data.rows;
    int ntrain_samples = (int)(nsamples_all*0.5);
    int var_count = data.cols;

    // Create or load Boosted Tree classifier
    if( !filename_to_load.empty() )
    {
        model = load_classifier<Boost>(filename_to_load);
        if( model.empty() )
            return false;
        ntrain_samples = 0;
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
        printf( "Unrolling the database...\n");
        for( i = 0; i < ntrain_samples; i++ )
        {
            const float* data_row = data.ptr<float>(i);
            for( j = 0; j < class_count; j++ )
            {
                float* new_data_row = (float*)new_data.ptr<float>(i*class_count+j);
                memcpy(new_data_row, data_row, var_count*sizeof(data_row[0]));
                new_data_row[var_count] = (float)j;
                new_responses.at<int>(i*class_count + j) = responses.at<int>(i) == j+'A';
            }
        }

        Mat var_type( 1, var_count + 2, CV_8U );
        var_type.setTo(Scalar::all(VAR_ORDERED));
        var_type.at<uchar>(var_count) = var_type.at<uchar>(var_count+1) = VAR_CATEGORICAL;

        Ptr<TrainData> tdata = TrainData::create(new_data, ROW_SAMPLE, new_responses,
                                                 noArray(), noArray(), noArray(), var_type);
        model = Boost::create(Boost::Params(Boost::REAL, 100, 0.95, 5, false, Mat() ));

        cout << "Training the classifier (may take a few minutes)...\n";
        model->train(tdata);
        cout << endl;
    }

    Mat temp_sample( 1, var_count + 1, CV_32F );
    float* tptr = temp_sample.ptr<float>();

    // compute prediction error on train and test data
    double train_hr = 0, test_hr = 0;
    for( i = 0; i < nsamples_all; i++ )
    {
        int best_class = 0;
        double max_sum = -DBL_MAX;
        const float* ptr = data.ptr<float>(i);
        for( k = 0; k < var_count; k++ )
            tptr[k] = ptr[k];

        for( j = 0; j < class_count; j++ )
        {
            tptr[var_count] = (float)j;
            float s = model->predict( temp_sample, noArray(), StatModel::RAW_OUTPUT );
            if( max_sum < s )
            {
                max_sum = s;
                best_class = j + 'A';
            }
        }

        double r = std::abs(best_class - responses.at<int>(i)) < FLT_EPSILON ? 1 : 0;
        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= nsamples_all-ntrain_samples;
    train_hr = ntrain_samples > 0 ? train_hr/ntrain_samples : 1.;
    printf( "Recognition rate: train = %.1f%%, test = %.1f%%\n",
            train_hr*100., test_hr*100. );

    cout << "Number of trees: " << model->getRoots().size() << endl;

    // Save classifier to file if needed
    if( !filename_to_save.empty() )
        model->save( filename_to_save );

    return true;
}


static bool
build_mlp_classifier( const string& data_filename,
                      const string& filename_to_save,
                      const string& filename_to_load )
{
    const int class_count = 26;
    Mat data;
    Mat responses;

    bool ok = read_num_class_data( data_filename, 16, &data, &responses );
    if( !ok )
        return ok;

    int i, j;
    Ptr<ANN_MLP> model;

    int nsamples_all = data.rows;
    int ntrain_samples = (int)(nsamples_all*0.8);

    // Create or load MLP classifier
    if( !filename_to_load.empty() )
    {
        model = load_classifier<ANN_MLP>(filename_to_load);
        if( model.empty() )
            return false;
        ntrain_samples = 0;
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

        Mat train_data = data.rowRange(0, ntrain_samples);
        Mat new_responses = Mat::zeros( ntrain_samples, class_count, CV_32F );

        // 1. unroll the responses
        cout << "Unrolling the responses...\n";
        for( i = 0; i < ntrain_samples; i++ )
        {
            int cls_label = responses.at<int>(i) - 'A'
            new_responses.at<float>(i, cls_label) = 1.f;
        }

        // 2. train classifier
        int layer_sz[] = { data.cols, 100, 100, class_count };
        int nlayers = (int)(sizeof(layer_sz)/sizeof(layer_sz[0]));
        Mat layer_sizes( 1, nlayers, CV_32S, layer_sz );

#if 1
        int method = ANN_MLP::Params::BACKPROP;
        double method_param = 0.001;
        int max_iter = 300;
#else
        int method = ANN_MLP::Params::RPROP;
        double method_param = 0.1;
        int max_iter = 1000;
#endif

        mlp.train( &train_data, new_responses, 0, 0,
                  ANN_MLP::Params(TC(max_iter,0), method, method_param));


        model = ANN_MLP::create() mlp.create( &layer_sizes );
        printf( "Training the classifier (may take a few minutes)...\n");

        cvReleaseMat( &new_responses );
        printf("\n");
    }

    Mat mlp_response;

    // compute prediction error on train and test data
    for( i = 0; i < nsamples_all; i++ )
    {
        int best_class;
        CvMat sample;
        cvGetRow( data, &sample, i );
        CvPoint max_loc;
        mlp.predict( &sample, mlp_response );
        cvMinMaxLoc( mlp_response, 0, 0, 0, &max_loc, 0 );
        best_class = max_loc.x + 'A';

        int r = fabs((double)best_class - responses->data.fl[i]) < FLT_EPSILON ? 1 : 0;

        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= (double)(nsamples_all-ntrain_samples);
    train_hr /= (double)ntrain_samples;
    printf( "Recognition rate: train = %.1f%%, test = %.1f%%\n",
            train_hr*100., test_hr*100. );

    if( !filename_to_save.empty() )
        model->save( filename_to_save );

    return true;
}

static bool
build_knearest_classifier( const string& data_filename, int K )
{
    const int var_count = 16;
    Mat data;
    CvMat train_data;
    Mat responses;

    bool ok = read_num_class_data( data_filename, 16, &data, &responses );
    if( !ok )
        return ok;

    int nsamples_all = 0, ntrain_samples = 0;

    nsamples_all = data->rows;
    ntrain_samples = (int)(nsamples_all*0.8);

    // 1. unroll the responses
    printf( "Unrolling the responses...\n");
    cvGetRows( data, &train_data, 0, ntrain_samples );

    // 2. train classifier
    Mat train_resp = cvCreateMat( ntrain_samples, 1, CV_32FC1);
    for (int i = 0; i < ntrain_samples; i++)
        train_resp->data.fl[i] = responses->data.fl[i];
    Ptr<KNearest> model = KNearest::create(true);
    model->train(train_data, train_resp);

    Mat nearests = cvCreateMat( (nsamples_all - ntrain_samples), K, CV_32FC1);
    float* _sample = new float[var_count * (nsamples_all - ntrain_samples)];
    CvMat sample = cvMat( nsamples_all - ntrain_samples, 16, CV_32FC1, _sample );
    float* true_results = new float[nsamples_all - ntrain_samples];
    for (int j = ntrain_samples; j < nsamples_all; j++)
    {
        float *s = data->data.fl + j * var_count;

        for (int i = 0; i < var_count; i++)
        {
            sample.data.fl[(j - ntrain_samples) * var_count + i] = s[i];
        }
        true_results[j - ntrain_samples] = responses->data.fl[j];
    }
    CvMat *result = cvCreateMat(1, nsamples_all - ntrain_samples, CV_32FC1);
    knearest.find_nearest(&sample, K, result, 0, nearests, 0);
    int true_resp = 0;
    int accuracy = 0;
    for (int i = 0; i < nsamples_all - ntrain_samples; i++)
    {
        if (result->data.fl[i] == true_results[i])
            true_resp++;
        for(int k = 0; k < K; k++ )
        {
            if( nearests->data.fl[i * K + k] == true_results[i])
            accuracy++;
        }
    }

    printf("true_resp = %f%%\tavg accuracy = %f%%\n", (float)true_resp / (nsamples_all - ntrain_samples) * 100,
                                                      (float)accuracy / (nsamples_all - ntrain_samples) / K * 100);

    delete[] true_results;
    delete[] _sample;
    cvReleaseMat( &train_resp );
    cvReleaseMat( &nearests );
    cvReleaseMat( &result );
    cvReleaseMat( &data );
    cvReleaseMat( &responses );

    return 0;
}

static bool
build_nbayes_classifier( const string& data_filename )
{
    const int var_count = 16;
    Mat data;
    CvMat train_data;
    Mat responses;

    bool ok = read_num_class_data( data_filename, 16, &data, &responses );
    if( !ok )
        return ok;

    int nsamples_all = 0, ntrain_samples = 0;

    nsamples_all = data->rows;
    ntrain_samples = (int)(nsamples_all*0.5);

    // 1. unroll the responses
    printf( "Unrolling the responses...\n");
    cvGetRows( data, &train_data, 0, ntrain_samples );

    // 2. train classifier
    Mat train_resp = cvCreateMat( ntrain_samples, 1, CV_32FC1);
    for (int i = 0; i < ntrain_samples; i++)
        train_resp->data.fl[i] = responses->data.fl[i];
    CvNormalBayesClassifier nbayes(&train_data, train_resp);

    float* _sample = new float[var_count * (nsamples_all - ntrain_samples)];
    CvMat sample = cvMat( nsamples_all - ntrain_samples, 16, CV_32FC1, _sample );
    float* true_results = new float[nsamples_all - ntrain_samples];
    for (int j = ntrain_samples; j < nsamples_all; j++)
    {
        float *s = data->data.fl + j * var_count;

        for (int i = 0; i < var_count; i++)
        {
            sample.data.fl[(j - ntrain_samples) * var_count + i] = s[i];
        }
        true_results[j - ntrain_samples] = responses->data.fl[j];
    }
    CvMat *result = cvCreateMat(1, nsamples_all - ntrain_samples, CV_32FC1);
    nbayes.predict(&sample, result);
    int true_resp = 0;
    //int accuracy = 0;
    for (int i = 0; i < nsamples_all - ntrain_samples; i++)
    {
        if (result->data.fl[i] == true_results[i])
            true_resp++;
    }

    printf("true_resp = %f%%\n", (float)true_resp / (nsamples_all - ntrain_samples) * 100);

    delete[] true_results;
    delete[] _sample;
    cvReleaseMat( &train_resp );
    cvReleaseMat( &result );
    cvReleaseMat( &data );
    cvReleaseMat( &responses );

    return 0;
}

static bool
build_svm_classifier( const string& data_filename,
                      const string& filename_to_save,
                      const string& filename_to_load )
{
    Mat data;
    Mat responses;
    Mat train_resp;
    CvMat train_data;
    int nsamples_all = 0, ntrain_samples = 0;
    int var_count;
    Ptr<SVM> model;

    bool ok = read_num_class_data( data_filename, 16, &data, &responses );
    if( !ok )
        return ok;

    ////////// SVM parameters ///////////////////////////////
    CvSVMParams param;
    param.kernel_type=CvSVM::LINEAR;
    param.svm_type=CvSVM::C_SVC;
    param.C=1;
    ///////////////////////////////////////////////////////////

    printf( "The database %s is loaded.\n", data_filename );
    nsamples_all = data->rows;
    ntrain_samples = (int)(nsamples_all*0.1);
    var_count = data->cols;

    // Create or load Random Trees classifier
    if( filename_to_load )
    {
        // load classifier from the specified file
        svm.load( filename_to_load );
        ntrain_samples = 0;
        if( svm.get_var_count() == 0 )
        {
            printf( "Could not read the classifier %s\n", filename_to_load );
            return -1;
        }
        printf( "The classifier %s is loaded.\n", filename_to_load );
    }
    else
    {
        // train classifier
        printf( "Training the classifier (may take a few minutes)...\n");
        cvGetRows( data, &train_data, 0, ntrain_samples );
        train_resp = cvCreateMat( ntrain_samples, 1, CV_32FC1);
        for (int i = 0; i < ntrain_samples; i++)
            train_resp->data.fl[i] = responses->data.fl[i];
        svm.train(&train_data, train_resp, 0, 0, param);
    }

    // classification
    std::vector<float> _sample(var_count * (nsamples_all - ntrain_samples));
    CvMat sample = cvMat( nsamples_all - ntrain_samples, 16, CV_32FC1, &_sample[0] );
    std::vector<float> true_results(nsamples_all - ntrain_samples);
    for (int j = ntrain_samples; j < nsamples_all; j++)
    {
        float *s = data->data.fl + j * var_count;

        for (int i = 0; i < var_count; i++)
        {
            sample.data.fl[(j - ntrain_samples) * var_count + i] = s[i];
        }
        true_results[j - ntrain_samples] = responses->data.fl[j];
    }
    CvMat *result = cvCreateMat(1, nsamples_all - ntrain_samples, CV_32FC1);

    printf("Classification (may take a few minutes)...\n");
    double t = (double)cvGetTickCount();
    svm.predict(&sample, result);
    t = (double)cvGetTickCount() - t;
    printf("Prediction type: %gms\n", t/(cvGetTickFrequency()*1000.));

    int true_resp = 0;
    for (int i = 0; i < nsamples_all - ntrain_samples; i++)
    {
        if (result->data.fl[i] == true_results[i])
            true_resp++;
    }

    printf("true_resp = %f%%\n", (float)true_resp / (nsamples_all - ntrain_samples) * 100);

    if( !filename_to_save.empty() )
        model->save( filename_to_save );

    return true;
}

int main( int argc, char *argv[] )
{
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
        else if ( strcmp(argv[i], "-knearest") == 0)
        {
            method = 3;
        }
        else if ( strcmp(argv[i], "-nbayes") == 0)
        {
            method = 4;
        }
        else if ( strcmp(argv[i], "-svm") == 0)
        {
            method = 5;
        }
        else
            break;
    }

    if( i < argc ||
        (method == 0 ?
        build_rtrees_classifier( data_filename, filename_to_save, filename_to_load ) :
        method == 1 ?
        build_boost_classifier( data_filename, filename_to_save, filename_to_load ) :
        method == 2 ?
        build_mlp_classifier( data_filename, filename_to_save, filename_to_load ) :
        method == 3 ?
        build_knearest_classifier( data_filename, 10 ) :
        method == 4 ?
        build_nbayes_classifier( data_filename) :
        method == 5 ?
        build_svm_classifier( data_filename, filename_to_save, filename_to_load ):
        -1) < 0)
    {
        help();
    }
    return 0;
}
