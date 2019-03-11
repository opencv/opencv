#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"

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
            "The usage: letter_recog [-data=<path to letter-recognition.data>] \\\n"
            "  [-save=<output XML file for the classifier>] \\\n"
            "  [-load=<XML file with the pre-trained classifier>] \\\n"
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
        r = std::abs(r + rdelta - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;

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
//        Params( int maxDepth, int minSampleCount,
//                   double regressionAccuracy, bool useSurrogates,
//                   int maxCategories, const Mat& priors,
//                   bool calcVarImportance, int nactiveVars,
//                   TermCriteria termCrit );
        Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);
        model = RTrees::create();
        model->setMaxDepth(10);
        model->setMinSampleCount(10);
        model->setRegressionAccuracy(0);
        model->setUseSurrogates(false);
        model->setMaxCategories(15);
        model->setPriors(Mat());
        model->setCalculateVarImportance(true);
        model->setActiveVarCount(4);
        model->setTermCriteria(TC(100,0.01f));
        model->train(tdata);
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
        vector<double> priors(2);
        priors[0] = 1;
        priors[1] = 26;

        cout << "Training the classifier (may take a few minutes)...\n";
        model = Boost::create();
        model->setBoostType(Boost::GENTLE);
        model->setWeakCount(100);
        model->setWeightTrimRate(0.95);
        model->setMaxDepth(5);
        model->setUseSurrogates(false);
        model->setPriors(Mat(priors));
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
        Mat train_responses = Mat::zeros( ntrain_samples, class_count, CV_32F );

        // 1. unroll the responses
        cout << "Unrolling the responses...\n";
        for( int i = 0; i < ntrain_samples; i++ )
        {
            int cls_label = responses.at<int>(i) - 'A';
            train_responses.at<float>(i, cls_label) = 1.f;
        }

        // 2. train classifier
        int layer_sz[] = { data.cols, 100, 100, class_count };
        int nlayers = (int)(sizeof(layer_sz)/sizeof(layer_sz[0]));
        Mat layer_sizes( 1, nlayers, CV_32S, layer_sz );

#if 1
        int method = ANN_MLP::BACKPROP;
        double method_param = 0.001;
        int max_iter = 300;
#else
        int method = ANN_MLP::RPROP;
        double method_param = 0.1;
        int max_iter = 1000;
#endif

        Ptr<TrainData> tdata = TrainData::create(train_data, ROW_SAMPLE, train_responses);

        cout << "Training the classifier (may take a few minutes)...\n";
        model = ANN_MLP::create();
        model->setLayerSizes(layer_sizes);
        model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
        model->setTermCriteria(TC(max_iter,0));
        model->setTrainMethod(method, method_param);
        model->train(tdata);
        cout << endl;
    }

    test_and_save_classifier(model, data, responses, ntrain_samples, 'A', filename_to_save);
    return true;
}

static bool
build_knearest_classifier( const string& data_filename, int K )
{
    Mat data;
    Mat responses;
    bool ok = read_num_class_data( data_filename, 16, &data, &responses );
    if( !ok )
        return ok;


    int nsamples_all = data.rows;
    int ntrain_samples = (int)(nsamples_all*0.8);

    // create classifier by using <data> and <responses>
    cout << "Training the classifier ...\n";
    Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);
    Ptr<KNearest> model = KNearest::create();
    model->setDefaultK(K);
    model->setIsClassifier(true);
    model->train(tdata);
    cout << endl;

    test_and_save_classifier(model, data, responses, ntrain_samples, 0, string());
    return true;
}

static bool
build_nbayes_classifier( const string& data_filename )
{
    Mat data;
    Mat responses;
    bool ok = read_num_class_data( data_filename, 16, &data, &responses );
    if( !ok )
        return ok;

    Ptr<NormalBayesClassifier> model;

    int nsamples_all = data.rows;
    int ntrain_samples = (int)(nsamples_all*0.8);

    // create classifier by using <data> and <responses>
    cout << "Training the classifier ...\n";
    Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);
    model = NormalBayesClassifier::create();
    model->train(tdata);
    cout << endl;

    test_and_save_classifier(model, data, responses, ntrain_samples, 0, string());
    return true;
}

static bool
build_svm_classifier( const string& data_filename,
                      const string& filename_to_save,
                      const string& filename_to_load )
{
    Mat data;
    Mat responses;
    bool ok = read_num_class_data( data_filename, 16, &data, &responses );
    if( !ok )
        return ok;

    Ptr<SVM> model;

    int nsamples_all = data.rows;
    int ntrain_samples = (int)(nsamples_all*0.8);

    // Create or load Random Trees classifier
    if( !filename_to_load.empty() )
    {
        model = load_classifier<SVM>(filename_to_load);
        if( model.empty() )
            return false;
        ntrain_samples = 0;
    }
    else
    {
        // create classifier by using <data> and <responses>
        cout << "Training the classifier ...\n";
        Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);
        model = SVM::create();
        model->setType(SVM::C_SVC);
        model->setKernel(SVM::LINEAR);
        model->setC(1);
        model->train(tdata);
        cout << endl;
    }

    test_and_save_classifier(model, data, responses, ntrain_samples, 0, filename_to_save);
    return true;
}

int main( int argc, char *argv[] )
{
    string filename_to_save = "";
    string filename_to_load = "";
    string data_filename;
    int method = 0;

    cv::CommandLineParser parser(argc, argv, "{data|letter-recognition.data|}{save||}{load||}{boost||}"
            "{mlp||}{knn knearest||}{nbayes||}{svm||}");
    data_filename = samples::findFile(parser.get<string>("data"));
    if (parser.has("save"))
        filename_to_save = parser.get<string>("save");
    if (parser.has("load"))
        filename_to_load = samples::findFile(parser.get<string>("load"));
    if (parser.has("boost"))
        method = 1;
    else if (parser.has("mlp"))
        method = 2;
    else if (parser.has("knearest"))
        method = 3;
    else if (parser.has("nbayes"))
        method = 4;
    else if (parser.has("svm"))
        method = 5;

    help();

    if( (method == 0 ?
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

    return 0;
}
