#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"
#include "opencv2/ocl/ocl.hpp"

#include <cstdio>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;

static void help()
{
    std::cout << "\nThe sample demonstrates how to train Knearest classifier\n"
              << "(or Support Vector Machines - see main()) using the provided dataset.\n"
              << "\n"
              << "We use the sample database letter-recognition.data\n"
              << "from UCI Repository, here is the link:\n"
              << "\n"
              << "Newman, D.J. & Hettich, S. & Blake, C.L. & Merz, C.J. (1998).\n"
              << "UCI Repository of machine learning databases\n"
              << "[http://www.ics.uci.edu/~mlearn/MLRepository.html].\n"
              << "Irvine, CA: University of California, Department of Information and Computer Science.\n"
              << "\n"
              << "The dataset consists of 20000 feature vectors along with the\n"
              << "responses - capital latin letters A..Z.\n"
              << "The first 16000 (10000 for boosting)) samples are used for training\n"
              << "and the remaining 4000 (10000 for boosting) - to test the classifier.\n"
              << "======================================================\n";
    std::cout << "\nThis is letter recognition sample.\n"
              << "The usage: letter_recog [-data=path to letter-recognition.data] \\\n"
              << "  [-save=<output XML file for the classifier>] \\\n"
              << "  [-load=<XML file with the pre-trained classifier>] \\\n"
              << "  [knn|svm] # to use knearest/SVM classifier instead of default Random Trees\n" ;
}

// This function reads data and responses from the file <filename>
static int
read_num_class_data( cv::String filename, int var_count,
                     cv::Mat& data, cv::Mat& responses )
{
    const int M = 1024;
    ifstream f(filename.c_str());
    MemStorage storage;

    storage = cvCreateMemStorage();
    Seq<float> seq(storage);
    char buf[M + 2];
    float* el_ptr;
    Seq<float>::iterator reader;
    int i, j;

    if( !f )
    {
        return 0;
    }

    el_ptr = new float[var_count + 1];

    seq.seq = cvCreateSeq( 0, sizeof(*(seq.seq)), (var_count + 1) * sizeof(float), storage );

    for(;;)
    {
        char* ptr;
        if( !f.getline( buf, M ) || !strchr( buf, ',' ) )
        {
            break;
        }
        el_ptr[0] = buf[0];
        ptr = buf + 2;
        for( i = 1; i <= var_count; i++ )
        {
            int n = 0;
            sscanf( ptr, "%f%n", el_ptr + i, &n );
            ptr += n + 1;
        }
        if( i <= var_count )
        {
            break;
        }
        seq.push_back(*el_ptr);
    }
    f.close();

    data = cv::Mat( static_cast<int>(seq.size()), var_count, CV_32F );
    responses = cv::Mat( static_cast<int>(seq.size()), 1, CV_32F );

    reader = seq.begin();
    for( i = 0; i < static_cast<int>(seq.size()); i++ )
    {
        const float* sdata = (float*)reader.ptr + 1;
        float* ddata = (float*) data.data + var_count * i;
        float* dr = (float*) responses.data + i;

        for( j = 0; j < var_count; j++ )
        {
            ddata[j] = sdata[j];
        }
        *dr = sdata[-1];
        CV_NEXT_SEQ_ELEM( seq.elemSize(), reader );
    }

    delete[] el_ptr;
    return 1;
}

static
int build_knearest_classifier( cv::String data_filename, int K )
{
    const int var_count = 16;
    cv::Mat data;
    cv::Mat train_data;
    cv::Mat responses;

    int ok = read_num_class_data( data_filename, 16, data, responses );
    int nsamples_all = 0, ntrain_samples = 0;

    if( !ok )
    {
        std::cout << "Could not read the database " << data_filename << std::endl;
        return -1;
    }

    std::cout << "The database " << data_filename << " is loaded." << std::endl;
    nsamples_all = data.rows;
    ntrain_samples = (int)(nsamples_all * 0.1);

    // 1. unroll the responses
    std::cout << "Unrolling the responses...\n";
    train_data = data(cv::Range(0, ntrain_samples), cv::Range::all());

    // 2. train classifier
    cv::Mat train_resp = cv::Mat( ntrain_samples, 1, CV_32FC1);
    for (int i = 0; i < ntrain_samples; i++)
    {
        train_resp.at<float>(i, 0) = responses.at<float>(i, 0);
    }

    cv::ocl::KNearestNeighbour knearest;

    float* _sample = new float[var_count * (nsamples_all - ntrain_samples)];
    cv::Mat sample = cv::Mat( nsamples_all - ntrain_samples, 16, CV_32FC1, _sample );
    float* true_results = new float[nsamples_all - ntrain_samples];

    for (int j = ntrain_samples; j < nsamples_all; j++)
    {
        float *s = (float*) data.data + j * var_count;

        for (int i = 0; i < var_count; i++)
        {
            sample.at<float>((j - ntrain_samples), i) = s[i];
        }
        true_results[j - ntrain_samples] = responses.at<float>(j, 0);
    }

    cv::ocl::oclMat result_ocl, sample_ocl;
    sample_ocl.upload(sample);

    knearest.train(train_data, train_resp);
    knearest.find_nearest(sample_ocl, K, result_ocl);

    cv::Mat result;
    result_ocl.download(result);
    int true_resp = 0;

    for (int i = 0; i < nsamples_all - ntrain_samples; i++)
    {
        if (result.at<float>(i, 0) == true_results[i])
        {
            true_resp++;
        }
    }

    cout << "true_resp = " << (float)true_resp / (nsamples_all - ntrain_samples) * 100 << "%" << endl;

    delete[] true_results;
    delete[] _sample;

    return 0;
}

static
int build_svm_classifier( cv::String& data_filename, cv::String& filename_to_save, cv::String& filename_to_load )
{
    cv::Mat data, responses, train_resp, train_data;
    int nsamples_all = 0, ntrain_samples = 0;
    int var_count;
    cv::ocl::CvSVM_OCL svm;

    int ok = read_num_class_data( data_filename, 16, data, responses );
    if( !ok )
    {
        cout << "Could not read the database " << data_filename << endl;
        return -1;
    }

    ////////// SVM parameters ///////////////////////////////
    CvSVMParams param;
    param.kernel_type = CvSVM::RBF;
    param.svm_type = CvSVM::C_SVC;
    param.C = 1;
    ///////////////////////////////////////////////////////////

    cout << "The database " << data_filename << " is loaded." << endl;
    nsamples_all = data.rows;
    ntrain_samples = (int)(nsamples_all * 0.1);
    var_count = data.cols;

    // Create or load Random Trees classifier
    if( !filename_to_load.empty() )
    {
        // load classifier from the specified file
        svm.load( filename_to_load.c_str() );
        ntrain_samples = 0;
        if( svm.get_var_count() == 0 )
        {
            cout << "Could not read the classifier " << filename_to_load << endl;
            return -1;
        }
        cout << "The classifier " << filename_to_load << " is loaded." << endl;
    }
    else
    {
        // train classifier
        cout << "Training the classifier (may take a few minutes)..." << endl;
        train_data = data(cv::Range(0, ntrain_samples), cv::Range::all());
        train_resp = cv::Mat( ntrain_samples, 1, CV_32FC1);
        for (int i = 0; i < ntrain_samples; i++)
        {
            train_resp.at<float>(i, 0) = responses.at<float>(i, 0);
        }

        double t = (double)cvGetTickCount();
        svm.train(train_data, train_resp, cv::Mat(), cv::Mat(), param);
        t = (double)cvGetTickCount() - t;
        cout << "Training time: " << t / (cvGetTickFrequency() * 1000.) << "ms" << endl;
    }

    // classification
    std::vector<float> _sample(var_count * (nsamples_all - ntrain_samples));
    cv::Mat sample = Mat( nsamples_all - ntrain_samples, 16, CV_32FC1, &_sample[0] );
    std::vector<float> true_results(nsamples_all - ntrain_samples);
    for (int j = ntrain_samples; j < nsamples_all; j++)
    {
        float *s = (float*)data.data + j * var_count;

        for (int i = 0; i < var_count; i++)
        {
            sample.at<float>((j - ntrain_samples), i) = s[i];
        }
        true_results[j - ntrain_samples] = responses.at<float>(j, 0);
    }
    Mat result = Mat(1, nsamples_all - ntrain_samples, CV_32FC1);

    CvMat sample_pre = sample;
    CvMat result_pre = result;

    cout << "Classification (may take a few minutes)..." << endl;
    double t = (double)cvGetTickCount();
    svm.predict(&sample_pre, &result_pre);
    t = (double)cvGetTickCount() - t;
    cout << "Prediction time: " << t / (cvGetTickFrequency() * 1000.) << "ms" << endl;

    int true_resp = 0;
    for (int i = 0; i < nsamples_all - ntrain_samples; i++)
    {
        if (result.at<float>(0, i) == true_results[i])
        {
            true_resp++;
        }
    }

    cout << "true_resp = " << (float)true_resp / (nsamples_all - ntrain_samples) * 100 << "%" << endl;

    if( !filename_to_save.empty() )
    {
        svm.save( filename_to_save.c_str() );
    }

    return 0;
}

enum
{
    METHOD_KNN = 0,
    METHOD_SVM
};

int main( int argc, char *argv[] )
{
    cv::String filename_to_save = "";
    cv::String filename_to_load = "";
    cv::String default_data_filename = "./letter-recognition.data";
    cv::String data_filename = default_data_filename;
    int method = METHOD_SVM;

    const char* keys =
        {
        "{    d| data|| flag -data=letter_recognition.xml  }"
        "{    s| save|| flag -save=filename.xml }"
        "{    l| load|| flag -load=filename.xml }"
        "{    1|     | svm| method will be used(knn or svm), default is svm}"
        };
    cv::CommandLineParser parse(argc, argv, keys);
    string data = parse.get<string>("d");
    if( !data.empty())
        data_filename = data;
    filename_to_save = parse.get<string>("s");
    filename_to_load = parse.get<string>("l");
    string method_arg = parse.get<string>("1");
    if ( method_arg.compare("knn") == 0)
        method = METHOD_KNN;

    if( (method == METHOD_KNN ?
             build_knearest_classifier( data_filename, 10 ) :
             method == METHOD_SVM ?
             build_svm_classifier( data_filename, filename_to_save, filename_to_load ) :
             -1) < 0)
    {
        help();
    }
    return 0;
}
