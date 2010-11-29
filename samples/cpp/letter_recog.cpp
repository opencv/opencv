#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"

#include <cstdio>
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

// This function reads data and responses from the file <filename>
static int
read_num_class_data( const char* filename, int var_count,
                     CvMat** data, CvMat** responses )
{
    const int M = 1024;
    FILE* f = fopen( filename, "rt" );
    CvMemStorage* storage;
    CvSeq* seq;
    char buf[M+2];
    float* el_ptr;
    CvSeqReader reader;
    int i, j;

    if( !f )
        return 0;

    el_ptr = new float[var_count+1];
    storage = cvCreateMemStorage();
    seq = cvCreateSeq( 0, sizeof(*seq), (var_count+1)*sizeof(float), storage );

    for(;;)
    {
        char* ptr;
        if( !fgets( buf, M, f ) || !strchr( buf, ',' ) )
            break;
        el_ptr[0] = buf[0];
        ptr = buf+2;
        for( i = 1; i <= var_count; i++ )
        {
            int n = 0;
            sscanf( ptr, "%f%n", el_ptr + i, &n );
            ptr += n + 1;
        }
        if( i <= var_count )
            break;
        cvSeqPush( seq, el_ptr );
    }
    fclose(f);

    *data = cvCreateMat( seq->total, var_count, CV_32F );
    *responses = cvCreateMat( seq->total, 1, CV_32F );

    cvStartReadSeq( seq, &reader );

    for( i = 0; i < seq->total; i++ )
    {
        const float* sdata = (float*)reader.ptr + 1;
        float* ddata = data[0]->data.fl + var_count*i;
        float* dr = responses[0]->data.fl + i;

        for( j = 0; j < var_count; j++ )
            ddata[j] = sdata[j];
        *dr = sdata[-1];
        CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
    }

    cvReleaseMemStorage( &storage );
    delete el_ptr;
    return 1;
}

static
int build_rtrees_classifier( char* data_filename,
    char* filename_to_save, char* filename_to_load )
{
    CvMat* data = 0;
    CvMat* responses = 0;
    CvMat* var_type = 0;
    CvMat* sample_idx = 0;

    int ok = read_num_class_data( data_filename, 16, &data, &responses );
    int nsamples_all = 0, ntrain_samples = 0;
    int i = 0;
    double train_hr = 0, test_hr = 0;
    CvRTrees forest;
    CvMat* var_importance = 0;

    if( !ok )
    {
        printf( "Could not read the database %s\n", data_filename );
        return -1;
    }

    printf( "The database %s is loaded.\n", data_filename );
    nsamples_all = data->rows;
    ntrain_samples = (int)(nsamples_all*0.8);

    // Create or load Random Trees classifier
    if( filename_to_load )
    {
        // load classifier from the specified file
        forest.load( filename_to_load );
        ntrain_samples = 0;
        if( forest.get_tree_count() == 0 )
        {
            printf( "Could not read the classifier %s\n", filename_to_load );
            return -1;
        }
        printf( "The classifier %s is loaded.\n", data_filename );
    }
    else
    {
        // create classifier by using <data> and <responses>
        printf( "Training the classifier ...\n");

        // 1. create type mask
        var_type = cvCreateMat( data->cols + 1, 1, CV_8U );
        cvSet( var_type, cvScalarAll(CV_VAR_ORDERED) );
        cvSetReal1D( var_type, data->cols, CV_VAR_CATEGORICAL );

        // 2. create sample_idx
        sample_idx = cvCreateMat( 1, nsamples_all, CV_8UC1 );
        {
            CvMat mat;
            cvGetCols( sample_idx, &mat, 0, ntrain_samples );
            cvSet( &mat, cvRealScalar(1) );

            cvGetCols( sample_idx, &mat, ntrain_samples, nsamples_all );
            cvSetZero( &mat );
        }

        // 3. train classifier
        forest.train( data, CV_ROW_SAMPLE, responses, 0, sample_idx, var_type, 0,
            CvRTParams(10,10,0,false,15,0,true,4,100,0.01f,CV_TERMCRIT_ITER));
        printf( "\n");
    }

    // compute prediction error on train and test data
    for( i = 0; i < nsamples_all; i++ )
    {
        double r;
        CvMat sample;
        cvGetRow( data, &sample, i );

        r = forest.predict( &sample );
        r = fabs((double)r - responses->data.fl[i]) <= FLT_EPSILON ? 1 : 0;

        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= (double)(nsamples_all-ntrain_samples);
    train_hr /= (double)ntrain_samples;
    printf( "Recognition rate: train = %.1f%%, test = %.1f%%\n",
            train_hr*100., test_hr*100. );

    printf( "Number of trees: %d\n", forest.get_tree_count() );

    // Print variable importance
    var_importance = (CvMat*)forest.get_var_importance();
    if( var_importance )
    {
        double rt_imp_sum = cvSum( var_importance ).val[0];
        printf("var#\timportance (in %%):\n");
        for( i = 0; i < var_importance->cols; i++ )
            printf( "%-2d\t%-4.1f\n", i,
            100.f*var_importance->data.fl[i]/rt_imp_sum);
    }

    //Print some proximitites
    printf( "Proximities between some samples corresponding to the letter 'T':\n" );
    {
        CvMat sample1, sample2;
        const int pairs[][2] = {{0,103}, {0,106}, {106,103}, {-1,-1}};

        for( i = 0; pairs[i][0] >= 0; i++ )
        {
            cvGetRow( data, &sample1, pairs[i][0] );
            cvGetRow( data, &sample2, pairs[i][1] );
            printf( "proximity(%d,%d) = %.1f%%\n", pairs[i][0], pairs[i][1],
                forest.get_proximity( &sample1, &sample2 )*100. );
        }
    }

    // Save Random Trees classifier to file if needed
    if( filename_to_save )
        forest.save( filename_to_save );

    cvReleaseMat( &sample_idx );
    cvReleaseMat( &var_type );
    cvReleaseMat( &data );
    cvReleaseMat( &responses );

    return 0;
}


static
int build_boost_classifier( char* data_filename,
    char* filename_to_save, char* filename_to_load )
{
    const int class_count = 26;
    CvMat* data = 0;
    CvMat* responses = 0;
    CvMat* var_type = 0;
    CvMat* temp_sample = 0;
    CvMat* weak_responses = 0;

    int ok = read_num_class_data( data_filename, 16, &data, &responses );
    int nsamples_all = 0, ntrain_samples = 0;
    int var_count;
    int i, j, k;
    double train_hr = 0, test_hr = 0;
    CvBoost boost;

    if( !ok )
    {
        printf( "Could not read the database %s\n", data_filename );
        return -1;
    }

    printf( "The database %s is loaded.\n", data_filename );
    nsamples_all = data->rows;
    ntrain_samples = (int)(nsamples_all*0.5);
    var_count = data->cols;

    // Create or load Boosted Tree classifier
    if( filename_to_load )
    {
        // load classifier from the specified file
        boost.load( filename_to_load );
        ntrain_samples = 0;
        if( !boost.get_weak_predictors() )
        {
            printf( "Could not read the classifier %s\n", filename_to_load );
            return -1;
        }
        printf( "The classifier %s is loaded.\n", data_filename );
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

        CvMat* new_data = cvCreateMat( ntrain_samples*class_count, var_count + 1, CV_32F );
        CvMat* new_responses = cvCreateMat( ntrain_samples*class_count, 1, CV_32S );

        // 1. unroll the database type mask
        printf( "Unrolling the database...\n");
        for( i = 0; i < ntrain_samples; i++ )
        {
            float* data_row = (float*)(data->data.ptr + data->step*i);
            for( j = 0; j < class_count; j++ )
            {
                float* new_data_row = (float*)(new_data->data.ptr +
                                new_data->step*(i*class_count+j));
                for( k = 0; k < var_count; k++ )
                    new_data_row[k] = data_row[k];
                new_data_row[var_count] = (float)j;
                new_responses->data.i[i*class_count + j] = responses->data.fl[i] == j+'A';
            }
        }

        // 2. create type mask
        var_type = cvCreateMat( var_count + 2, 1, CV_8U );
        cvSet( var_type, cvScalarAll(CV_VAR_ORDERED) );
        // the last indicator variable, as well
        // as the new (binary) response are categorical
        cvSetReal1D( var_type, var_count, CV_VAR_CATEGORICAL );
        cvSetReal1D( var_type, var_count+1, CV_VAR_CATEGORICAL );

        // 3. train classifier
        printf( "Training the classifier (may take a few minutes)...\n");
        boost.train( new_data, CV_ROW_SAMPLE, new_responses, 0, 0, var_type, 0,
            CvBoostParams(CvBoost::REAL, 100, 0.95, 5, false, 0 ));
        cvReleaseMat( &new_data );
        cvReleaseMat( &new_responses );
        printf("\n");
    }

    temp_sample = cvCreateMat( 1, var_count + 1, CV_32F );
    weak_responses = cvCreateMat( 1, boost.get_weak_predictors()->total, CV_32F ); 

    // compute prediction error on train and test data
    for( i = 0; i < nsamples_all; i++ )
    {
        int best_class = 0;
        double max_sum = -DBL_MAX;
        double r;
        CvMat sample;
        cvGetRow( data, &sample, i );
        for( k = 0; k < var_count; k++ )
            temp_sample->data.fl[k] = sample.data.fl[k];

        for( j = 0; j < class_count; j++ )
        {
            temp_sample->data.fl[var_count] = (float)j;
            boost.predict( temp_sample, 0, weak_responses );
            double sum = cvSum( weak_responses ).val[0];
            if( max_sum < sum )
            {
                max_sum = sum;
                best_class = j + 'A';
            }
        }

        r = fabs(best_class - responses->data.fl[i]) < FLT_EPSILON ? 1 : 0;

        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= (double)(nsamples_all-ntrain_samples);
    train_hr /= (double)ntrain_samples;
    printf( "Recognition rate: train = %.1f%%, test = %.1f%%\n",
            train_hr*100., test_hr*100. );

    printf( "Number of trees: %d\n", boost.get_weak_predictors()->total );

    // Save classifier to file if needed
    if( filename_to_save )
        boost.save( filename_to_save );

    cvReleaseMat( &temp_sample );
    cvReleaseMat( &weak_responses );
    cvReleaseMat( &var_type );
    cvReleaseMat( &data );
    cvReleaseMat( &responses );

    return 0;
}


static
int build_mlp_classifier( char* data_filename,
    char* filename_to_save, char* filename_to_load )
{
    const int class_count = 26;
    CvMat* data = 0;
    CvMat train_data;
    CvMat* responses = 0;
    CvMat* mlp_response = 0;

    int ok = read_num_class_data( data_filename, 16, &data, &responses );
    int nsamples_all = 0, ntrain_samples = 0;
    int i, j;
    double train_hr = 0, test_hr = 0;
    CvANN_MLP mlp;

    if( !ok )
    {
        printf( "Could not read the database %s\n", data_filename );
        return -1;
    }

    printf( "The database %s is loaded.\n", data_filename );
    nsamples_all = data->rows;
    ntrain_samples = (int)(nsamples_all*0.8);

    // Create or load MLP classifier
    if( filename_to_load )
    {
        // load classifier from the specified file
        mlp.load( filename_to_load );
        ntrain_samples = 0;
        if( !mlp.get_layer_count() )
        {
            printf( "Could not read the classifier %s\n", filename_to_load );
            return -1;
        }
        printf( "The classifier %s is loaded.\n", data_filename );
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

        CvMat* new_responses = cvCreateMat( ntrain_samples, class_count, CV_32F );

        // 1. unroll the responses
        printf( "Unrolling the responses...\n");
        for( i = 0; i < ntrain_samples; i++ )
        {
            int cls_label = cvRound(responses->data.fl[i]) - 'A';
            float* bit_vec = (float*)(new_responses->data.ptr + i*new_responses->step);
            for( j = 0; j < class_count; j++ )
                bit_vec[j] = 0.f;
            bit_vec[cls_label] = 1.f;
        }
        cvGetRows( data, &train_data, 0, ntrain_samples );

        // 2. train classifier
        int layer_sz[] = { data->cols, 100, 100, class_count };
        CvMat layer_sizes =
            cvMat( 1, (int)(sizeof(layer_sz)/sizeof(layer_sz[0])), CV_32S, layer_sz );
        mlp.create( &layer_sizes );
        printf( "Training the classifier (may take a few minutes)...\n");
        mlp.train( &train_data, new_responses, 0, 0,
            CvANN_MLP_TrainParams(cvTermCriteria(CV_TERMCRIT_ITER,300,0.01),
#if 1
            CvANN_MLP_TrainParams::BACKPROP,0.001));
#else
            CvANN_MLP_TrainParams::RPROP,0.05));
#endif
        cvReleaseMat( &new_responses );
        printf("\n");
    }

    mlp_response = cvCreateMat( 1, class_count, CV_32F );

    // compute prediction error on train and test data
    for( i = 0; i < nsamples_all; i++ )
    {
        int best_class;
        CvMat sample;
        cvGetRow( data, &sample, i );
        CvPoint max_loc = {0,0};
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

    // Save classifier to file if needed
    if( filename_to_save )
        mlp.save( filename_to_save );

    cvReleaseMat( &mlp_response );
    cvReleaseMat( &data );
    cvReleaseMat( &responses );

    return 0;
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
        -1) < 0)
    {
        printf("This is letter recognition sample.\n"
                "The usage: letter_recog [-data <path to letter-recognition.data>] \\\n"
                "  [-save <output XML file for the classifier>] \\\n"
                "  [-load <XML file with the pre-trained classifier>] \\\n"
                "  [-boost|-mlp] # to use boost/mlp classifier instead of default Random Trees\n" );
    }
    return 0;
}
