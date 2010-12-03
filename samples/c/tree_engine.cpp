#include "opencv2/ml/ml.hpp"
#include <stdio.h>

void help()
{
	printf(
		"\nThis sample demonstrates how to use different decision trees and forests including boosting and random trees:\n"
		"CvDTree dtree;\n"
		"CvBoost boost;\n"
		"CvRTrees rtrees;\n"
		"CvERTrees ertrees;\n"
		"CvGBTrees gbtrees;\n"
		"Date is hard coded to come from filename = \"../../../OpenCV/samples/c/waveform.data\";\n"
		"Or can come from filename = \"../../../OpenCV/samples/c/waveform.data\";\n"
		"Call:\n"
		"./tree_engine\n\n");
}
void print_result(float train_err, float test_err, const CvMat* var_imp)
{
    printf( "train error    %f\n", train_err );
    printf( "test error    %f\n\n", test_err );
       
    if (var_imp)
    {
        bool is_flt = false;
        if ( CV_MAT_TYPE( var_imp->type ) == CV_32FC1)
            is_flt = true;
        printf( "variable impotance\n" );
        for( int i = 0; i < var_imp->cols; i++)
        {
            printf( "%d     %f\n", i, is_flt ? var_imp->data.fl[i] : var_imp->data.db[i] );
        }
    }
    printf("\n");
}

int main()
{
    const int train_sample_count = 300;

//#define LEPIOTA
#ifdef LEPIOTA
    const char* filename = "../../../OpenCV/samples/c/agaricus-lepiota.data";
#else
    const char* filename = "../../../OpenCV/samples/c/waveform.data";
#endif

    CvDTree dtree;
    CvBoost boost;
    CvRTrees rtrees;
    CvERTrees ertrees;
	CvGBTrees gbtrees;

    CvMLData data;

    CvTrainTestSplit spl( train_sample_count );
    
    if ( data.read_csv( filename ) == 0)
    {

#ifdef LEPIOTA
        data.set_response_idx( 0 );     
#else
        data.set_response_idx( 21 );     
        data.change_var_type( 21, CV_VAR_CATEGORICAL );
#endif

        data.set_train_test_split( &spl );
        
        printf("======DTREE=====\n");
        dtree.train( &data, CvDTreeParams( 10, 2, 0, false, 16, 0, false, false, 0 ));
        print_result( dtree.calc_error( &data, CV_TRAIN_ERROR), dtree.calc_error( &data, CV_TEST_ERROR ), dtree.get_var_importance() );

#ifdef LEPIOTA
        printf("======BOOST=====\n");
        boost.train( &data, CvBoostParams(CvBoost::DISCRETE, 100, 0.95, 2, false, 0));
        print_result( boost.calc_error( &data, CV_TRAIN_ERROR ), boost.calc_error( &data ), 0 );
#endif

        printf("======RTREES=====\n");
        rtrees.train( &data, CvRTParams( 10, 2, 0, false, 16, 0, true, 0, 100, 0, CV_TERMCRIT_ITER ));
        print_result( rtrees.calc_error( &data, CV_TRAIN_ERROR), rtrees.calc_error( &data, CV_TEST_ERROR ), rtrees.get_var_importance() );

        printf("======ERTREES=====\n");
        ertrees.train( &data, CvRTParams( 10, 2, 0, false, 16, 0, true, 0, 100, 0, CV_TERMCRIT_ITER ));
        print_result( ertrees.calc_error( &data, CV_TRAIN_ERROR), ertrees.calc_error( &data, CV_TEST_ERROR ), ertrees.get_var_importance() );

		printf("======GBTREES=====\n");
		gbtrees.train( &data, CvGBTreesParams(CvGBTrees::DEVIANCE_LOSS, 100, 0.05f, 0.6f, 10, true));
		print_result( gbtrees.calc_error( &data, CV_TRAIN_ERROR), gbtrees.calc_error( &data, CV_TEST_ERROR ), 0 );
    }
    else
        printf("File can not be read");

    return 0;
}
