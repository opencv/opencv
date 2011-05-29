#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/core/core_c.h"
#include <stdio.h>
#include <map>

using namespace std;
using namespace cv;

void help()
{
    printf(
        "\nThis sample demonstrates how to use different decision trees and forests including boosting and random trees:\n"
        "CvDTree dtree;\n"
        "CvBoost boost;\n"
        "CvRTrees rtrees;\n"
        "CvERTrees ertrees;\n"
        "CvGBTrees gbtrees;\n"
        "Usage: \n"
        "       ./tree_engine [--response_column]=<specified the 0-based index of the response, 0 as default> \n"
        "[--categorical_response]=<specifies that the response is categorical, 0-false, 1-true, 0 as default> \n"
        "[--csv_filename]=<is the name of training data file in comma-separated value format> \n"
        );
}


int count_classes(CvMLData& data)
{
    cv::Mat r(data.get_responses());
    std::map<int, int> rmap;
    int i, n = (int)r.total();
    for( i = 0; i < n; i++ )
    {
        float val = r.at<float>(i);
        int ival = cvRound(val);
        if( ival != val )
            return -1;
        rmap[ival] = 1; 
    }
    return rmap.size();
}

void print_result(float train_err, float test_err, const CvMat* _var_imp)
{
    printf( "train error    %f\n", train_err );
    printf( "test error    %f\n\n", test_err );
       
    if (_var_imp)
    {
        cv::Mat var_imp(_var_imp), sorted_idx;
        cv::sortIdx(var_imp, sorted_idx, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
        
        printf( "variable importance:\n" );
        int i, n = (int)var_imp.total();
        int type = var_imp.type();
        CV_Assert(type == CV_32F || type == CV_64F);
        
        for( i = 0; i < n; i++)
        {
            int k = sorted_idx.at<int>(i);
            printf( "%d\t%f\n", k, type == CV_32F ? var_imp.at<float>(k) : var_imp.at<double>(k));
        }
    }
    printf("\n");
}

int main(int argc, const char** argv)
{
    help();

    CommandLineParser parser(argc, argv);

    string filename = parser.get<string>("csv_filename");
    int response_idx = parser.get<int>("response_column", 0);
    bool categorical_response = (bool)parser.get<int>("categorical_response", 1);

    if(filename.empty())
    {
        printf("\n Please, select value for --csv_filename key \n");
        help();
        return -1;
    }
        
    printf("\nReading in %s...\n\n",filename.c_str());
    CvDTree dtree;
    CvBoost boost;
    CvRTrees rtrees;
    CvERTrees ertrees;
	CvGBTrees gbtrees;

    CvMLData data;

    
    CvTrainTestSplit spl( 0.5f );
    
    if ( data.read_csv( filename.c_str() ) == 0)
    {
        data.set_response_idx( response_idx );
        if(categorical_response)
            data.change_var_type( response_idx, CV_VAR_CATEGORICAL );
        data.set_train_test_split( &spl );
        
        printf("======DTREE=====\n");
        dtree.train( &data, CvDTreeParams( 10, 2, 0, false, 16, 0, false, false, 0 ));
        print_result( dtree.calc_error( &data, CV_TRAIN_ERROR), dtree.calc_error( &data, CV_TEST_ERROR ), dtree.get_var_importance() );

        if( categorical_response && count_classes(data) == 2 )
        {
        printf("======BOOST=====\n");
        boost.train( &data, CvBoostParams(CvBoost::DISCRETE, 100, 0.95, 2, false, 0));
        print_result( boost.calc_error( &data, CV_TRAIN_ERROR ), boost.calc_error( &data, CV_TEST_ERROR ), 0 ); //doesn't compute importance
        }

        printf("======RTREES=====\n");
        rtrees.train( &data, CvRTParams( 10, 2, 0, false, 16, 0, true, 0, 100, 0, CV_TERMCRIT_ITER ));
        print_result( rtrees.calc_error( &data, CV_TRAIN_ERROR), rtrees.calc_error( &data, CV_TEST_ERROR ), rtrees.get_var_importance() );

        printf("======ERTREES=====\n");
        ertrees.train( &data, CvRTParams( 10, 2, 0, false, 16, 0, true, 0, 100, 0, CV_TERMCRIT_ITER ));
        print_result( ertrees.calc_error( &data, CV_TRAIN_ERROR), ertrees.calc_error( &data, CV_TEST_ERROR ), ertrees.get_var_importance() );

        printf("======GBTREES=====\n");
        gbtrees.train( &data, CvGBTreesParams(CvGBTrees::DEVIANCE_LOSS, 100, 0.05f, 0.6f, 10, true));
        print_result( gbtrees.calc_error( &data, CV_TRAIN_ERROR), gbtrees.calc_error( &data, CV_TEST_ERROR ), 0 ); //doesn't compute importance
    }
    else
        printf("File can not be read");

    return 0;
}
