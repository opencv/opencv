#include "opencv2/ml/ml.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include <stdio.h>
#include <string>
#include <map>

using namespace cv;
using namespace cv::ml;

static void help()
{
    printf(
        "\nThis sample demonstrates how to use different decision trees and forests including boosting and random trees.\n"
        "Usage:\n\t./tree_engine [-r <response_column>] [-ts type_spec] <csv filename>\n"
        "where -r <response_column> specified the 0-based index of the response (0 by default)\n"
        "-ts specifies the var type spec in the form ord[n1,n2-n3,n4-n5,...]cat[m1-m2,m3,m4-m5,...]\n"
        "<csv filename> is the name of training data file in comma-separated value format\n\n");
}

static void train_and_print_errs(Ptr<StatModel> model, const Ptr<TrainData>& data)
{
    bool ok = model->train(data);
    if( !ok )
    {
        printf("Training failed\n");
    }
    else
    {
        printf( "train error: %f\n", model->calcError(data, false, noArray()) );
        printf( "test error: %f\n\n", model->calcError(data, true, noArray()) );
    }
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        help();
        return 0;
    }
    const char* filename = 0;
    int response_idx = 0;
    std::string typespec;

    for(int i = 1; i < argc; i++)
    {
        if(strcmp(argv[i], "-r") == 0)
            sscanf(argv[++i], "%d", &response_idx);
        else if(strcmp(argv[i], "-ts") == 0)
            typespec = argv[++i];
        else if(argv[i][0] != '-' )
            filename = argv[i];
        else
        {
            printf("Error. Invalid option %s\n", argv[i]);
            help();
            return -1;
        }
    }

    printf("\nReading in %s...\n\n",filename);
    const double train_test_split_ratio = 0.5;

    Ptr<TrainData> data = TrainData::loadFromCSV(filename, 0, response_idx, response_idx+1, typespec);

    if( data.empty() )
    {
        printf("ERROR: File %s can not be read\n", filename);
        return 0;
    }

    data->setTrainTestSplitRatio(train_test_split_ratio);

    printf("======DTREE=====\n");
    Ptr<DTrees> dtree = DTrees::create(DTrees::Params( 10, 2, 0, false, 16, 0, false, false, Mat() ));
    train_and_print_errs(dtree, data);

    if( (int)data->getClassLabels().total() <= 2 ) // regression or 2-class classification problem
    {
        printf("======BOOST=====\n");
        Ptr<Boost> boost = Boost::create(Boost::Params(Boost::GENTLE, 100, 0.95, 2, false, Mat()));
        train_and_print_errs(boost, data);
    }

    printf("======RTREES=====\n");
    Ptr<RTrees> rtrees = RTrees::create(RTrees::Params(10, 2, 0, false, 16, Mat(), false, 0, TermCriteria(TermCriteria::MAX_ITER, 100, 0)));
    train_and_print_errs(rtrees, data);

    return 0;
}
