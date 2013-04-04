#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

CV_ENUM(NormType, NORM_L1, NORM_L2, NORM_L2SQR, NORM_HAMMING, NORM_HAMMING2)

typedef std::tr1::tuple<NormType, MatType, bool> Norm_Destination_CrossCheck_t;
typedef perf::TestBaseWithParam<Norm_Destination_CrossCheck_t> Norm_Destination_CrossCheck;

typedef std::tr1::tuple<NormType, bool> Norm_CrossCheck_t;
typedef perf::TestBaseWithParam<Norm_CrossCheck_t> Norm_CrossCheck;

typedef std::tr1::tuple<MatType, bool> Source_CrossCheck_t;
typedef perf::TestBaseWithParam<Source_CrossCheck_t> Source_CrossCheck;

void generateData( Mat& query, Mat& train, const int sourceType );

PERF_TEST_P(Norm_Destination_CrossCheck, batchDistance_8U,
            testing::Combine(testing::Values((int)NORM_L1, (int)NORM_L2SQR),
                             testing::Values(CV_32S, CV_32F),
                             testing::Bool()
                             )
            )
{
    NormType normType = get<0>(GetParam());
    int destinationType = get<1>(GetParam());
    bool isCrossCheck = get<2>(GetParam());
    int knn = isCrossCheck ? 1 : 0;

    Mat queryDescriptors;
    Mat trainDescriptors;
    Mat dist;
    Mat ndix;

    generateData(queryDescriptors, trainDescriptors, CV_8U);

    TEST_CYCLE()
    {
        batchDistance(queryDescriptors, trainDescriptors, dist, destinationType, (isCrossCheck) ? ndix : noArray(),
                      normType, knn, Mat(), 0, isCrossCheck);
    }

    SANITY_CHECK(dist);
    if (isCrossCheck) SANITY_CHECK(ndix);
}

PERF_TEST_P(Norm_CrossCheck, batchDistance_Dest_32S,
            testing::Combine(testing::Values((int)NORM_HAMMING, (int)NORM_HAMMING2),
                             testing::Bool()
                             )
            )
{
    NormType normType = get<0>(GetParam());
    bool isCrossCheck = get<1>(GetParam());
    int knn = isCrossCheck ? 1 : 0;

    Mat queryDescriptors;
    Mat trainDescriptors;
    Mat dist;
    Mat ndix;

    generateData(queryDescriptors, trainDescriptors, CV_8U);

    TEST_CYCLE()
    {
        batchDistance(queryDescriptors, trainDescriptors, dist, CV_32S, (isCrossCheck) ? ndix : noArray(),
                      normType, knn, Mat(), 0, isCrossCheck);
    }

    SANITY_CHECK(dist);
    if (isCrossCheck) SANITY_CHECK(ndix);
}

PERF_TEST_P(Source_CrossCheck, batchDistance_L2,
            testing::Combine(testing::Values(CV_8U, CV_32F),
                             testing::Bool()
                             )
            )
{
    int sourceType = get<0>(GetParam());
    bool isCrossCheck = get<1>(GetParam());
    int knn = isCrossCheck ? 1 : 0;

    Mat queryDescriptors;
    Mat trainDescriptors;
    Mat dist;
    Mat ndix;

    generateData(queryDescriptors, trainDescriptors, sourceType);

    declare.time(50);
    TEST_CYCLE()
    {
        batchDistance(queryDescriptors, trainDescriptors, dist, CV_32F, (isCrossCheck) ? ndix : noArray(),
                      NORM_L2, knn, Mat(), 0, isCrossCheck);
    }

    SANITY_CHECK(dist);
    if (isCrossCheck) SANITY_CHECK(ndix);
}

PERF_TEST_P(Norm_CrossCheck, batchDistance_32F,
            testing::Combine(testing::Values((int)NORM_L1, (int)NORM_L2SQR),
                             testing::Bool()
                             )
            )
{
    NormType normType = get<0>(GetParam());
    bool isCrossCheck = get<1>(GetParam());
    int knn = isCrossCheck ? 1 : 0;

    Mat queryDescriptors;
    Mat trainDescriptors;
    Mat dist;
    Mat ndix;

    generateData(queryDescriptors, trainDescriptors, CV_32F);
    declare.time(100);

    TEST_CYCLE()
    {
        batchDistance(queryDescriptors, trainDescriptors, dist, CV_32F, (isCrossCheck) ? ndix : noArray(),
                      normType, knn, Mat(), 0, isCrossCheck);
    }

    SANITY_CHECK(dist, 1e-4);
    if (isCrossCheck) SANITY_CHECK(ndix);
}

void generateData( Mat& query, Mat& train, const int sourceType )
{
    const int dim = 500;
    const int queryDescCount = 300; // must be even number because we split train data in some cases in two
    const int countFactor = 4; // do not change it
    RNG& rng = theRNG();

    // Generate query descriptors randomly.
    // Descriptor vector elements are integer values.
    Mat buf( queryDescCount, dim, CV_32SC1 );
    rng.fill( buf, RNG::UNIFORM, Scalar::all(0), Scalar(3) );
    buf.convertTo( query, sourceType );

    // Generate train decriptors as follows:
    // copy each query descriptor to train set countFactor times
    // and perturb some one element of the copied descriptors in
    // in ascending order. General boundaries of the perturbation
    // are (0.f, 1.f).
    train.create( query.rows*countFactor, query.cols, sourceType );
    float step = (sourceType == CV_8U ? 256.f : 1.f) / countFactor;
    for( int qIdx = 0; qIdx < query.rows; qIdx++ )
    {
        Mat queryDescriptor = query.row(qIdx);
        for( int c = 0; c < countFactor; c++ )
        {
            int tIdx = qIdx * countFactor + c;
            Mat trainDescriptor = train.row(tIdx);
            queryDescriptor.copyTo( trainDescriptor );
            int elem = rng(dim);
            float diff = rng.uniform( step*c, step*(c+1) );
            trainDescriptor.col(elem) += diff;
        }
    }
}
