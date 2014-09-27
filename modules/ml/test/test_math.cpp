//////////////////////////////////////////////////////////////////////////////////////////
/////////////////// tests for matrix operations and math functions ///////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

#include "test_precomp.hpp"
#include <float.h>
#include <math.h>

using namespace cv;
using namespace cv::ml;
using namespace std;


// TODO: eigenvv, invsqrt, cbrt, fastarctan, (round, floor, ceil(?)),

enum
{
    MAT_N_DIM_C1,
    MAT_N_1_CDIM,
    MAT_1_N_CDIM,
    MAT_N_DIM_C1_NONCONT,
    MAT_N_1_CDIM_NONCONT,
    VECTOR
};

class CV_KMeansSingularTest : public cvtest::BaseTest
{
public:
    CV_KMeansSingularTest() {}
    ~CV_KMeansSingularTest() {}
protected:
    void run(int inVariant)
    {
        int i, iter = 0, N = 0, N0 = 0, K = 0, dims = 0;
        Mat labels;
        try
        {
            RNG& rng = theRNG();
            const int MAX_DIM=5;
            int MAX_POINTS = 100, maxIter = 100;
            for( iter = 0; iter < maxIter; iter++ )
            {
                ts->update_context(this, iter, true);
                dims = rng.uniform(inVariant == MAT_1_N_CDIM ? 2 : 1, MAX_DIM+1);
                N = rng.uniform(1, MAX_POINTS+1);
                N0 = rng.uniform(1, MAX(N/10, 2));
                K = rng.uniform(1, N+1);

                if (inVariant == VECTOR)
                {
                    dims = 2;

                    std::vector<cv::Point2f> data0(N0);
                    rng.fill(data0, RNG::UNIFORM, -1, 1);

                    std::vector<cv::Point2f> data(N);
                    for( i = 0; i < N; i++ )
                        data[i] = data0[rng.uniform(0, N0)];

                    kmeans(data, K, labels, TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 30, 0),
                           5, KMEANS_PP_CENTERS);
                }
                else
                {
                    Mat data0(N0, dims, CV_32F);
                    rng.fill(data0, RNG::UNIFORM, -1, 1);

                    Mat data;

                    switch (inVariant)
                    {
                    case MAT_N_DIM_C1:
                        data.create(N, dims, CV_32F);
                        for( i = 0; i < N; i++ )
                            data0.row(rng.uniform(0, N0)).copyTo(data.row(i));
                        break;

                    case MAT_N_1_CDIM:
                        data.create(N, 1, CV_32FC(dims));
                        for( i = 0; i < N; i++ )
                            memcpy(data.ptr(i), data0.ptr(rng.uniform(0, N0)), dims * sizeof(float));
                        break;

                    case MAT_1_N_CDIM:
                        data.create(1, N, CV_32FC(dims));
                        for( i = 0; i < N; i++ )
                            memcpy(data.ptr() + i * dims * sizeof(float), data0.ptr(rng.uniform(0, N0)), dims * sizeof(float));
                        break;

                    case MAT_N_DIM_C1_NONCONT:
                        data.create(N, dims + 5, CV_32F);
                        data = data(Range(0, N), Range(0, dims));
                        for( i = 0; i < N; i++ )
                            data0.row(rng.uniform(0, N0)).copyTo(data.row(i));
                        break;

                    case MAT_N_1_CDIM_NONCONT:
                        data.create(N, 3, CV_32FC(dims));
                        data = data.colRange(0, 1);
                        for( i = 0; i < N; i++ )
                            memcpy(data.ptr(i), data0.ptr(rng.uniform(0, N0)), dims * sizeof(float));
                        break;
                    }

                    kmeans(data, K, labels, TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 30, 0),
                           5, KMEANS_PP_CENTERS);
                }

                Mat hist(K, 1, CV_32S, Scalar(0));
                for( i = 0; i < N; i++ )
                {
                    int l = labels.at<int>(i);
                    CV_Assert(0 <= l && l < K);
                    hist.at<int>(l)++;
                }
                for( i = 0; i < K; i++ )
                    CV_Assert( hist.at<int>(i) != 0 );
            }
        }
        catch(...)
        {
            ts->printf(cvtest::TS::LOG,
                       "context: iteration=%d, N=%d, N0=%d, K=%d\n",
                       iter, N, N0, K);
            std::cout << labels << std::endl;
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        }
    }
};

TEST(Core_KMeans, singular) { CV_KMeansSingularTest test; test.safe_run(MAT_N_DIM_C1); }

CV_ENUM(KMeansInputVariant, MAT_N_DIM_C1, MAT_N_1_CDIM, MAT_1_N_CDIM, MAT_N_DIM_C1_NONCONT, MAT_N_1_CDIM_NONCONT, VECTOR)

typedef testing::TestWithParam<KMeansInputVariant> Core_KMeans_InputVariants;

TEST_P(Core_KMeans_InputVariants, singular)
{
    CV_KMeansSingularTest test;
    test.safe_run(GetParam());
}

INSTANTIATE_TEST_CASE_P(AllVariants, Core_KMeans_InputVariants, KMeansInputVariant::all());


/* End of file. */
