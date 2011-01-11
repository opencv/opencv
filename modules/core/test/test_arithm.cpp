#include "test_precomp.hpp"
#include <iostream>

using namespace cv;
using namespace std;

const int ARITHM_NTESTS = 1000;
const int ARITHM_RNG_SEED = -1;
const int ARITHM_MAX_NDIMS = 4;
const int ARITHM_MAX_SIZE_LOG = 10;
const int ARITHM_MAX_CHANNELS = 4;

static void getArithmValueRange(int depth, double& minval, double& maxval)
{
    minval = depth < CV_32S ? cvtest::getMinVal(depth) : depth == CV_32S ? -1000000 : -1000.;
    maxval = depth < CV_32S ? cvtest::getMinVal(depth) : depth == CV_32S ? 1000000 : 1000.;
}

static double getArithmMaxErr(int depth)
{
    return depth < CV_32F ? 0 : 4;
}

TEST(ArithmTest, add)
{
    int testIdx = 0;
    RNG rng(ARITHM_RNG_SEED);
    for( testIdx = 0; testIdx < ARITHM_NTESTS; testIdx++ )
    {
        double minval, maxval;
        vector<int> size;
        cvtest::randomSize(rng, 2, ARITHM_MAX_NDIMS, ARITHM_MAX_SIZE_LOG, size);
        int type = cvtest::randomType(rng, cvtest::TYPE_MASK_ALL, 1, ARITHM_MAX_CHANNELS);
        int depth = CV_MAT_DEPTH(type);
        bool haveMask = rng.uniform(0, 4) == 0;
        
        getArithmValueRange(depth, minval, maxval);
        Mat src1 = cvtest::randomMat(rng, size, type, minval, maxval, true);
        Mat src2 = cvtest::randomMat(rng, size, type, minval, maxval, true);
        Mat dst0 = cvtest::randomMat(rng, size, type, minval, maxval, false);
        Mat dst = cvtest::randomMat(rng, size, type, minval, maxval, true);
        Mat mask;
        if( haveMask )
        {
            mask = cvtest::randomMat(rng, size, CV_8U, 0, 2, true);
            cvtest::copy(dst0, dst);
            cvtest::add(src1, 1, src2, 1, Scalar::all(0), dst0, dst.type());
            cvtest::copy(dst, dst0, mask, true);
            add(src1, src2, dst, mask);
        }
        else
        {
            cvtest::add(src1, 1, src2, 1, Scalar::all(0), dst0, dst.type());
            add(src1, src2, dst);
        }
        
        double maxErr = getArithmMaxErr(depth);
        vector<int> pos;
        ASSERT_TRUE(cvtest::cmpEps(dst0, dst, maxErr, &pos)) << "position: " << Mat(pos);
    }
}
