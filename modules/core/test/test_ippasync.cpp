#include "test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#include "opencv2/core/ippasync.hpp"

using namespace cv;
using namespace std;
using namespace cvtest;

namespace cvtest {
namespace ocl {

PARAM_TEST_CASE(IPPAsync, MatDepth, Channels, hppAccelType)
{
    int type;
    int cn;
    int depth;
    hppAccelType accelType;

    Mat matrix, result;
    Ptr<hppiMatrix> hppMat;
    hppAccel accel;
    hppiVirtualMatrix * virtMatrix;
    hppStatus sts;

    virtual void SetUp()
    {
        type = CV_MAKE_TYPE(GET_PARAM(0), GET_PARAM(1));
        depth = GET_PARAM(0);
        cn = GET_PARAM(1);
        accelType = GET_PARAM(2);
    }

    virtual void generateTestData()
    {
        Size matrix_Size = randomSize(2, 100);
        const double upValue = 100;

        matrix = randomMat(matrix_Size, type, -upValue, upValue);
    }

    void Near(double threshold = 0.0)
    {
        EXPECT_MAT_NEAR(matrix, result, threshold);
    }
};

TEST_P(IPPAsync, accuracy)
{
    if (depth==CV_32S || depth==CV_64F)
        return;
    
    sts = hppCreateInstance(accelType, 0, &accel);
    CV_Assert(sts==HPP_STATUS_NO_ERROR);
    virtMatrix = hppiCreateVirtualMatrices(accel, 2);

    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();
        hppMat = hpp::getHpp(matrix);

        hppScalar a = 3;

        sts = hppiAddC(accel, hppMat, a, 0, virtMatrix[0]);
        CV_Assert(sts==HPP_STATUS_NO_ERROR);
        sts = hppiSubC(accel, virtMatrix[0], a, 0, virtMatrix[1]);
        CV_Assert(sts==HPP_STATUS_NO_ERROR);

        sts = hppWait(accel, HPP_TIME_OUT_INFINITE);
        CV_Assert(sts==HPP_STATUS_NO_ERROR);
        
        result = hpp::getMat(virtMatrix[1], accel, cn);

        Near(5.0e-6);
    }

    sts = hppiDeleteVirtualMatrices(accel, virtMatrix);
    CV_Assert(sts==HPP_STATUS_NO_ERROR);
    sts = hppDeleteInstance(accel);
    CV_Assert(sts==HPP_STATUS_NO_ERROR);
}

TEST_P(IPPAsync, conversion)
{
    sts = hppCreateInstance(accelType, 0, &accel);
    CV_Assert(sts==HPP_STATUS_NO_ERROR);
    virtMatrix = hppiCreateVirtualMatrices(accel, 1);

    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();
        hppMat = hpp::getHpp(matrix);

        sts = hppiCopy (accel, hppMat, virtMatrix[0]);
        CV_Assert(sts==HPP_STATUS_NO_ERROR);

        sts = hppWait(accel, HPP_TIME_OUT_INFINITE);
        CV_Assert(sts==HPP_STATUS_NO_ERROR);
        
        result = hpp::getMat(virtMatrix[0], accel, cn);

        Near();
    }

    sts = hppiDeleteVirtualMatrices(accel, virtMatrix);
    CV_Assert(sts==HPP_STATUS_NO_ERROR);
    sts = hppDeleteInstance(accel);
    CV_Assert(sts==HPP_STATUS_NO_ERROR);
}

INSTANTIATE_TEST_CASE_P(IppATest, IPPAsync, Combine(Values(CV_8U, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                                                   Values(1, 2, 3, 4),
                                                   Values( HPP_ACCEL_TYPE_CPU, HPP_ACCEL_TYPE_GPU)));

}
}