#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_OPENCL

namespace cvtest
{
namespace ocl
{

using std::tr1::tuple;
using std::tr1::get;
using std::tr1::make_tuple;
using std::make_pair;
using namespace perf;
using namespace testing;
using namespace cv;
using namespace cv::dnn;

enum {STRIDE_OFF = 1, STRIDE_ON = 2};
CV_ENUM(StrideSize, STRIDE_OFF, STRIDE_ON);

enum {GROUP_OFF = 1, GROUP_2 = 2};
CV_ENUM(GroupSize, GROUP_OFF, GROUP_2);

//Squared Size
#define SSZ(n) cv::Size(n, n)

typedef std::pair<MatShape, int> InpShapeNumOut;
typedef tuple<Size, InpShapeNumOut, GroupSize, StrideSize> ConvParam; //kernel_size, inp shape, groups, stride
typedef TestBaseWithParam<ConvParam> ConvolutionPerfTest;

static inline MatShape blobShape(int count, int nplanes, int height, int width)
{
    int data[] = {count, nplanes, height, width};
    return MatShape(data, data+4);
}

OCL_PERF_TEST_P( ConvolutionPerfTest, perf, Combine(
    Values(Size(1, 1), Size(3, 3), Size(5, 5), Size(11, 11)),
    Values(make_pair(blobShape(1,   4, 224, 224),  64),
           make_pair(blobShape(1,  64, 112, 122), 128),
           make_pair(blobShape(1, 256,  28,  28), 512)),
    GroupSize::all(),
    StrideSize::all())
)
{
    RNG rng(0);

    ConvParam params = GetParam();
    int ksz     = get<0>(params).width;
    MatShape inpShape = get<1>(params).first;
    int outCn   = get<1>(params).second;
    int groups  = get<2>(params);
    int stride  = (ksz >= 11) ? 4 : (int)get<3>(params);

    int inpCn = inpShape[1];
    int wgtSize[] = { outCn, inpCn/groups, ksz, ksz };
    int biasSize[] = { outCn, 1, 1, 1 };
    const int wtype = CV_32F;
    Mat wgtBlob(4, wgtSize, wtype), biasBlob(4, biasSize, wtype);
    Mat inpBlob(4, &inpShape[0], wtype);
    rng.fill(biasBlob, RNG::UNIFORM, -1, +1);
    rng.fill(wgtBlob, RNG::UNIFORM, -1, +1);
    rng.fill(inpBlob, RNG::UNIFORM, -1, +1);

    LayerParams lp;
    lp.set("num_output", outCn);
    lp.set("group", groups);
    lp.set("stride", stride);
    lp.set("kernel_size", ksz);
    lp.blobs.reserve(2);
    lp.blobs.push_back(wgtBlob);
    lp.blobs.push_back(biasBlob);

    std::vector<Mat*> inpBlobs(1, &inpBlob);
    std::vector<Mat> outBlobs, internalBlobs;

    cv::setNumThreads(cv::getNumberOfCPUs());

    Ptr<Layer> layer = cv::dnn::LayerFactory::createLayerInstance("Convolution", lp);
    std::vector<MatShape> inputShapes(1, shape(inpBlob)), outShapes, internals;
    layer->getMemoryShapes(inputShapes, 0, outShapes, internals);
    for (int i = 0; i < outShapes.size(); i++)
    {
        outBlobs.push_back(Mat(outShapes[i], CV_32F));
    }
    for (int i = 0; i < internals.size(); i++)
    {
        internalBlobs.push_back(Mat());
        if (total(internals[i]))
            internalBlobs.back().create(internals[i], CV_32F);
    }

    layer->finalize(inpBlobs, outBlobs);
    layer->preferableTarget = DNN_TARGET_OPENCL;

    Mat inpBlob2D = inpBlob.reshape(1, outCn);
    Mat wgtBlob2D = wgtBlob.reshape(1, outCn*(inpCn/groups));
    Mat outBlob2D = outBlobs[0].reshape(1, outBlobs[0].size[0]);
    declare.in(inpBlob2D, wgtBlob2D, WARMUP_RNG).out(outBlob2D).tbb_threads(cv::getNumThreads());

    // warmup
    layer->forward(inpBlobs, outBlobs, internalBlobs);

    TEST_CYCLE()
    {
        layer->forward(inpBlobs, outBlobs, internalBlobs);
    }

    SANITY_CHECK_NOTHING();
}

}
}

#endif
