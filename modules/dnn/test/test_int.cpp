// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace opencv_test { namespace {

int64_t getValueAt(const Mat &m, const int *indices)
{
    if (m.type() == CV_32S)
        return m.at<int32_t>(indices);
    else if (m.type() == CV_64S)
        return m.at<int64_t>(indices);
    else
        CV_Error(Error::BadDepth, "Unsupported type");
    return -1;
}

typedef testing::TestWithParam<tuple<Backend, Target> > Test_int64_sum;
TEST_P(Test_int64_sum, basic)
{
    Backend backend = get<0>(GetParam());
    Target target = get<1>(GetParam());

    int64_t a_value = 1000000000000000ll;
    int64_t b_value = 1;
    int64_t result_value = 1000000000000001ll;
    EXPECT_NE(int64_t(float(a_value) + float(b_value)), result_value);

    Mat a(3, 5, CV_64SC1, cv::Scalar_<int64_t>(a_value));
    Mat b = Mat::ones(3, 5, CV_64S);

    Net net;
    LayerParams lp;
    lp.type = "NaryEltwise";
    lp.name = "testLayer";
    lp.set("operation", "sum");
    int id = net.addLayerToPrev(lp.name, lp.type, lp);
    net.connect(0, 1, id, 1);

    vector<String> inpNames(2);
    inpNames[0] = "a";
    inpNames[1] = "b";
    net.setInputsNames(inpNames);
    net.setInput(a, inpNames[0]);
    net.setInput(b, inpNames[1]);

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), CV_64S);
    auto ptr_re = (int64_t *) re.data;
    for (int i = 0; i < re.total(); i++)
        ASSERT_EQ(result_value, ptr_re[i]);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Test_int64_sum,
    dnnBackendsAndTargets()
);

typedef testing::TestWithParam<tuple<int, tuple<Backend, Target> > > Test_Expand_Int;
TEST_P(Test_Expand_Int, random)
{
    int matType = get<0>(GetParam());
    tuple<Backend, Target> backend_target= get<1>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    std::vector<int> inShape{2, 3, 1, 5};
    int64_t low = matType == CV_64S ? 1000000000000000ll : 1000000000;
    Mat input(inShape, matType);
    cv::randu(input, low, low + 100);
    std::vector<int> outShape{2, 1, 4, 5};

    Net net;
    LayerParams lp;
    lp.type = "Expand";
    lp.name = "testLayer";
    lp.set("shape", DictValue::arrayInt<int*>(&outShape[0], outShape.size()));
    net.addLayerToPrev(lp.name, lp.type, lp);

    net.setInput(input);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), matType);
    EXPECT_EQ(re.size.dims(), 4);
    EXPECT_EQ(re.size[0], 2);
    EXPECT_EQ(re.size[1], 3);
    EXPECT_EQ(re.size[2], 4);
    EXPECT_EQ(re.size[3], 5);

    std::vector<int> inIndices(4);
    std::vector<int> reIndices(4);
    for (int i0 = 0; i0 < re.size[0]; ++i0)
    {
        inIndices[0] = i0 % inShape[0];
        reIndices[0] = i0;
        for (int i1 = 0; i1 < re.size[1]; ++i1)
        {
            inIndices[1] = i1 % inShape[1];
            reIndices[1] = i1;
            for (int i2 = 0; i2 < re.size[2]; ++i2)
            {
                inIndices[2] = i2 % inShape[2];
                reIndices[2] = i2;
                for (int i3 = 0; i3 < re.size[3]; ++i3)
                {
                    inIndices[3] = i3 % inShape[3];
                    reIndices[3] = i3;
                    EXPECT_EQ(getValueAt(re, reIndices.data()), getValueAt(input, inIndices.data()));
                }
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Expand_Int, Combine(
    testing::Values(CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<int, tuple<Backend, Target> > > Test_Permute_Int;
TEST_P(Test_Permute_Int, random)
{
    int matType = get<0>(GetParam());
    tuple<Backend, Target> backend_target= get<1>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    if(backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA);

    std::vector<int> inShape{2, 3, 4, 5};
    int64_t low = matType == CV_64S ? 1000000000000000ll : 1000000000;
    Mat input(inShape, matType);
    cv::randu(input, low, low + 100);
    std::vector<int> order{0, 2, 3, 1};

    Net net;
    LayerParams lp;
    lp.type = "Permute";
    lp.name = "testLayer";
    lp.set("order", DictValue::arrayInt<int*>(&order[0], order.size()));
    net.addLayerToPrev(lp.name, lp.type, lp);

    net.setInput(input);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), matType);
    EXPECT_EQ(re.size.dims(), 4);
    EXPECT_EQ(re.size[0], 2);
    EXPECT_EQ(re.size[1], 4);
    EXPECT_EQ(re.size[2], 5);
    EXPECT_EQ(re.size[3], 3);

    std::vector<int> inIndices(4);
    std::vector<int> reIndices(4);
    for (int i0 = 0; i0 < input.size[0]; ++i0)
    {
        inIndices[0] = i0;
        reIndices[0] = i0;
        for (int i1 = 0; i1 < input.size[1]; ++i1)
        {
            inIndices[1] = i1;
            reIndices[3] = i1;
            for (int i2 = 0; i2 < input.size[2]; ++i2)
            {
                inIndices[2] = i2;
                reIndices[1] = i2;
                for (int i3 = 0; i3 < input.size[3]; ++i3)
                {
                    inIndices[3] = i3;
                    reIndices[2] = i3;
                    EXPECT_EQ(getValueAt(re, reIndices.data()), getValueAt(input, inIndices.data()));
                }
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Permute_Int, Combine(
    testing::Values(CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<int, int, tuple<Backend, Target> > > Test_GatherElements_Int;
TEST_P(Test_GatherElements_Int, random)
{
    int matType = get<0>(GetParam());
    int indicesType = get<1>(GetParam());
    tuple<Backend, Target> backend_target= get<2>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    std::vector<int> inShape{2, 3, 4, 5};
    int64_t low = matType == CV_64S ? 1000000000000000ll : 1000000000;
    Mat input(inShape, matType);
    cv::randu(input, low, low + 100);

    std::vector<int> indicesShape{2, 3, 10, 5};
    Mat indicesMat(indicesShape, indicesType);
    cv::randu(indicesMat, 0, 4);

    Net net;
    LayerParams lp;
    lp.type = "GatherElements";
    lp.name = "testLayer";
    lp.set("axis", 2);
    int id = net.addLayerToPrev(lp.name, lp.type, lp);
    net.connect(0, 1, id, 1);

    std::vector<String> inpNames(2);
    inpNames[0] = "gather_input";
    inpNames[1] = "gather_indices";
    net.setInputsNames(inpNames);
    net.setInput(input, inpNames[0]);
    net.setInput(indicesMat, inpNames[1]);

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), matType);
    EXPECT_EQ(re.size.dims(), 4);
    ASSERT_EQ(shape(indicesMat), shape(re));

    std::vector<int> inIndices(4);
    std::vector<int> reIndices(4);
    for (int i0 = 0; i0 < input.size[0]; ++i0)
    {
        inIndices[0] = i0;
        reIndices[0] = i0;
        for (int i1 = 0; i1 < input.size[1]; ++i1)
        {
            inIndices[1] = i1;
            reIndices[1] = i1;
            for (int i2 = 0; i2 < indicesMat.size[2]; ++i2)
            {
                reIndices[2] = i2;
                for (int i3 = 0; i3 < input.size[3]; ++i3)
                {
                    inIndices[3] = i3;
                    reIndices[3] = i3;
                    inIndices[2] = getValueAt(indicesMat, reIndices.data());
                    EXPECT_EQ(getValueAt(re, reIndices.data()), getValueAt(input, inIndices.data()));
                }
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_GatherElements_Int, Combine(
    testing::Values(CV_32S, CV_64S),
    testing::Values(CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<int, int, tuple<Backend, Target> > > Test_Gather_Int;
TEST_P(Test_Gather_Int, random)
{
    int matType = get<0>(GetParam());
    int indicesType = get<1>(GetParam());
    tuple<Backend, Target> backend_target= get<2>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    std::vector<int> inShape{5, 1};
    int64_t low = matType == CV_64S ? 1000000000000000ll : 1000000000;
    Mat input(inShape, matType);
    cv::randu(input, low, low + 100);

    std::vector<int> indices_shape = {1, 1};
    Mat indicesMat = cv::Mat(indices_shape, indicesType, 0.0);

    std::vector<int> output_shape = {5, 1};
    cv::Mat outputRef = cv::Mat(output_shape, matType, input(cv::Range::all(), cv::Range(0, 1)).data);

    Net net;
    LayerParams lp;
    lp.type = "Gather";
    lp.name = "testLayer";
    lp.set("axis", 1);
    lp.set("real_ndims", 1);
    int id = net.addLayerToPrev(lp.name, lp.type, lp);
    net.connect(0, 1, id, 1);

    std::vector<String> inpNames(2);
    inpNames[0] = "gather_input";
    inpNames[1] = "gather_indices";
    net.setInputsNames(inpNames);
    net.setInput(input, inpNames[0]);
    net.setInput(indicesMat, inpNames[1]);

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), matType);

    ASSERT_EQ(shape(outputRef), shape(re));
    normAssert(outputRef, re);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Gather_Int, Combine(
    testing::Values(CV_32S, CV_64S),
    testing::Values(CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<int, int, tuple<Backend, Target> > > Test_Cast_Int;
TEST_P(Test_Cast_Int, random)
{
    int inMatType = get<0>(GetParam());
    int outMatType = get<1>(GetParam());
    tuple<Backend, Target> backend_target= get<2>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    std::vector<int> inShape{2, 3, 4, 5};
    Mat input(inShape, inMatType);
    cv::randu(input, 200, 300);
    Mat outputRef;
    input.convertTo(outputRef, outMatType);

    Net net;
    LayerParams lp;
    lp.type = "Cast";
    lp.name = "testLayer";
    lp.set("outputType", outMatType);
    net.addLayerToPrev(lp.name, lp.type, lp);

    net.setInput(input);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), outMatType);
    EXPECT_EQ(re.size.dims(), 4);

    ASSERT_EQ(shape(input), shape(re));
    normAssert(outputRef, re);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Cast_Int, Combine(
    testing::Values(CV_32S, CV_64S),
    testing::Values(CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

}} // namespace
