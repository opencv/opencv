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
    if (m.type() == CV_Bool)
        return m.at<bool>(indices);
    else if (m.type() == CV_8U)
        return m.at<uint8_t>(indices);
    else if (m.type() == CV_8S)
        return m.at<int8_t>(indices);
    else if (m.type() == CV_32S)
        return m.at<int32_t>(indices);
    else if (m.type() == CV_64S)
        return m.at<int64_t>(indices);
    else
        CV_Error(Error::BadDepth, "Unsupported type");
    return -1;
}

int64_t getValueAt(const Mat &m, int index)
{
    if (m.type() == CV_Bool)
        return m.ptr<bool>()[index];
    else if (m.type() == CV_8U)
        return m.ptr<uint8_t>()[index];
    else if (m.type() == CV_8S)
        return m.ptr<int8_t>()[index];
    else if (m.type() == CV_32S)
        return m.ptr<int32_t>()[index];
    else if (m.type() == CV_64S)
        return m.ptr<int64_t>()[index];
    else
        CV_Error(Error::BadDepth, "Unsupported type");
    return -1;
}

void fillRandom(Mat& m, int matType, Backend backend)
{
    if (matType == CV_64S && backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        cv::randu(m, 1000000000, 1000000100); // Looks like OpenVINO uses int32 internal values for int64 operations
    else if (matType == CV_64S)
        cv::randu(m, 1000000000000000ll, 1000000000000100ll);
    else if (matType == CV_32S)
        cv::randu(m, 1000000000, 1000000100);
    else if (matType == CV_8S)
        cv::randu(m, -50, 50);
    else if (matType == CV_8U)
        cv::randu(m, 0, 100);
    else if (matType == CV_Bool)
        cv::randu(m, 0, 2);
    else
        CV_Error(Error::BadDepth, "Unsupported type");
}

typedef testing::TestWithParam<tuple<int, tuple<Backend, Target> > > Test_NaryEltwise_Int;
TEST_P(Test_NaryEltwise_Int, random)
{
    int matType = get<0>(GetParam());
    tuple<Backend, Target> backend_target= get<1>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    std::vector<int> inShape{2, 3, 4, 5};
    Mat input1(inShape, matType);
    Mat input2(inShape, matType);
    fillRandom(input1, matType, backend);
    fillRandom(input2, matType, backend);

    Net net;
    LayerParams lp;
    lp.type = "NaryEltwise";
    lp.name = "testLayer";
    if (matType == CV_Bool)
        lp.set("operation", "or");
    else
        lp.set("operation", "add");
    int id = net.addLayerToPrev(lp.name, lp.type, lp);
    net.connect(0, 1, id, 1);

    vector<String> inpNames(2);
    inpNames[0] = "input1";
    inpNames[1] = "input2";
    net.setInputsNames(inpNames);
    net.setInput(input1, inpNames[0]);
    net.setInput(input2, inpNames[1]);

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), matType);
    EXPECT_EQ(re.size.dims(), 4);
    EXPECT_EQ(re.size[0], input1.size[0]);
    EXPECT_EQ(re.size[1], input1.size[1]);
    EXPECT_EQ(re.size[2], input1.size[2]);
    EXPECT_EQ(re.size[3], input1.size[3]);

    std::vector<int> reIndices(4);
    for (int i0 = 0; i0 < re.size[0]; ++i0)
    {
        reIndices[0] = i0;
        for (int i1 = 0; i1 < re.size[1]; ++i1)
        {
            reIndices[1] = i1;
            for (int i2 = 0; i2 < re.size[2]; ++i2)
            {
                reIndices[2] = i2;
                for (int i3 = 0; i3 < re.size[3]; ++i3)
                {
                    reIndices[3] = i3;
                    if (matType == CV_Bool)
                        EXPECT_EQ(getValueAt(re, reIndices.data()), getValueAt(input1, reIndices.data()) | getValueAt(input2, reIndices.data()));
                    else
                        EXPECT_EQ(getValueAt(re, reIndices.data()), getValueAt(input1, reIndices.data()) + getValueAt(input2, reIndices.data()));
                }
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_NaryEltwise_Int, Combine(
    testing::Values(CV_Bool, CV_8U, CV_8S, CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<int, tuple<Backend, Target> > > Test_Const_Int;
TEST_P(Test_Const_Int, random)
{
    int matType = get<0>(GetParam());
    tuple<Backend, Target> backend_target= get<1>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    std::vector<int> inShape{2, 3, 4, 5};
    Mat input1(inShape, matType);
    Mat inputConst(inShape, matType);
    fillRandom(input1, matType, backend);
    fillRandom(inputConst, matType, backend);

    Net net;

    LayerParams lpConst;
    lpConst.type = "Const";
    lpConst.name = "constLayer";
    lpConst.blobs.push_back(inputConst);
    int idConst = net.addLayer(lpConst.name, lpConst.type, lpConst);

    LayerParams lp;
    lp.type = "NaryEltwise";
    lp.name = "testLayer";
    if (matType == CV_Bool)
        lp.set("operation", "or");
    else
        lp.set("operation", "add");
    int idSum = net.addLayer(lp.name, lp.type, lp);

    net.connect(0, 0, idSum, 0);
    net.connect(idConst, 0, idSum, 1);

    net.setInput(input1);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), matType);
    EXPECT_EQ(re.size.dims(), 4);
    EXPECT_EQ(re.size[0], input1.size[0]);
    EXPECT_EQ(re.size[1], input1.size[1]);
    EXPECT_EQ(re.size[2], input1.size[2]);
    EXPECT_EQ(re.size[3], input1.size[3]);

    std::vector<int> reIndices(4);
    for (int i0 = 0; i0 < re.size[0]; ++i0)
    {
        reIndices[0] = i0;
        for (int i1 = 0; i1 < re.size[1]; ++i1)
        {
            reIndices[1] = i1;
            for (int i2 = 0; i2 < re.size[2]; ++i2)
            {
                reIndices[2] = i2;
                for (int i3 = 0; i3 < re.size[3]; ++i3)
                {
                    reIndices[3] = i3;
                    if (matType == CV_Bool)
                        EXPECT_EQ(getValueAt(re, reIndices.data()), getValueAt(input1, reIndices.data()) | getValueAt(inputConst, reIndices.data()));
                    else
                        EXPECT_EQ(getValueAt(re, reIndices.data()), getValueAt(input1, reIndices.data()) + getValueAt(inputConst, reIndices.data()));
                }
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Const_Int, Combine(
    testing::Values(CV_Bool, CV_8U, CV_8S, CV_32S, CV_64S),
    dnnBackendsAndTargets()
));


typedef testing::TestWithParam<tuple<int, int, tuple<Backend, Target> > > Test_ScatterND_Int;
TEST_P(Test_ScatterND_Int, random)
{
    int matType = get<0>(GetParam());
    int indicesType = get<1>(GetParam());
    tuple<Backend, Target> backend_target= get<2>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    std::vector<int> inShape{2, 3, 4, 5};
    Mat input(inShape, matType);
    fillRandom(input, matType, backend);

    std::vector<int64_t> indicesValues{0, 1, 2, 3,
                                       1, 2, 3, 4};
    std::vector<int64_t> updatesValues{25, 35};
    if (matType == CV_Bool)
    {
        updatesValues[0] = 1;
        updatesValues[1] = 0;
    }

    Mat indices(2, 4, indicesType);
    std::vector<int> updatesShape{2};
    Mat updates(updatesShape, matType);

    for (int i = 0; i < indicesValues.size(); ++i)
    {
        if (indicesType == CV_32S)
            indices.ptr<int32_t>()[i] = indicesValues[i];
        else
            indices.ptr<int64_t>()[i] = indicesValues[i];
    }

    for (int i = 0; i < updatesValues.size(); ++i)
    {
        if (matType == CV_32S)
            updates.ptr<int32_t>()[i] = updatesValues[i];
        else if (matType == CV_64S)
            updates.ptr<int64_t>()[i] = updatesValues[i];
        else if (matType == CV_8S)
            updates.ptr<int8_t>()[i] = updatesValues[i];
        else if (matType == CV_8U)
            updates.ptr<uint8_t>()[i] = updatesValues[i];
        else if (matType == CV_Bool)
            updates.ptr<bool>()[i] = updatesValues[i];
    }

    Net net;
    LayerParams lp;
    lp.type = "ScatterND";
    lp.name = "testLayer";
    int id = net.addLayerToPrev(lp.name, lp.type, lp);
    net.connect(0, 1, id, 1);
    net.connect(0, 2, id, 2);

    std::vector<String> inpNames(3);
    inpNames[0] = "scattedND_input";
    inpNames[1] = "scatterND_indices";
    inpNames[2] = "scatterND_updates";
    net.setInputsNames(inpNames);
    net.setInput(input, inpNames[0]);
    net.setInput(indices, inpNames[1]);
    net.setInput(updates, inpNames[2]);

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), matType);
    EXPECT_EQ(re.size.dims(), 4);
    ASSERT_EQ(shape(input), shape(re));

    std::vector<int> reIndices(4);
    for (int i0 = 0; i0 < input.size[0]; ++i0)
    {
        reIndices[0] = i0;
        for (int i1 = 0; i1 < input.size[1]; ++i1)
        {
            reIndices[1] = i1;
            for (int i2 = 0; i2 < input.size[2]; ++i2)
            {
                reIndices[2] = i2;
                for (int i3 = 0; i3 < input.size[3]; ++i3)
                {
                    reIndices[3] = i3;
                    if (reIndices[0] == indicesValues[0] &&
                        reIndices[1] == indicesValues[1] &&
                        reIndices[2] == indicesValues[2] &&
                        reIndices[3] == indicesValues[3])
                    {
                        EXPECT_EQ(getValueAt(re, reIndices.data()), updatesValues[0]);
                    }
                    else if (reIndices[0] == indicesValues[4] &&
                             reIndices[1] == indicesValues[5] &&
                             reIndices[2] == indicesValues[6] &&
                             reIndices[3] == indicesValues[7])
                    {
                        EXPECT_EQ(getValueAt(re, reIndices.data()), updatesValues[1]);
                    }
                    else
                    {
                        EXPECT_EQ(getValueAt(re, reIndices.data()), getValueAt(input, reIndices.data()));
                    }
                }
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_ScatterND_Int, Combine(
    testing::Values(CV_Bool, CV_8U, CV_8S, CV_32S, CV_64S),
    testing::Values(CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<int, tuple<Backend, Target> > > Test_Concat_Int;
TEST_P(Test_Concat_Int, random)
{
    int matType = get<0>(GetParam());
    tuple<Backend, Target> backend_target= get<1>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    std::vector<int> inShape1{2, 3, 4, 5};
    Mat input1(inShape1, matType);
    fillRandom(input1, matType, backend);
    std::vector<int> inShape2{2, 2, 4, 5};
    Mat input2(inShape2, matType);
    fillRandom(input2, matType, backend);

    Net net;
    LayerParams lp;
    lp.type = "Concat";
    lp.name = "testLayer";
    lp.set<int>("axis", 1);

    int id = net.addLayerToPrev(lp.name, lp.type, lp);
    net.connect(0, 1, id, 1);

    vector<String> inpNames(2);
    inpNames[0] = "input1";
    inpNames[1] = "input2";
    net.setInputsNames(inpNames);
    net.setInput(input1, inpNames[0]);
    net.setInput(input2, inpNames[1]);

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), matType);
    EXPECT_EQ(re.size.dims(), 4);
    EXPECT_EQ(re.size[0], input1.size[0]);
    EXPECT_EQ(re.size[1], input1.size[1] + input2.size[1]);
    EXPECT_EQ(re.size[2], input1.size[2]);
    EXPECT_EQ(re.size[3], input1.size[3]);

    std::vector<int> inIndices(4);
    std::vector<int> reIndices(4);
    for (int i0 = 0; i0 < re.size[0]; ++i0)
    {
        reIndices[0] = i0;
        inIndices[0] = i0;
        for (int i1 = 0; i1 < re.size[1]; ++i1)
        {
            reIndices[1] = i1;
            if (i1 < input1.size[1])
                inIndices[1] = i1;
            else
                inIndices[1] = i1 - input1.size[1];
            for (int i2 = 0; i2 < re.size[2]; ++i2)
            {
                reIndices[2] = i2;
                inIndices[2] = i2;
                for (int i3 = 0; i3 < re.size[3]; ++i3)
                {
                    reIndices[3] = i3;
                    inIndices[3] = i3;
                    if (i1 < input1.size[1])
                    {
                        EXPECT_EQ(getValueAt(re, reIndices.data()), getValueAt(input1, inIndices.data()));
                    }
                    else
                    {
                        EXPECT_EQ(getValueAt(re, reIndices.data()), getValueAt(input2, inIndices.data()));
                    }
                }
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Concat_Int, Combine(
    testing::Values(CV_Bool, CV_8U, CV_8S, CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<int, tuple<Backend, Target> > > Test_ArgMax_Int;
TEST_P(Test_ArgMax_Int, random)
{
    int matType = get<0>(GetParam());
    tuple<Backend, Target> backend_target= get<1>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH); // There is a problem with OpenVINO and custom int64 layers. After model compilation the output tensor type changes from int64 to int32

    std::vector<int> inShape{5, 4, 3, 2};
    Mat input(inShape, matType);
    fillRandom(input, matType, backend);

    Net net;
    LayerParams lp;
    lp.type = "Arg";
    lp.name = "testLayer";
    lp.set("op", "max");
    lp.set<int>("keepdims", 0);
    lp.set<int>("axis", 1);
    net.addLayerToPrev(lp.name, lp.type, lp);

    net.setInput(input);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), CV_64S);
    EXPECT_EQ(re.size.dims(), 3);
    EXPECT_EQ(re.size[0], inShape[0]);
    EXPECT_EQ(re.size[1], inShape[2]);
    EXPECT_EQ(re.size[2], inShape[3]);

    std::vector<int> inIndices(4);
    std::vector<int> reIndices(3);

    for (int i0 = 0; i0 < re.size[0]; ++i0)
    {
        inIndices[0] = i0;
        reIndices[0] = i0;
        for (int i1 = 0; i1 < re.size[1]; ++i1)
        {
            inIndices[2] = i1;
            reIndices[1] = i1;
            for (int i2 = 0; i2 < re.size[2]; ++i2)
            {
                inIndices[3] = i2;
                reIndices[2] = i2;

                int64_t max_value = -1000000000000000000l;
                int64_t index = 0;
                for (int j = 0; j < input.size[1]; ++j)
                {
                    inIndices[1] = j;
                    int64_t cur_value = getValueAt(input, inIndices.data());
                    if (cur_value > max_value)
                    {
                        max_value = cur_value;
                        index = j;
                    }
                }
                EXPECT_EQ(getValueAt(re, reIndices.data()), index);
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_ArgMax_Int, Combine(
    testing::Values(CV_8U, CV_8S, CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<int, tuple<Backend, Target> > > Test_Blank_Int;
TEST_P(Test_Blank_Int, random)
{
    int matType = get<0>(GetParam());
    tuple<Backend, Target> backend_target= get<1>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    std::vector<int> inShape{2, 3, 4, 5};
    Mat input(inShape, matType);
    fillRandom(input, matType, backend);

    Net net;
    LayerParams lp;
    lp.type = "Identity";
    lp.name = "testLayer";
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

    std::vector<int> reIndices(4);
    for (int i0 = 0; i0 < re.size[0]; ++i0)
    {
        reIndices[0] = i0;
        for (int i1 = 0; i1 < re.size[1]; ++i1)
        {
            reIndices[1] = i1;
            for (int i2 = 0; i2 < re.size[2]; ++i2)
            {
                reIndices[2] = i2;
                for (int i3 = 0; i3 < re.size[3]; ++i3)
                {
                    reIndices[3] = i3;
                    EXPECT_EQ(getValueAt(re, reIndices.data()), getValueAt(input, reIndices.data()));
                }
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Blank_Int, Combine(
    testing::Values(CV_Bool, CV_8U, CV_8S, CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<int, tuple<Backend, Target> > > Test_Expand_Int;
TEST_P(Test_Expand_Int, random)
{
    int matType = get<0>(GetParam());
    tuple<Backend, Target> backend_target= get<1>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    std::vector<int> inShape{2, 3, 1, 5};
    Mat input(inShape, matType);
    fillRandom(input, matType, backend);
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
    testing::Values(CV_Bool, CV_8U, CV_8S, CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<int, tuple<Backend, Target> > > Test_Permute_Int;
TEST_P(Test_Permute_Int, random)
{
    int matType = get<0>(GetParam());
    tuple<Backend, Target> backend_target= get<1>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    std::vector<int> inShape{2, 3, 4, 5};
    Mat input(inShape, matType);
    fillRandom(input, matType, backend);
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
    testing::Values(CV_Bool, CV_8U, CV_8S, CV_32S, CV_64S),
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
    Mat input(inShape, matType);
    fillRandom(input, matType, backend);

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
    testing::Values(CV_Bool, CV_8U, CV_8S, CV_32S, CV_64S),
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
    Mat input(inShape, matType);
    fillRandom(input, matType, backend);

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
    testing::Values(CV_Bool, CV_8U, CV_8S, CV_32S, CV_64S),
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
    if (inMatType == CV_Bool || outMatType == CV_Bool)
        cv::randu(input, 0, 1.1);
    else
        cv::randu(input, 0, 100);
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
    testing::Values(CV_Bool, CV_8U, CV_8S, CV_32S, CV_64S),
    testing::Values(CV_Bool, CV_8U, CV_8S, CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<int, tuple<Backend, Target> > > Test_Pad_Int;
TEST_P(Test_Pad_Int, random)
{
    int matType = get<0>(GetParam());
    tuple<Backend, Target> backend_target= get<1>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    std::vector<int> inShape{2, 3, 4, 5};
    Mat input(inShape, matType);
    fillRandom(input, matType, backend);
    std::vector<int> paddings{0, 0, 0, 0, 1, 0, 0, 1};
    int64_t padValue = matType == CV_Bool ? 1 : 25;

    Net net;
    LayerParams lp;
    lp.type = "Padding";
    lp.name = "testLayer";
    lp.set("paddings", DictValue::arrayInt<int*>(&paddings[0], paddings.size()));
    lp.set<double>("value", padValue);

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
    EXPECT_EQ(re.size[2], 5);
    EXPECT_EQ(re.size[3], 6);

    std::vector<int> reIndices(4);
    std::vector<int> inIndices(4);
    for (int i0 = 0; i0 < re.size[0]; ++i0)
    {
        reIndices[0] = i0;
        inIndices[0] = i0;
        for (int i1 = 0; i1 < re.size[1]; ++i1)
        {
            reIndices[1] = i1;
            inIndices[1] = i1;
            for (int i2 = 0; i2 < re.size[2]; ++i2)
            {
                reIndices[2] = i2;
                inIndices[2] = i2 - 1;
                for (int i3 = 0; i3 < re.size[3]; ++i3)
                {
                    reIndices[3] = i3;
                    inIndices[3] = i3;
                    if (i2 < 1 || i3 >= input.size[3])
                    {
                        EXPECT_EQ(getValueAt(re, reIndices.data()), padValue);
                    }
                    else
                    {
                        EXPECT_EQ(getValueAt(re, reIndices.data()), getValueAt(input, inIndices.data()));
                    }
                }
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Pad_Int, Combine(
    testing::Values(CV_Bool, CV_8U, CV_8S, CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<int, tuple<Backend, Target> > > Test_Slice_Int;
TEST_P(Test_Slice_Int, random)
{
    int matType = get<0>(GetParam());
    tuple<Backend, Target> backend_target= get<1>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    std::vector<int> inputShape{1, 16, 6, 8};
    std::vector<int> begin{0, 4, 0, 0};
    std::vector<int> end{1, 8, 6, 8};
    Mat input(inputShape, matType);
    fillRandom(input, matType, backend);

    std::vector<Range> range(4);
    for (int i = 0; i < 4; ++i)
        range[i] = Range(begin[i], end[i]);

    Net net;
    LayerParams lp;
    lp.type = "Slice";
    lp.name = "testLayer";
    lp.set("begin", DictValue::arrayInt<int*>(&(begin[0]), 4));
    lp.set("end", DictValue::arrayInt<int*>(&(end[0]), 4));
    net.addLayerToPrev(lp.name, lp.type, lp);

    net.setInput(input);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    Mat out = net.forward();

    Mat gt = input(range);
    EXPECT_EQ(out.size.dims(), 4);
    EXPECT_EQ(out.size[0], gt.size[0]);
    EXPECT_EQ(out.size[1], gt.size[1]);
    EXPECT_EQ(out.size[2], gt.size[2]);
    EXPECT_EQ(out.size[3], gt.size[3]);
    for (int i = 0; i < out.total(); ++i)
        EXPECT_EQ(getValueAt(out, i), getValueAt(gt, i));
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Slice_Int, Combine(
    testing::Values(CV_Bool, CV_8U, CV_8S, CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<int, tuple<Backend, Target> > > Test_Reshape_Int;
TEST_P(Test_Reshape_Int, random)
{
    int matType = get<0>(GetParam());
    tuple<Backend, Target> backend_target= get<1>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    std::vector<int> inShape{2, 3, 4, 5};
    std::vector<int> outShape{2, 3, 2, 10};
    Mat input(inShape, matType);
    fillRandom(input, matType, backend);

    Net net;
    LayerParams lp;
    lp.type = "Reshape";
    lp.name = "testLayer";
    lp.set("dim", DictValue::arrayInt<int*>(&outShape[0], outShape.size()));
    net.addLayerToPrev(lp.name, lp.type, lp);

    net.setInput(input);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), matType);
    EXPECT_EQ(re.size.dims(), 4);
    EXPECT_EQ(re.size[0], outShape[0]);
    EXPECT_EQ(re.size[1], outShape[1]);
    EXPECT_EQ(re.size[2], outShape[2]);
    EXPECT_EQ(re.size[3], outShape[3]);

    for (int i = 0; i < input.total(); ++i)
        EXPECT_EQ(getValueAt(re, i), getValueAt(input, i));
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Reshape_Int, Combine(
    testing::Values(CV_Bool, CV_8U, CV_8S, CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<int, tuple<Backend, Target> > > Test_Flatten_Int;
TEST_P(Test_Flatten_Int, random)
{
    int matType = get<0>(GetParam());
    tuple<Backend, Target> backend_target= get<1>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    std::vector<int> inShape{2, 3, 4, 5};
    Mat input(inShape, matType);
    fillRandom(input, matType, backend);

    Net net;
    LayerParams lp;
    lp.type = "Flatten";
    lp.name = "testLayer";
    lp.set("axis", 1);
    net.addLayerToPrev(lp.name, lp.type, lp);

    net.setInput(input);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), matType);
    EXPECT_EQ(re.size.dims(), 2);
    EXPECT_EQ(re.size[0], inShape[0]);
    EXPECT_EQ(re.size[1], inShape[1] * inShape[2] * inShape[3]);

    for (int i = 0; i < input.total(); ++i)
        EXPECT_EQ(getValueAt(re, i), getValueAt(input, i));
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Flatten_Int, Combine(
    testing::Values(CV_Bool, CV_8U, CV_8S, CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<int, tuple<Backend, Target> > > Test_Tile_Int;
TEST_P(Test_Tile_Int, random)
{
    int matType = get<0>(GetParam());
    tuple<Backend, Target> backend_target= get<1>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    std::vector<int> inShape{2, 3, 4, 5};
    Mat input(inShape, matType);
    fillRandom(input, matType, backend);
    std::vector<int> repeats{1, 1, 2, 3};

    Net net;
    LayerParams lp;
    lp.type = "Tile";
    lp.name = "testLayer";
    lp.set("repeats", DictValue::arrayInt<int*>(repeats.data(), repeats.size()));
    net.addLayerToPrev(lp.name, lp.type, lp);

    net.setInput(input);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), matType);
    EXPECT_EQ(re.size.dims(), 4);
    EXPECT_EQ(re.size[0], inShape[0] * repeats[0]);
    EXPECT_EQ(re.size[1], inShape[1] * repeats[1]);
    EXPECT_EQ(re.size[2], inShape[2] * repeats[2]);
    EXPECT_EQ(re.size[3], inShape[3] * repeats[3]);

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

INSTANTIATE_TEST_CASE_P(/**/, Test_Tile_Int, Combine(
    testing::Values(CV_Bool, CV_8U, CV_8S, CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

typedef testing::TestWithParam<tuple<int, tuple<Backend, Target> > > Test_Reduce_Int;
TEST_P(Test_Reduce_Int, random)
{
    int matType = get<0>(GetParam());
    tuple<Backend, Target> backend_target= get<1>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && matType == CV_64S)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH); // There is a problem with OpenVINO and custom int64 layers. After model compilation the output tensor type changes from int64 to int32

    std::vector<int> inShape{5, 4, 3, 2};
    Mat input(inShape, matType);
    if (matType == CV_64S && backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        cv::randu(input, 100000000, 100000100); // Looks like OpenVINO uses int32 internal values for int64 operations
    else if (matType == CV_64S)
        cv::randu(input, 1000000000000000ll, 1000000000000100ll);
    else if (matType == CV_32S)
        cv::randu(input, 100000000, 100000100);
    else if (matType == CV_8S)
        cv::randu(input, -25, 25);
    else if (matType == CV_8U)
        cv::randu(input, 0, 50);
    else
        CV_Error(Error::BadDepth, "Unsupported type");

    std::vector<int> axes{1};
    Net net;

    LayerParams lp;
    lp.type = "Reduce";
    lp.name = "testLayer";
    lp.set("reduce", "SUM");
    lp.set("keepdims", false);
    lp.set("axes", DictValue::arrayInt<int*>(axes.data(), axes.size()));
    net.addLayerToPrev(lp.name, lp.type, lp);

    net.setInput(input);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), matType);
    EXPECT_EQ(re.size.dims(), 3);
    EXPECT_EQ(re.size[0], inShape[0]);
    EXPECT_EQ(re.size[1], inShape[2]);
    EXPECT_EQ(re.size[2], inShape[3]);

    std::vector<int> inIndices(4);
    std::vector<int> reIndices(3);

    for (int i0 = 0; i0 < re.size[0]; ++i0)
    {
        inIndices[0] = i0;
        reIndices[0] = i0;
        for (int i1 = 0; i1 < re.size[1]; ++i1)
        {
            inIndices[2] = i1;
            reIndices[1] = i1;
            for (int i2 = 0; i2 < re.size[2]; ++i2)
            {
                inIndices[3] = i2;
                reIndices[2] = i2;

                int64_t value = 0;
                for (int j = 0; j < input.size[1]; ++j)
                {
                    inIndices[1] = j;
                    value += getValueAt(input, inIndices.data());
                }
                EXPECT_EQ(getValueAt(re, reIndices.data()), value);
            }
        }
    }
}

typedef testing::TestWithParam<tuple<int, tuple<Backend, Target> > > Test_Reduce_Int;
TEST_P(Test_Reduce_Int, two_axes)
{
    int matType = get<0>(GetParam());
    tuple<Backend, Target> backend_target= get<1>(GetParam());
    Backend backend = get<0>(backend_target);
    Target target = get<1>(backend_target);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && matType == CV_64S)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH); // There is a problem with OpenVINO and custom int64 layers. After model compilation the output tensor type changes from int64 to int32

    std::vector<int> inShape{5, 4, 3, 2};
    Mat input(inShape, matType);
    if (matType == CV_64S && backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        cv::randu(input, 100000000, 100000100); // Looks like OpenVINO uses int32 internal values for int64 operations
    else if (matType == CV_64S)
        cv::randu(input, 1000000000000000ll, 1000000000000100ll);
    else if (matType == CV_32S)
        cv::randu(input, 100000000, 100000100);
    else if (matType == CV_8S)
        cv::randu(input, -15, 15);
    else if (matType == CV_8U)
        cv::randu(input, 0, 30);
    else
        CV_Error(Error::BadDepth, "Unsupported type");

    std::vector<int> axes{1, 3};

    Net net;
    LayerParams lp;
    lp.type = "Reduce";
    lp.name = "testLayer";
    lp.set("reduce", "SUM");
    lp.set("keepdims", false);
    lp.set("axes", DictValue::arrayInt<int*>(axes.data(), axes.size()));
    net.addLayerToPrev(lp.name, lp.type, lp);

    net.setInput(input);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), matType);
    EXPECT_EQ(re.size.dims(), 2);
    EXPECT_EQ(re.size[0], inShape[0]);
    EXPECT_EQ(re.size[1], inShape[2]);

    std::vector<int> inIndices(4);
    std::vector<int> reIndices(2);

    for (int i0 = 0; i0 < re.size[0]; ++i0)
    {
        inIndices[0] = i0;
        reIndices[0] = i0;
        for (int i1 = 0; i1 < re.size[1]; ++i1)
        {
            inIndices[2] = i1;
            reIndices[1] = i1;
            int64_t value = 0;
            for (int i2 = 0; i2 < input.size[3]; ++i2)
            {
                inIndices[3] = i2;

                for (int j = 0; j < input.size[1]; ++j)
                {
                    inIndices[1] = j;
                    value += getValueAt(input, inIndices.data());
                }
            }
            EXPECT_EQ(getValueAt(re, reIndices.data()), value);
        }
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Reduce_Int, Combine(
    testing::Values(CV_8U, CV_8S, CV_32S, CV_64S),
    dnnBackendsAndTargets()
));

}} // namespace
