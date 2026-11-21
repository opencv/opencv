/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace opencv_test {

// The importer requires Protobuf; when it is missing readNetFromSafetensors
// raises a clear error. Detect that case at runtime so the test skips cleanly.
static bool missingProtobuf(const Exception& e)
{
    return e.msg.find("Protobuf") != std::string::npos;
}

// Deterministic synthetic model exercising the F32 -> INT64 conversion paths:
//   x (1,1,6,6)
//   -> Slice (starts/ends/axes/steps as F32) -> (1,1,4,4)
//   -> Conv (W (2,1,3,3), B (2,))            -> (1,2,2,2)
//   -> Pad (pads as F32)                     -> (1,2,4,4)
//   -> Reshape (shape as F32)                -> (1,32)
//   -> Expand (shape as F32)                 -> (3,32)
//   -> Add (C (1,32), broadcast)             -> (3,32)
// Real export pipelines store integer metadata tensors as float32 inside the
// safetensors file; the importer must convert them to INT64.

static float sliceStarts[4] = {0, 0, 1, 1};
static float sliceEnds[4]   = {1, 1, 5, 5};
static float sliceAxes[4]   = {0, 1, 2, 3};
static float sliceSteps[4]  = {1, 1, 1, 1};
static float reshapeShape[2] = {1, 32};
static float expandShape[2]  = {3, 32};
static float padPads[8]     = {0, 0, 1, 1, 0, 0, 1, 1};

static float convWeightValue(int oc, int k, int l)
{
    return 0.1f * static_cast<float>((oc * 9 + k * 3 + l) % 7 - 3);
}

static float inputXValue(int i, int j)
{
    return 0.25f * static_cast<float>(i * 6 + j) - 4.f;
}

static std::string makeGraphJson()
{
    return R"({
  "model_name": "synthetic_safetensors_test",
  "inputs": [
    {"name": "x", "shape": [1, 1, 6, 6], "dtype": "float32"}
  ],
  "outputs": [
    {"name": "y", "shape": [3, 32], "dtype": "float32"}
  ],
  "nodes": [
    {"name": "slice.0", "op_type": "Slice",
     "inputs": ["x", "slice.starts", "slice.ends", "slice.axes", "slice.steps"],
     "outputs": ["sliced"], "attrs": {}},
    {"name": "conv.0", "op_type": "Conv",
     "inputs": ["sliced", "conv.weight", "conv.bias"],
     "outputs": ["convd"],
     "attrs": {"dilations": [1, 1], "kernel_shape": [3, 3],
               "strides": [1, 1], "group": 1, "pads": [0, 0, 0, 0]}},
    {"name": "pad.0", "op_type": "Pad",
     "inputs": ["convd", "pad.pads"],
     "outputs": ["padded"],
     "attrs": {"mode": "constant"}},
    {"name": "reshape.0", "op_type": "Reshape",
     "inputs": ["padded", "reshape.shape"],
     "outputs": ["reshaped"], "attrs": {"allowzero": 0}},
    {"name": "expand.0", "op_type": "Expand",
     "inputs": ["reshaped", "expand.shape"],
     "outputs": ["expanded"], "attrs": {}},
    {"name": "add.0", "op_type": "Add",
     "inputs": ["expanded", "add.bias"],
     "outputs": ["y"], "attrs": {}}
  ]
}
)";
}

// Appends one tensor record to the safetensors header and its raw bytes to data.
static void addTensor(std::ostringstream& header, std::vector<float>& data,
                      const std::string& name, const float* values, size_t count,
                      const std::vector<int64_t>& shape, uint64_t& offset)
{
    const uint64_t bytes = count * sizeof(float);
    if (offset > 0)
        header << ", ";
    header << "\"" << name << "\": {\"dtype\": \"F32\", \"shape\": [";
    for (size_t i = 0; i < shape.size(); ++i)
        header << (i ? ", " : "") << shape[i];
    header << "], \"data_offsets\": [" << offset << ", " << (offset + bytes) << "]}";
    data.insert(data.end(), values, values + count);
    offset += bytes;
}

static void writeSafetensorsModel(const std::string& weightsPath,
                                  const std::string& graphPath)
{
    float convBias[2] = {0.5f, 1.0f};
    float addBias[32];
    for (int j = 0; j < 32; ++j)
        addBias[j] = -0.02f * j;

    float convWeight[18];
    for (int oc = 0; oc < 2; ++oc)
        for (int k = 0; k < 3; ++k)
            for (int l = 0; l < 3; ++l)
                convWeight[oc * 9 + k * 3 + l] = convWeightValue(oc, k, l);

    std::ostringstream header;
    std::vector<float> data;
    uint64_t offset = 0;
    header << "{";
    addTensor(header, data, "conv.weight", convWeight, 18, {2, 1, 3, 3}, offset);
    addTensor(header, data, "conv.bias", convBias, 2, {2}, offset);
    addTensor(header, data, "add.bias", addBias, 32, {1, 32}, offset);
    addTensor(header, data, "slice.starts", sliceStarts, 4, {4}, offset);
    addTensor(header, data, "slice.ends", sliceEnds, 4, {4}, offset);
    addTensor(header, data, "slice.axes", sliceAxes, 4, {4}, offset);
    addTensor(header, data, "slice.steps", sliceSteps, 4, {4}, offset);
    addTensor(header, data, "reshape.shape", reshapeShape, 2, {2}, offset);
    addTensor(header, data, "expand.shape", expandShape, 2, {2}, offset);
    addTensor(header, data, "pad.pads", padPads, 8, {8}, offset);
    header << "}";
    std::string headerStr = header.str();
    // 8-byte align the header (as the reference implementation does)
    while (headerStr.size() % 8 != 0)
        headerStr += ' ';

    std::ofstream wf(weightsPath.c_str(), std::ios::binary);
    ASSERT_TRUE(wf.is_open());
    uint64_t headerLen = headerStr.size();
    wf.write(reinterpret_cast<const char*>(&headerLen), 8);
    wf.write(headerStr.data(), headerLen);
    wf.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    wf.close();
    ASSERT_TRUE(wf.good());

    std::ofstream gf(graphPath.c_str(), std::ios::binary);
    ASSERT_TRUE(gf.is_open());
    gf << makeGraphJson();
    gf.close();
    ASSERT_TRUE(gf.good());
}

// Reference result computed independently in C++.
static Mat referenceOutput(const Mat& x)
{
    const float* xp = x.ptr<float>();
    float sliced[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            sliced[i][j] = xp[(i + 1) * 6 + (j + 1)];

    float convd[2][2][2];
    for (int oc = 0; oc < 2; ++oc)
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
            {
                float s = (oc == 0) ? 0.5f : 1.0f;
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                        s += sliced[i + k][j + l] * convWeightValue(oc, k, l);
                convd[oc][i][j] = s;
            }

    // (1,2,2,2) padded with 1 on H and W -> (1,2,4,4), then flattened to 32
    float flat[32];
    int idx = 0;
    for (int oc = 0; oc < 2; ++oc)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j, ++idx)
                flat[idx] = (i >= 1 && i <= 2 && j >= 1 && j <= 2)
                                ? convd[oc][i - 1][j - 1] : 0.f;

    int sz[2] = {3, 32};
    Mat y(2, sz, CV_32F);
    for (int e = 0; e < 3; ++e)
        for (int j = 0; j < 32; ++j)
            y.at<float>(e, j) = flat[j] + (-0.02f * j);
    return y;
}

TEST(Test_Safetensors, synthetic_model)
{
    const std::string graphPath = cv::tempfile("opencv_test_safetensors_graph.json");
    const std::string weightsPath = cv::tempfile("opencv_test_safetensors_weights.safetensors");
    writeSafetensorsModel(weightsPath, graphPath);

    try
    {
        Net net = cv::dnn::readNetFromSafetensors(graphPath, weightsPath, cv::dnn::ENGINE_OPENCV);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        int sz[4] = {1, 1, 6, 6};
        Mat x(4, sz, CV_32F);
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                x.ptr<float>()[i * 6 + j] = inputXValue(i, j);

        net.setInput(x);
        Mat out = net.forward();
        Mat expected = referenceOutput(x);

        ASSERT_EQ(out.size, expected.size);
        double maxErr = 0.0;
        for (int e = 0; e < 3; ++e)
            for (int j = 0; j < 32; ++j)
            {
                double err = std::fabs(double(out.at<float>(e, j)) - double(expected.at<float>(e, j)));
                maxErr = std::max(maxErr, err);
            }
        EXPECT_LT(maxErr, 1e-4) << "max output error: " << maxErr;
    }
    catch (const Exception& e)
    {
        if (missingProtobuf(e))
            throw SkipTestException("OpenCV built without Protobuf");
        throw;
    }

    std::remove(graphPath.c_str());
    std::remove(weightsPath.c_str());
}

TEST(Test_Safetensors, corrupted_file_raises)
{
    const std::string graphPath = cv::tempfile("opencv_test_safetensors_bad_graph.json");
    const std::string weightsPath = cv::tempfile("opencv_test_safetensors_bad.safetensors");

    {
        std::ofstream wf(weightsPath.c_str(), std::ios::binary);
        uint64_t headerLen = 1000000;  // header claims more bytes than the file has
        wf.write(reinterpret_cast<const char*>(&headerLen), 8);
        wf.write("{}", 2);
    }
    {
        std::ofstream gf(graphPath.c_str(), std::ios::binary);
        gf << makeGraphJson();
    }

    try
    {
        EXPECT_ANY_THROW(cv::dnn::readNetFromSafetensors(graphPath, weightsPath, cv::dnn::ENGINE_OPENCV));
    }
    catch (const Exception& e)
    {
        if (missingProtobuf(e))
            throw SkipTestException("OpenCV built without Protobuf");
        throw;
    }

    std::remove(graphPath.c_str());
    std::remove(weightsPath.c_str());
}

TEST(Test_Safetensors, shape_span_mismatch_raises)
{
    const std::string graphPath = cv::tempfile("opencv_test_safetensors_mismatch_graph.json");
    const std::string weightsPath = cv::tempfile("opencv_test_safetensors_mismatch.safetensors");

    {
        // header declares 4 float32 elements (16 bytes) but the data span is 4 bytes
        std::ostringstream header;
        header << "{\"w\": {\"dtype\": \"F32\", \"shape\": [4], \"data_offsets\": [0, 4]}}";
        std::string headerStr = header.str();
        while (headerStr.size() % 8 != 0)
            headerStr += ' ';
        std::ofstream wf(weightsPath.c_str(), std::ios::binary);
        ASSERT_TRUE(wf.is_open());
        uint64_t headerLen = headerStr.size();
        wf.write(reinterpret_cast<const char*>(&headerLen), 8);
        wf.write(headerStr.data(), headerLen);
        const float dummy = 1.f;
        wf.write(reinterpret_cast<const char*>(&dummy), sizeof(dummy));
    }
    {
        std::ofstream gf(graphPath.c_str(), std::ios::binary);
        gf << makeGraphJson();
    }

    try
    {
        EXPECT_ANY_THROW(cv::dnn::readNetFromSafetensors(graphPath, weightsPath, cv::dnn::ENGINE_OPENCV));
    }
    catch (const Exception& e)
    {
        if (missingProtobuf(e))
            throw SkipTestException("OpenCV built without Protobuf");
        throw;
    }

    std::remove(graphPath.c_str());
    std::remove(weightsPath.c_str());
}

TEST(Test_Safetensors, unsupported_dtype_raises)
{
    const std::string graphPath = cv::tempfile("opencv_test_safetensors_dtype_graph.json");
    const std::string weightsPath = cv::tempfile("opencv_test_safetensors_dtype.safetensors");

    {
        std::ostringstream header;
        header << "{\"w\": {\"dtype\": \"COMPLEX64\", \"shape\": [1], \"data_offsets\": [0, 8]}}";
        std::string headerStr = header.str();
        while (headerStr.size() % 8 != 0)
            headerStr += ' ';
        std::ofstream wf(weightsPath.c_str(), std::ios::binary);
        ASSERT_TRUE(wf.is_open());
        uint64_t headerLen = headerStr.size();
        wf.write(reinterpret_cast<const char*>(&headerLen), 8);
        wf.write(headerStr.data(), headerLen);
        const uint64_t dummy = 0;
        wf.write(reinterpret_cast<const char*>(&dummy), sizeof(dummy));
    }
    {
        std::ofstream gf(graphPath.c_str(), std::ios::binary);
        gf << makeGraphJson();
    }

    try
    {
        EXPECT_ANY_THROW(cv::dnn::readNetFromSafetensors(graphPath, weightsPath, cv::dnn::ENGINE_OPENCV));
    }
    catch (const Exception& e)
    {
        if (missingProtobuf(e))
            throw SkipTestException("OpenCV built without Protobuf");
        throw;
    }

    std::remove(graphPath.c_str());
    std::remove(weightsPath.c_str());
}

TEST(Test_Safetensors, int64_output_interface)
{
    const std::string graphPath = cv::tempfile("opencv_test_safetensors_i64_graph.json");
    const std::string weightsPath = cv::tempfile("opencv_test_safetensors_i64.safetensors");

    {
        // empty weights file: header "{}" (8-byte aligned), no data section
        std::string headerStr = "{}";
        while (headerStr.size() % 8 != 0)
            headerStr += ' ';
        std::ofstream wf(weightsPath.c_str(), std::ios::binary);
        ASSERT_TRUE(wf.is_open());
        uint64_t headerLen = headerStr.size();
        wf.write(reinterpret_cast<const char*>(&headerLen), 8);
        wf.write(headerStr.data(), headerStr.size());
    }
    {
        // F32 input -> Cast to INT64, declared with a non-F32 interface dtype
        std::ofstream gf(graphPath.c_str(), std::ios::binary);
        ASSERT_TRUE(gf.is_open());
        gf << R"({
  "model_name": "int64_interface_test",
  "inputs":  [ {"name": "x", "shape": [1, 4], "dtype": "float32"} ],
  "outputs": [ {"name": "y", "shape": [1, 4], "dtype": "int64"} ],
  "nodes": [
    {"name": "cast.0", "op_type": "Cast", "inputs": ["x"], "outputs": ["y"], "attrs": {"to": 7}}
  ]
})";
    }

    try
    {
        Net net = cv::dnn::readNetFromSafetensors(graphPath, weightsPath, cv::dnn::ENGINE_OPENCV);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        int sz[2] = {1, 4};
        Mat x(2, sz, CV_32F);
        const float vals[4] = {1.7f, 2.2f, -3.9f, 4.1f};
        for (int i = 0; i < 4; ++i)
            x.ptr<float>()[i] = vals[i];

        net.setInput(x);
        Mat out = net.forward();
        ASSERT_EQ(out.total(), (size_t)4);
        ASSERT_TRUE(out.depth() == CV_64S || out.depth() == CV_32S)
            << "a non-F32 interface must produce an integer output Mat (depth=" << out.depth() << ")";

        // values are truncated toward zero by Cast: 1, 2, -3, 4
        const double expected[4] = {1.0, 2.0, -3.0, 4.0};
        for (int i = 0; i < 4; ++i)
        {
            const double v = (out.depth() == CV_64S) ? (double)out.ptr<int64_t>()[i]
                                                    : (double)out.ptr<int>()[i];
            EXPECT_EQ(v, expected[i]) << "element " << i;
        }
    }
    catch (const Exception& e)
    {
        if (missingProtobuf(e))
            throw SkipTestException("OpenCV built without Protobuf");
        throw;
    }

    std::remove(graphPath.c_str());
    std::remove(weightsPath.c_str());
}

TEST(Test_Safetensors, unsupported_interface_dtype_raises)
{
    const std::string graphPath = cv::tempfile("opencv_test_safetensors_iface_graph.json");
    const std::string weightsPath = cv::tempfile("opencv_test_safetensors_iface.safetensors");

    {
        std::string headerStr = "{}";
        while (headerStr.size() % 8 != 0)
            headerStr += ' ';
        std::ofstream wf(weightsPath.c_str(), std::ios::binary);
        uint64_t headerLen = headerStr.size();
        wf.write(reinterpret_cast<const char*>(&headerLen), 8);
        wf.write(headerStr.data(), headerStr.size());
    }
    {
        // replace the input dtype with an unsupported value
        std::string g = makeGraphJson();
        size_t pos = g.find("\"dtype\": \"float32\"");
        ASSERT_NE(pos, std::string::npos);
        g.replace(pos, 18, "\"dtype\": \"complex128\"");
        std::ofstream gf(graphPath.c_str(), std::ios::binary);
        gf << g;
    }

    try
    {
        EXPECT_ANY_THROW(cv::dnn::readNetFromSafetensors(graphPath, weightsPath, cv::dnn::ENGINE_OPENCV));
    }
    catch (const Exception& e)
    {
        if (missingProtobuf(e))
            throw SkipTestException("OpenCV built without Protobuf");
        throw;
    }

    std::remove(graphPath.c_str());
    std::remove(weightsPath.c_str());
}

}  // namespace opencv_test
