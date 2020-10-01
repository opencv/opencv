/// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

// This sample tests AzureEye with a single detection network.
// Every frame received from FW is processed. No H264 stream produced.

#include <algorithm>
#include <iostream>
#include <sstream>
#include <map>

#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/render.hpp>

#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include "../test_precomp.hpp"
#include <opencv2/gapi/infer/ie.hpp>

namespace opencv_test {
namespace custom {

using GDetections = cv::GArray<cv::Rect>;
using GPrims      = cv::GArray<cv::gapi::wip::draw::Prim>;

G_API_OP(BBoxes, <GPrims(GDetections)>, "sample.custom.b-boxes") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc &) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVBBoxes, BBoxes) {
    // Converts the rectangles into G-API's rendering primitives
    static void run(const std::vector<cv::Rect> &in_face_rcs,
                          std::vector<cv::gapi::wip::draw::Prim> &out_prims) {
        out_prims.clear();
        const auto cvt = [](const cv::Rect &rc, const cv::Scalar &clr) {
            return cv::gapi::wip::draw::Rect(rc, clr, 2);
        };
        for (auto &&rc : in_face_rcs) {
            out_prims.emplace_back(cvt(rc, CV_RGB(0,255,0)));   // green
        }
    }
};

G_API_OP(GParseSSD, <cv::GArray<int>(GMat)>, "org.opencv.dnn.parseSSD") {
    static GArrayDesc outMeta(const GMatDesc&) {
        return empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVParseSSD, GParseSSD) {
    static void run(const Mat& in_ssd_result, std::vector<int>& output) {
        output.clear();
        output.resize(5);
        struct Item {int idx; float conf;};
        std::vector<Item> tmp(1000);
        for (int i = 0; i < 1000; i++) {
            tmp[i] = Item{i, reinterpret_cast<const float*>(in_ssd_result.data)[i]};
        }
        std::sort(tmp.begin(), tmp.end(), [](const Item& i1, const Item& i2) {
            return i1.conf < i2.conf;
        });
        for (int i = 0; i < 5; i++) {
            std::cout << tmp[1000-1-i].conf << ", " << tmp[1000-1-i].idx << std::endl;
        }
    }
};
} // namespace custom

G_API_NET(ObjectDetector, <cv::GMat(cv::GMat)>, "com.intel.azure.object-detector");

TEST(TestMobileNetIE, InferBasicImage) {
    // FIXME: initTestDataPath
    auto detector = cv::gapi::ie::Params<ObjectDetector>("/data/rgarnov/mobilenet.blob", {}, "KMB", cv::gapi::ie::detail::ParamDesc::Kind::Import );
    auto networks = cv::gapi::networks(detector);

    auto kernels = cv::gapi::kernels<custom::OCVBBoxes, custom::OCVParseSSD>();

    // Now build the graph. The graph structure may vary
    // based on the input parameters

    // Graph construction //////////////////////////////////////////////////////

    // Declare the pipeline inputs
    cv::GMat in;                  // marks start of the pipeline (Camera for us)

    // Run Inference on the full frame
    auto blob = cv::gapi::infer<ObjectDetector>(in);

    std::string input = "/data/rgarnov/testdata/dnn/grace_hopper_227.png";
    cv::Mat in_mat = cv::imread(input);

    // Parse the detections and project those to the original image frame
    auto tags = custom::GParseSSD::on(blob);

    // Draw bounding boxes and ROI on the BGR frame
//    auto rendered = cv::gapi::wip::draw::render3ch(in, custom::BBoxes::on(objs));
    auto graph_outs = cv::GOut(tags);

    // Graph compilation ///////////////////////////////////////////////////////
    auto pipeline = cv::GComputation(cv::GIn(in), std::move(graph_outs))
        .compileStreaming(cv::descr_of(in_mat),
                          cv::compile_args(kernels, networks));

    // Graph execution /////////////////////////////////////////////////////////

    pipeline.setSource(cv::gin(in_mat));
    pipeline.start();

    std::vector<int>      out_labels;
    auto pipeline_outputs = cv::gout(out_labels);

    (pipeline.pull(std::move(pipeline_outputs))); {
        std::cout << "pull. Got " << out_labels.size() << std::endl;
    }
}
} // namespace opencv_test
