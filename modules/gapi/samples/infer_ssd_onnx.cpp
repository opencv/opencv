#include <algorithm>
#include <iostream>
#include <sstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/render.hpp>
#include <opencv2/gapi/infer/onnx.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/highgui.hpp>

namespace custom {

G_API_NET(ObjDetector,   <cv::GMat(cv::GMat)>, "object-detector");

using GDetections = cv::GArray<cv::Rect>;
using GSize       = cv::GOpaque<cv::Size>;
using GPrims      = cv::GArray<cv::gapi::wip::draw::Prim>;

G_API_OP(GetSize, <GSize(cv::GMat)>, "sample.custom.get-size") {
    static cv::GOpaqueDesc outMeta(const cv::GMatDesc &) {
        return cv::empty_gopaque_desc();
    }
};
G_API_OP(ParseSSD, <GDetections(cv::GMat, GSize)>, "sample.custom.parse-ssd") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GOpaqueDesc &) {
        return cv::empty_array_desc();
    }
};
G_API_OP(BBoxes, <GPrims(GDetections)>, "sample.custom.b-boxes") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc &) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVGetSize, GetSize) {
    static void run(const cv::Mat &in, cv::Size &out) {
        out = {in.cols, in.rows};
    }
};
GAPI_OCV_KERNEL(OCVParseSSD, ParseSSD) {
    static void run(const cv::Mat &in_ssd_result,
                    const cv::Size &in_parent_size,
                    std::vector<cv::Rect> &out_objects) {
        const auto &in_ssd_dims = in_ssd_result.size;
        CV_Assert(in_ssd_dims.dims() == 4u);

        const int MAX_PROPOSALS = in_ssd_dims[2];
        const int OBJECT_SIZE   = in_ssd_dims[3];

        CV_Assert(OBJECT_SIZE  == 7); // fixed SSD object size

        const cv::Rect surface({0,0}, in_parent_size);

        out_objects.clear();

        const float *data = in_ssd_result.ptr<float>();
        for (int i = 0; i < MAX_PROPOSALS; i++) {
            const float image_id   = data[i * OBJECT_SIZE + 0];
            const float label      = data[i * OBJECT_SIZE + 1];
            const float confidence = data[i * OBJECT_SIZE + 2];
            const float rc_left    = data[i * OBJECT_SIZE + 3];
            const float rc_top     = data[i * OBJECT_SIZE + 4];
            const float rc_right   = data[i * OBJECT_SIZE + 5];
            const float rc_bottom  = data[i * OBJECT_SIZE + 6];
            (void) label; // unused

            if (image_id < 0.f) {
                break;    // marks end-of-detections
            }
            if (confidence < 0.5f) {
                continue; // skip objects with low confidence
            }

            // map relative coordinates to the original image scale
            cv::Rect rc;
            rc.x      = static_cast<int>(rc_left   * in_parent_size.width);
            rc.y      = static_cast<int>(rc_top    * in_parent_size.height);
            rc.width  = static_cast<int>(rc_right  * in_parent_size.width)  - rc.x;
            rc.height = static_cast<int>(rc_bottom * in_parent_size.height) - rc.y;
            out_objects.emplace_back(rc & surface);
        }
    }
};
GAPI_OCV_KERNEL(OCVBBoxes, BBoxes) {
    // This kernel converts the rectangles into G-API's
    // rendering primitives
    static void run(const std::vector<cv::Rect> &in_obj_rcs,
                          std::vector<cv::gapi::wip::draw::Prim> &out_prims) {
        out_prims.clear();
        const auto cvt = [](const cv::Rect &rc, const cv::Scalar &clr) {
            return cv::gapi::wip::draw::Rect(rc, clr, 2);
        };
        for (auto &&rc : in_obj_rcs) {
            out_prims.emplace_back(cvt(rc, CV_RGB(0,255,0)));   // green
        }

        std::cout << "Detections:";
        for (auto &&rc : in_obj_rcs) std::cout << ' ' << rc;
        std::cout << std::endl;
    }
};

} // namespace custom

namespace {
void remap_ssd_ports(const std::unordered_map<std::string, cv::Mat> &onnx,
                           std::unordered_map<std::string, cv::Mat> &gapi) {
    // Assemble ONNX-processed outputs back to a single 1x1x200x7 blob
    // to preserve compatibility with OpenVINO-based SSD pipeline
    const cv::Mat &num_detections = onnx.at("num_detections:0");
    const cv::Mat &detection_boxes = onnx.at("detection_boxes:0");
    const cv::Mat &detection_scores = onnx.at("detection_scores:0");
    const cv::Mat &detection_classes = onnx.at("detection_classes:0");

    GAPI_Assert(num_detections.depth() == CV_32F);
    GAPI_Assert(detection_boxes.depth() == CV_32F);
    GAPI_Assert(detection_scores.depth() == CV_32F);
    GAPI_Assert(detection_classes.depth() == CV_32F);

    cv::Mat &ssd_output = gapi.at("detection_output");

    const int num_objects = static_cast<int>(num_detections.ptr<float>()[0]);
    const float *in_boxes = detection_boxes.ptr<float>();
    const float *in_scores = detection_scores.ptr<float>();
    const float *in_classes = detection_classes.ptr<float>();
    float *ptr = ssd_output.ptr<float>();

    for (int i = 0; i < num_objects; i++) {
        ptr[0] = 0.f;               // "image_id"
        ptr[1] = in_classes[i];     // "label"
        ptr[2] = in_scores[i];      // "confidence"
        ptr[3] = in_boxes[4*i + 1]; // left
        ptr[4] = in_boxes[4*i + 0]; // top
        ptr[5] = in_boxes[4*i + 3]; // right
        ptr[6] = in_boxes[4*i + 2]; // bottom

        ptr      += 7;
        in_boxes += 4;
    }
    if (num_objects < ssd_output.size[2]-1) {
        // put a -1 mark at the end of output blob if there is space left
        ptr[0] = -1.f;
    }
}
} // anonymous namespace


const std::string keys =
    "{ h help | | Print this help message }"
    "{ input  | | Path to the input video file }"
    "{ output | | (Optional) path to output video file }"
    "{ detm   | | Path to an ONNX SSD object detection model (.onnx) }"
    ;

int main(int argc, char *argv[])
{
    cv::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    // Prepare parameters first
    const std::string input = cmd.get<std::string>("input");
    const std::string output = cmd.get<std::string>("output");
    const auto obj_model_path = cmd.get<std::string>("detm");

    auto obj_net = cv::gapi::onnx::Params<custom::ObjDetector>{obj_model_path}
        .cfgOutputLayers({"detection_output"})
        .cfgPostProc({cv::GMatDesc{CV_32F, {1,1,200,7}}}, remap_ssd_ports);
    auto kernels = cv::gapi::kernels< custom::OCVGetSize
                                    , custom::OCVParseSSD
                                    , custom::OCVBBoxes>();
    auto networks = cv::gapi::networks(obj_net);

    // Now build the graph
    cv::GMat in;
    auto blob = cv::gapi::infer<custom::ObjDetector>(in);
    auto  rcs = custom::ParseSSD::on(blob, custom::GetSize::on(in));
    auto  out = cv::gapi::wip::draw::render3ch(in, custom::BBoxes::on(rcs));
    cv::GStreamingCompiled pipeline = cv::GComputation(cv::GIn(in), cv::GOut(out))
        .compileStreaming(cv::compile_args(kernels, networks));

    auto inputs = cv::gin(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input));

    // The execution part
    pipeline.setSource(std::move(inputs));
    pipeline.start();

    cv::VideoWriter writer;

    cv::Mat outMat;
    while (pipeline.pull(cv::gout(outMat))) {
        cv::imshow("Out", outMat);
        cv::waitKey(1);
        if (!output.empty()) {
            if (!writer.isOpened()) {
                const auto sz = cv::Size{outMat.cols, outMat.rows};
                writer.open(output, cv::VideoWriter::fourcc('M','J','P','G'), 25.0, sz);
                CV_Assert(writer.isOpened());
            }
            writer << outMat;
        }
    }
    return 0;
}
