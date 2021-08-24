#include <algorithm>
#include <fstream>
#include <iostream>
#include <cctype>

#include <opencv2/imgproc.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/render.hpp>
#include <opencv2/gapi/streaming/onevpl/source.hpp>
#include <opencv2/highgui.hpp> // CommandLineParser

const std::string about =
    "This is an OpenCV-based version of oneVPLSource decoder example";
const std::string keys =
    "{ h help       |                                           | Print this help message }"
    "{ input        |                                           | Path to the input demultiplexed video file }"
    "{ output       |                                           | Path to the output RAW video file. Use .avi extension }"
    "{ facem        | face-detection-adas-0001.xml              | Path to OpenVINO IE face detection model (.xml) }"
    "{ cfg_params   | <prop name>:<value>;<prop name>:<value>   | Semicolon separated list of oneVPL mfxVariants which is used for configuring source (see `MFXSetConfigFilterProperty` by https://spec.oneapi.io/versions/latest/elements/oneVPL/source/index.html) }";


namespace {
std::string get_weights_path(const std::string &model_path) {
    const auto EXT_LEN = 4u;
    const auto sz = model_path.size();
    CV_Assert(sz > EXT_LEN);

    auto ext = model_path.substr(sz - EXT_LEN);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){
            return static_cast<unsigned char>(std::tolower(c));
        });
    CV_Assert(ext == ".xml");
    return model_path.substr(0u, sz - EXT_LEN) + ".bin";
}
} // anonymous namespace

namespace custom {
G_API_NET(FaceDetector,   <cv::GMat(cv::GMat)>, "face-detector");

using GDetections = cv::GArray<cv::Rect>;
using GRect       = cv::GOpaque<cv::Rect>;
using GSize       = cv::GOpaque<cv::Size>;
using GPrims      = cv::GArray<cv::gapi::wip::draw::Prim>;

G_API_OP(LocateROI, <GRect(GSize)>, "sample.custom.locate-roi") {
    static cv::GOpaqueDesc outMeta(const cv::GOpaqueDesc &) {
        return cv::empty_gopaque_desc();
    }
};

G_API_OP(ParseSSD, <GDetections(cv::GMat, GRect, GSize)>, "sample.custom.parse-ssd") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GOpaqueDesc &, const cv::GOpaqueDesc &) {
        return cv::empty_array_desc();
    }
};

G_API_OP(BBoxes, <GPrims(GDetections, GRect)>, "sample.custom.b-boxes") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc &, const cv::GOpaqueDesc &) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVLocateROI, LocateROI) {
    // This is the place where we can run extra analytics
    // on the input image frame and select the ROI (region
    // of interest) where we want to detect our objects (or
    // run any other inference).
    //
    // Currently it doesn't do anything intelligent,
    // but only crops the input image to square (this is
    // the most convenient aspect ratio for detectors to use)

    static void run(const cv::Size& in_size, cv::Rect &out_rect) {

        // Identify the central point & square size (- some padding)
        const auto center = cv::Point{in_size.width/2, in_size.height/2};
        auto sqside = std::min(in_size.width, in_size.height);

        // Now build the central square ROI
        out_rect = cv::Rect{ center.x - sqside/2
                           , center.y - sqside/2
                           , sqside
                           , sqside
                           };
    }
};

GAPI_OCV_KERNEL(OCVParseSSD, ParseSSD) {
    static void run(const cv::Mat &in_ssd_result,
                    const cv::Rect &in_roi,
                    const cv::Size &in_parent_size,
                    std::vector<cv::Rect> &out_objects) {
        const auto &in_ssd_dims = in_ssd_result.size;
        CV_Assert(in_ssd_dims.dims() == 4u);

        const int MAX_PROPOSALS = in_ssd_dims[2];
        const int OBJECT_SIZE   = in_ssd_dims[3];
        CV_Assert(OBJECT_SIZE  == 7); // fixed SSD object size

        const cv::Size up_roi = in_roi.size();
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
            // taking the ROI into account
            cv::Rect rc;
            rc.x      = static_cast<int>(rc_left   * up_roi.width);
            rc.y      = static_cast<int>(rc_top    * up_roi.height);
            rc.width  = static_cast<int>(rc_right  * up_roi.width)  - rc.x;
            rc.height = static_cast<int>(rc_bottom * up_roi.height) - rc.y;
            rc.x += in_roi.x;
            rc.y += in_roi.y;
            out_objects.emplace_back(rc & surface);
        }
    }
};

GAPI_OCV_KERNEL(OCVBBoxes, BBoxes) {
    // This kernel converts the rectangles into G-API's
    // rendering primitives
    static void run(const std::vector<cv::Rect> &in_face_rcs,
                    const             cv::Rect  &in_roi,
                          std::vector<cv::gapi::wip::draw::Prim> &out_prims) {
        out_prims.clear();
        const auto cvt = [](const cv::Rect &rc, const cv::Scalar &clr) {
            return cv::gapi::wip::draw::Rect(rc, clr, 2);
        };
        out_prims.emplace_back(cvt(in_roi, CV_RGB(0,255,255))); // cyan
        for (auto &&rc : in_face_rcs) {
            out_prims.emplace_back(cvt(rc, CV_RGB(0,255,0)));   // green
        }
    }
};

} // namespace custom

namespace cfg {
typename cv::gapi::wip::onevpl::CfgParam create_from_string(const std::string &line);
}

int main(int argc, char *argv[]) {

    cv::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    // get file name
    std::string file_path = cmd.get<std::string>("input");
    const std::string output = cmd.get<std::string>("output");
    const auto face_model_path = cmd.get<std::string>("facem");

    // check ouput file extension
    if (!output.empty()) {
        auto ext = output.find_last_of(".");
        if (ext == std::string::npos || (output.substr(ext + 1) != "avi")) {
            std::cerr << "Output file should have *.avi extension for output video" << std::endl;
            return -1;
        }
    }

    // get oneVPL cfg params from cmd
    std::stringstream params_list(cmd.get<std::string>("cfg_params"));
    std::vector<cv::gapi::wip::onevpl::CfgParam> source_cfgs;
    try {
        std::string line;
        while (std::getline(params_list, line, ';')) {
            source_cfgs.push_back(cfg::create_from_string(line));
        }
    } catch (const std::exception& ex) {
        std::cerr << "Invalid cfg parameter: " << ex.what() << std::endl;
        return -1;
    }

    auto face_net = cv::gapi::ie::Params<custom::FaceDetector> {
        face_model_path,                 // path to topology IR
        get_weights_path(face_model_path)   // path to weights
    };
    auto kernels = cv::gapi::kernels
        < custom::OCVLocateROI
        , custom::OCVParseSSD
        , custom::OCVBBoxes>();
    auto networks = cv::gapi::networks(face_net);

    // Create source
    cv::Ptr<cv::gapi::wip::IStreamSource> cap;
    try {
        cap = cv::gapi::wip::make_onevpl_src(file_path, source_cfgs);
        std::cout << "oneVPL source desription: " << cap->descr_of() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Cannot create source: " << ex.what() << std::endl;
        return -1;
    }

    cv::GMetaArg descr = cap->descr_of();
    auto frame_descr = cv::util::get<cv::GFrameDesc>(descr);

    // Now build the graph
    cv::GFrame in;
    auto size = cv::gapi::streaming::size(in);
    auto roi = custom::LocateROI::on(size);
    auto blob = cv::gapi::infer<custom::FaceDetector>(roi, in);
    auto rcs = custom::ParseSSD::on(blob, roi, size);
    auto out_frame = cv::gapi::wip::draw::renderFrame(in, custom::BBoxes::on(rcs, roi));
    auto out = cv::gapi::streaming::BGR(out_frame);

    cv::GStreamingCompiled pipeline;
    try {
        pipeline = cv::GComputation(cv::GIn(in), cv::GOut(out))
                .compileStreaming(cv::compile_args(kernels, networks));
    } catch (const std::exception& ex) {
        std::cerr << "Exception occured during pipeline construction: " << ex.what() << std::endl;
        return -1;
    }
    // The execution part

    // TODO USE may set pool size from outside and set queue_capacity size,
    // compile arg: cv::gapi::streaming::queue_capacity
    pipeline.setSource(std::move(cap));
    pipeline.start();

    int framesCount = 0;
    cv::TickMeter t;
    cv::VideoWriter writer;
    if (!output.empty() && !writer.isOpened()) {
        const auto sz = cv::Size{frame_descr.size.width, frame_descr.size.height};
        writer.open(output, cv::VideoWriter::fourcc('M','J','P','G'), 25.0, sz);
        CV_Assert(writer.isOpened());
    }

    cv::Mat outMat;
    t.start();
    while (pipeline.pull(cv::gout(outMat))) {
        cv::imshow("Out", outMat);
        cv::waitKey(1);
        if (!output.empty()) {
            writer << outMat;
        }
        framesCount++;
    }
    t.stop();
    std::cout << "Elapsed time: " << t.getTimeSec() << std::endl;
    std::cout << "FPS: " << framesCount /  t.getTimeSec() << std::endl;
    std::cout << "framesCount: " << framesCount << std::endl;

    return 0;
}


namespace cfg {
typename cv::gapi::wip::onevpl::CfgParam create_from_string(const std::string &line) {
    using namespace cv::gapi::wip;

    if (line.empty()) {
        throw std::runtime_error("Cannot parse CfgParam from emply line");
    }

    std::string::size_type name_endline_pos = line.find(':');
    if (name_endline_pos == std::string::npos) {
        throw std::runtime_error("Cannot parse CfgParam from: " + line +
                                 "\nExpected separator \":\"");
    }

    std::string name = line.substr(0, name_endline_pos);
    std::string value = line.substr(name_endline_pos + 1);

    return cv::gapi::wip::onevpl::CfgParam::create(name, value);
}
}
