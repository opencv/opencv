#include <algorithm>
#include <fstream>
#include <iostream>
#include <cctype>

#include <opencv2/imgproc.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/streaming/onevpl_cap.hpp>
#include <opencv2/highgui.hpp> // CommandLineParser

const std::string about =
    "This is an OpenCV-based version of oneVPLSource decoder example";
const std::string keys =
    "{ h help       |                                           | Print this help message }"
    "{ input        |                                           | Path to the input demultiplexed video file }"
    "{ output       |                                           | Path to the output RAW video file. Use .avi extension }"
    "{ ssm          | semantic-segmentation-adas-0001.xml       | Path to OpenVINO IE semantic segmentation model (.xml) }"
    "{ cfg_param_1  | <property name>:<property value>          | oneVPL variant is used for configuring source (see `MFXSetConfigFilterProperty` by https://spec.oneapi.io/versions/latest/elements/oneVPL/source/index.html) }"
    "{ cfg_param_2  |                                           | Another one oneVPL variant is used for configuring source }"
    "{ cfg_param_3  |                                           | Another one oneVPL variant is used for configuring source }"
    "{ cfg_param_4  |                                           | Another one oneVPL variant is used for configuring source }"
    "{ cfg_param_5  |                                           | Another one oneVPL variant is used for configuring source }";

// 20 colors for 20 classes of semantic-segmentation-adas-0001
const std::vector<cv::Vec3b> colors = {
    { 128, 64,  128 },
    { 232, 35,  244 },
    { 70,  70,  70 },
    { 156, 102, 102 },
    { 153, 153, 190 },
    { 153, 153, 153 },
    { 30,  170, 250 },
    { 0,   220, 220 },
    { 35,  142, 107 },
    { 152, 251, 152 },
    { 180, 130, 70 },
    { 60,  20,  220 },
    { 0,   0,   255 },
    { 142, 0,   0 },
    { 70,  0,   0 },
    { 100, 60,  0 },
    { 90,  0,   0 },
    { 230, 0,   0 },
    { 32,  11,  119 },
    { 0,   74,  111 },
};

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
G_API_OP(PostProcessing, <cv::GMat(cv::GMat, cv::GMat)>, "sample.custom.post_processing") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in, const cv::GMatDesc &) {
        return in;
    }
};

GAPI_OCV_KERNEL(OCVPostProcessing, PostProcessing) {
    static void run(const cv::Mat &in, const cv::Mat &detected_classes, cv::Mat &out) {
        // This kernel constructs output image by class table and colors vector

        // The semantic-segmentation-adas-0001 output a blob with the shape
        // [B, C=1, H=1024, W=2048]
        const int outHeight = 1024;
        const int outWidth = 2048;
        cv::Mat maskImg(outHeight, outWidth, CV_8UC3);
        const int* const classes = detected_classes.ptr<int>();
        for (int rowId = 0; rowId < outHeight; ++rowId) {
            for (int colId = 0; colId < outWidth; ++colId) {
                size_t classId = static_cast<size_t>(classes[rowId * outWidth + colId]);
                maskImg.at<cv::Vec3b>(rowId, colId) =
                    classId < colors.size()
                        ? colors[classId]
                        : cv::Vec3b{0, 0, 0}; // sample detects 20 classes
            }
        }

        cv::resize(maskImg, out, in.size());
        const float blending = 0.3f;
        out = in * blending + out * (1 - blending);
    }
};
} // namespace custom

namespace detail {
typename cv::gapi::wip::oneVPL_cfg_param create_from_string(const std::string &line);
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
    const auto model_path    = cmd.get<std::string>("ssm");
    const auto weights_path  = get_weights_path(model_path);

    // check ouput file extension
    if (!output.empty()) {
        auto ext = output.find_last_of(".");
        if (ext == std::string::npos || (output.substr(ext + 1) != "avi")) {
            std::cerr << "Output file should have *.avi extension for output video" << std::endl;
            return -1;
        }
    }

    // get VPL params from cmd
    int param_index = 1;
    std::vector<cv::gapi::wip::oneVPL_cfg_param> source_cfgs;
    try {
        do {
            std::string line = cmd.get<std::string>("cfg_param_" + std::to_string(param_index));
            if (line.empty()) {
                break;
            }
            source_cfgs.push_back(detail::create_from_string(line));
            param_index++;
        } while(true);
    } catch (const std::exception& ex) {
        std::cerr << "Invalid cfg parameter: " << ex.what() << std::endl;
        return -1;
    }

    const auto device = "CPU";
    G_API_NET(SemSegmNet, <cv::GMat(cv::GMat)>, "semantic-segmentation");
    const auto net = cv::gapi::ie::Params<SemSegmNet> {
        model_path, weights_path, device
    };
    const auto kernels = cv::gapi::kernels<custom::OCVPostProcessing>();
    const auto networks = cv::gapi::networks(net);

    // Now build the graph
    cv::GFrame in;
    cv::GMat detected_classes = cv::gapi::infer<SemSegmNet>(in);
    cv::GMat out = custom::PostProcessing::on(cv::gapi::streaming::BGR(in), detected_classes);

    cv::GStreamingCompiled pipeline = cv::GComputation(cv::GIn(in), cv::GOut(out))
        .compileStreaming(cv::compile_args(kernels, networks));

    // Create source
    cv::Ptr<cv::gapi::wip::IStreamSource> cap;
    try {
        cap = cv::gapi::wip::make_vpl_src(file_path, source_cfgs);
        std::cout << "oneVPL source desription: " << cap->descr_of() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Cannot create source: " << ex.what() << std::endl;
        return -1;
    }

    cv::GMetaArg descr = cap->descr_of();
    auto frame_descr = cv::util::get<cv::GFrameDesc>(descr);
    
    // The execution part
    pipeline.setSource(std::move(cap));
    pipeline.start();

    int framesCount = 0;
    cv::TickMeter t;
    cv::VideoWriter writer;
    cv::Mat outMat;
    t.start();
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
        framesCount++;
    }
    t.stop();
    std::cout << "Elapsed time: " << t.getTimeSec() << std::endl;
    std::cout << "FPS: " << framesCount /  t.getTimeSec() << std::endl;

    return 0;
}


namespace detail {
typename cv::gapi::wip::oneVPL_cfg_param create_from_string(const std::string &line) {
    using namespace cv::gapi::wip;

    if (line.empty()) {
        throw std::runtime_error("Cannot parse oneVPL_cfg_param from emply line");
    }

    std::string::size_type name_endline_pos = line.find(':');
    if (name_endline_pos == std::string::npos) {
        throw std::runtime_error("Cannot parse oneVPL_cfg_param from: " + line +
                                 "\nExpected separator \":\"");
    }

    std::string name = line.substr(0, name_endline_pos);
    std::string value = line.substr(name_endline_pos + 1);

    return cv::gapi::wip::oneVPLBulder::create_cfg_param(name, value);
}
}
