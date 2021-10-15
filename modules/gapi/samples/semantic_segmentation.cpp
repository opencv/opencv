#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/highgui.hpp>

const std::string keys =
    "{ h help |                                     | Print this help message }"
    "{ input  |                                     | Path to the input video file }"
    "{ output |                                     | Path to the output video file }"
    "{ ssm    | semantic-segmentation-adas-0001.xml | Path to OpenVINO IE semantic segmentation model (.xml) }";

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

void classesToColors(const cv::Mat &out_blob,
                           cv::Mat &mask_img) {
    const int H = out_blob.size[0];
    const int W = out_blob.size[1];

    mask_img.create(H, W, CV_8UC3);
    GAPI_Assert(out_blob.type() == CV_8UC1);
    const uint8_t* const classes = out_blob.ptr<uint8_t>();

    for (int rowId = 0; rowId < H; ++rowId) {
        for (int colId = 0; colId < W; ++colId) {
            uint8_t class_id = classes[rowId * W + colId];
            mask_img.at<cv::Vec3b>(rowId, colId) =
                class_id < colors.size()
                ? colors[class_id]
                : cv::Vec3b{0, 0, 0}; // NB: sample supports 20 classes
        }
    }
}

void probsToClasses(const cv::Mat& probs, cv::Mat& classes) {
     const int C = probs.size[1];
     const int H = probs.size[2];
     const int W = probs.size[3];

     classes.create(H, W, CV_8UC1);
     GAPI_Assert(probs.depth() == CV_32F);
     float* out_p       = reinterpret_cast<float*>(probs.data);
     uint8_t* classes_p = reinterpret_cast<uint8_t*>(classes.data);

     for (int h = 0; h < H; ++h) {
         for (int w = 0; w < W; ++w) {
             double max = 0;
             int class_id = 0;
             for (int c = 0; c < C; ++c) {
                int idx = c * H * W + h * W + w;
                    if (out_p[idx] > max) {
                        max = out_p[idx];
                        class_id = c;
                    }
             }
             classes_p[h * W + w] = static_cast<uint8_t>(class_id);
         }
     }
}

} // anonymous namespace

namespace custom {
G_API_OP(PostProcessing, <cv::GMat(cv::GMat, cv::GMat)>, "sample.custom.post_processing") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in, const cv::GMatDesc &) {
        return in;
    }
};

GAPI_OCV_KERNEL(OCVPostProcessing, PostProcessing) {
    static void run(const cv::Mat &in, const cv::Mat &out_blob, cv::Mat &out) {
        cv::Mat classes;
        // NB: If output has more than single plane, it contains probabilities
        // otherwise class id.
        if (out_blob.size[1] > 1) {
            probsToClasses(out_blob, classes);
        } else {
            out_blob.convertTo(classes, CV_8UC1);
            classes = classes.reshape(1, out_blob.size[2]);
        }

        cv::Mat mask_img;
        classesToColors(classes, mask_img);

        cv::resize(mask_img, out, in.size());
        const float blending = 0.3f;
        out = in * blending + out * (1 - blending);
    }
};
} // namespace custom

int main(int argc, char *argv[]) {
    cv::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    // Prepare parameters first
    const std::string input  = cmd.get<std::string>("input");
    const std::string output = cmd.get<std::string>("output");
    const auto model_path    = cmd.get<std::string>("ssm");
    const auto weights_path  = get_weights_path(model_path);
    const auto device        = "CPU";
    G_API_NET(SemSegmNet, <cv::GMat(cv::GMat)>, "semantic-segmentation");
    const auto net = cv::gapi::ie::Params<SemSegmNet> {
        model_path, weights_path, device
    };
    const auto kernels = cv::gapi::kernels<custom::OCVPostProcessing>();
    const auto networks = cv::gapi::networks(net);

    // Now build the graph
    cv::GMat in;
    cv::GMat out_blob = cv::gapi::infer<SemSegmNet>(in);
    cv::GMat out = custom::PostProcessing::on(in, out_blob);

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
