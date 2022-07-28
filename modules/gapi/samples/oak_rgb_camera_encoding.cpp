#include <fstream>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/gframe.hpp>

#include <opencv2/gapi/oak/oak.hpp>
#include <opencv2/gapi/streaming/format.hpp> // BGR accessor

#include <opencv2/highgui.hpp> // CommandLineParser

const std::string keys =
    "{ h help  |              | Print this help message }"
    "{ output  | output.h265  | Path to the output .h265 video file }";

int main(int argc, char *argv[]) {
    cv::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    const std::string output_name = cmd.get<std::string>("output");

    cv::gapi::oak::EncoderConfig cfg;
    cfg.profile = cv::gapi::oak::EncoderConfig::Profile::H265_MAIN;

    cv::GFrame in;
    cv::GArray<uint8_t> encoded = cv::gapi::oak::encode(in, cfg);

    auto args = cv::compile_args(cv::gapi::oak::ColorCameraParams{}, cv::gapi::oak::kernels());

    auto pipeline = cv::GComputation(cv::GIn(in), cv::GOut(encoded)).compileStreaming(std::move(args));

    // Graph execution /////////////////////////////////////////////////////////
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::oak::ColorCamera>());
    pipeline.start();

    std::vector<uint8_t> out_h265_data;

    std::ofstream out_h265_file;
    out_h265_file.open(output_name, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

    // Pull 300 frames from the camera
    uint32_t frames = 300;
    uint32_t pulled = 0;

    while (pipeline.pull(cv::gout(out_h265_data))) {
        if (out_h265_file.is_open()) {
            out_h265_file.write(reinterpret_cast<const char*>(out_h265_data.data()),
                                                              out_h265_data.size());
        }
        if (pulled++ == frames) {
            pipeline.stop();
            break;
        }
    }

    std::cout << "Pipeline finished: " << output_name << " file has been written." << std::endl;
}
