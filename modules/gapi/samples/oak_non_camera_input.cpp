#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/oak/oak.hpp>

#include <opencv2/highgui.hpp> // CommandLineParser

const std::string keys =
    "{ h help  |              | Print this help message }"
    "{ output  | output.png   | Path to the output file }";

int main(int argc, char *argv[]) {
    cv::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    const std::string output_name = cmd.get<std::string>("output");

    cv::GMat in;
    cv::GArray<uint8_t> out = cv::gapi::oak::encode(in, {});

    auto args = cv::compile_args(cv::gapi::oak::kernels());

    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    // Graph execution /////////////////////////////////////////////////////////
    cv::Mat in_mat(1920, 1080, CV_8UC3);
    cv::randu(in_mat, cv::Scalar::all(1), cv::Scalar::all(255));

    std::vector<uint8_t> out_data;

    c.apply(in_mat, out_data, std::move(args));

    cv::Mat out_mat(1920, 1080, CV_8UC3, out_data.data());

    cv::imwrite(output_name, out_mat);
}
