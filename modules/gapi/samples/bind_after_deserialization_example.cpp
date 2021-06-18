#include <opencv2/gapi.hpp>
#include <opencv2/gapi/s11n.hpp>
#include <opencv2/gapi/garg.hpp>

int main(int argc, char *argv[])
{
    (void) argc;
    (void) argv;

    cv::GCompiled compd;
    std::vector<char> bytes;
    auto graph = cv::gapi::deserialize<cv::GComputation>(bytes);
    auto meta = cv::gapi::deserialize<cv::GMetaArgs>(bytes);

    compd = graph.compile(std::move(meta), cv::compile_args());
    auto in_args  = cv::gapi::deserialize<cv::GRunArgs>(bytes);
    auto out_args = cv::gapi::deserialize<cv::GRunArgs>(bytes);
    compd(std::move(in_args), cv::gapi::bind(out_args));
}