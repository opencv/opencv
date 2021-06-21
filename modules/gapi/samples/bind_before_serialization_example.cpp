#include <opencv2/gapi.hpp>
#include <opencv2/gapi/s11n.hpp>
#include <opencv2/gapi/garg.hpp>

int main(int argc, char *argv[])
{
    (void) argc;
    (void) argv;

    std::vector<cv::GRunArgP> graph_outs;
    cv::GRunArgs out_args;

    for (auto &&out : graph_outs) {
        out_args.emplace_back(cv::gapi::bind(out));
    }
    const auto sargsout = cv::gapi::serialize(out_args);
}
