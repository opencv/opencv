#include <opencv2/gapi.hpp>
#include <opencv2/gapi/s11n.hpp>
#include <opencv2/gapi/garg.hpp>

// for internal usage
#include <../src/api/gbackend_priv.hpp>
#include <../src/backends/common/gbackend.hpp>

int main(int argc, char *argv[])
{
    (void) argc;
    (void) argv;

    std::vector<cv::gimpl::GIslandExecutable::OutObj> output_objs;
    std::map<cv::gimpl::RcDesc, std::size_t> m_out_map;
    cv::GRunArgs out_args(output_objs.size());

    for (auto &&out : output_objs) {
        const auto &idx = m_out_map.at(out.first);
        out_args[idx] = cv::gapi::bind(out.second);
    }
    const auto sargsout = cv::gapi::serialize(out_args);
}