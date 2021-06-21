#include <opencv2/gapi.hpp>
#include <opencv2/gapi/s11n.hpp>
#include <opencv2/gapi/garg.hpp>

// FIXME: forward declaration since this code represents
//        internal backend details
namespace cv
{
namespace gimpl
{
    class RcDesc;
    using OutObj = std::pair<RcDesc, cv::GRunArgP>;
} // namespace gimpl
} // namespace cv

int main(int argc, char *argv[])
{
    (void) argc;
    (void) argv;

    std::vector<cv::gimpl::OutObj> output_objs;
    std::map<cv::gimpl::RcDesc, std::size_t> m_out_map;
    cv::GRunArgs out_args(output_objs.size());

    for (auto &&out : output_objs) {
        const auto &idx = m_out_map.at(out.first);
        out_args[idx] = cv::gapi::bind(out.second);
    }
    const auto sargsout = cv::gapi::serialize(out_args);
}
