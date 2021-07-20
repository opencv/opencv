#ifndef OPENCV_GAPI_STREAMING_ONEVPL_PRIV_BUILDER_HPP
#define OPENCV_GAPI_STREAMING_ONEVPL_PRIV_BUILDER_HPP

#include <memory>
#include <string>

#include <opencv2/gapi/streaming/onevpl_cap.hpp>
#include "streaming/vpl/vpl_utils.hpp"

namespace cv {
namespace gapi {
namespace wip {
    
struct OneVPLCapture::IPriv;


class oneVPLBulder
{
public:

    oneVPLBulder();
    
    template<typename... Param>
    void set(Param&& ...params)
    {
        std::array<bool, sizeof...(params)> expander {
        (set_arg(params), true)...};
        (void)expander;
    }

    std::unique_ptr<OneVPLCapture::IPriv> build() const;

    void set_arg(const std::string& file_path);
    void set_arg(const CFGParams& params);
private:
    std::string filePath;
    CFGParams cfg_params;
};
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // OPENCV_GAPI_STREAMING_ONEVPL_PRIV_BUILDER_HPP
