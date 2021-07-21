#include <opencv2/gapi/util/throw.hpp>

#include <opencv2/gapi/streaming/onevpl_builder.hpp>
#include "streaming/vpl/vpl_source_impl.hpp"

namespace cv {
namespace gapi {
namespace wip {

std::shared_ptr<IStreamSource> oneVPLBulder::build() const
{
    if (filePath.empty()) {
        util::throw_error(std::logic_error("Cannot create 'OneVPLCapture' on empty source file name"));
    }
#ifdef HAVE_ONEVPL
    std::unique_ptr<VPLSourceImpl> impl(new VPLSourceImpl(filePath, cfg_params));
    impl->initializeHWAccel();
    return std::shared_ptr<IStreamSource>(new OneVPLCapture(std::move(impl)));
#else
    abort();
#endif
}

void oneVPLBulder::set_arg(const std::string& file_path)
{
    this->filePath = file_path;
}

void oneVPLBulder::set_arg(const CFGParams& params)
{
    cfg_params = params;
}
} // namespace wip
} // namespace gapi
} // namespace cv
