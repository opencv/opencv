#include <errno.h>
#include <string.h>

#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
DataProviderSystemErrorException::DataProviderSystemErrorException(int error_code, const std::string& desription) {
    reason = desription + ", error: " + std::to_string(error_code) + ", desctiption: " + strerror(error_code);
}

DataProviderSystemErrorException::~DataProviderSystemErrorException() = default;

const char* DataProviderSystemErrorException::what() const noexcept {
    return reason.c_str();
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
