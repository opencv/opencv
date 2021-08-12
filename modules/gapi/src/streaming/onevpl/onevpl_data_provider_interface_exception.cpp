#include <opencv2/gapi/streaming/onevpl/onevpl_data_provider_interface.hpp>

namespace cv {
namespace gapi {
namespace wip {
DataProviderSystemErrorException::DataProviderSystemErrorException(int error_code, const std::string& desription) {
    reason = desription + ", error: " + std::to_string(error_code) + ", desctiption: " + strerror(error_code);
}

DataProviderSystemErrorException::~DataProviderSystemErrorException() {
}

const char* DataProviderSystemErrorException::what() const noexcept {
    return reason.c_str();
}
} // namespace wip
} // namespace gapi
} // namespace cv
