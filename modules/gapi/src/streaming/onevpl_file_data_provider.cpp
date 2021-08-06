#include "streaming/onevpl_file_data_provider.hpp"

namespace cv {
namespace gapi {
namespace wip {

FileDataProvider::FileDataProvider(const std::string& file_path) : 
    source_handle(fopen(file_path.c_str(), "rb"), &fclose) {
    if (!source_handle) {
        throw std::runtime_error("FileDataProvider: cannot open source file: " + file_path);
    }
}

FileDataProvider::~FileDataProvider() {
}

size_t FileDataProvider::provide_data(size_t out_data_bytes_size, void* out_data) {
    if (empty()) {
        return 0;
    }

    size_t ret = fread(out_data, 1, out_data_bytes_size, source_handle.get());
    if (ret == 0) {
        if (feof(source_handle.get())) {
            source_handle.reset();
        } else { 
            throw DataProviderSystemErrorException (errno, "FileDataProvider::provide_data error read");
        }
    }
    return ret;
}

bool FileDataProvider::empty() const {
    return !source_handle;
}
} // namespace wip
} // namespace gapi
} // namespace cv
