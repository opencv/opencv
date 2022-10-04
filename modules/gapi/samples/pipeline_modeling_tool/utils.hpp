#ifndef OPENCV_GAPI_PIPELINE_MODELING_TOOL_UTILS_HPP
#define OPENCV_GAPI_PIPELINE_MODELING_TOOL_UTILS_HPP

#include <map>

#include <opencv2/core.hpp>

#if defined(_WIN32)
#include <windows.h>
#endif

// FIXME: It's better to place it somewhere in common.hpp
struct OutputDescr {
    std::vector<int> dims;
    int              precision;
};

namespace utils {

inline void createNDMat(cv::Mat& mat, const std::vector<int>& dims, int depth) {
    GAPI_Assert(!dims.empty());
    mat.create(dims, depth);
    if (dims.size() == 1) {
        //FIXME: Well-known 1D mat WA
        mat.dims = 1;
    }
}

inline void generateRandom(cv::Mat& out) {
    switch (out.depth()) {
        case CV_8U:
            cv::randu(out, 0, 255);
            break;
        case CV_32F:
            cv::randu(out, 0.f, 1.f);
            break;
        case CV_16F: {
            std::vector<int> dims;
            for (int i = 0; i < out.size.dims(); ++i) {
                dims.push_back(out.size[i]);
            }
            cv::Mat fp32_mat;
            createNDMat(fp32_mat, dims, CV_32F);
            cv::randu(fp32_mat, 0.f, 1.f);
            fp32_mat.convertTo(out, out.type());
            break;
        }
        default:
            throw std::logic_error("Unsupported preprocessing depth");
    }
}

inline void sleep(double ms) {
#if defined(_WIN32)
    // NB: It takes portions of 100 nanoseconds.
    int64_t ns_units = static_cast<int64_t>(ms * 1e4);
    // FIXME: Wrap it to RAII and instance only once.
    HANDLE timer = CreateWaitableTimer(NULL, true, NULL);
    if (!timer) {
        throw std::logic_error("Failed to create timer");
    }

    LARGE_INTEGER li;
    li.QuadPart = -ns_units;
    if(!SetWaitableTimer(timer, &li, 0, NULL, NULL, false)){
        CloseHandle(timer);
        throw std::logic_error("Failed to set timer");
    }
    if (WaitForSingleObject(timer, INFINITE) != WAIT_OBJECT_0) {
        CloseHandle(timer);
        throw std::logic_error("Failed to wait timer");
    }
    CloseHandle(timer);
#else
    using namespace std::chrono;
    std::this_thread::sleep_for(duration<double, std::milli>(ms));
#endif
}

template <typename duration_t>
typename duration_t::rep measure(std::function<void()> f) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    f();
    return duration_cast<duration_t>(
            high_resolution_clock::now() - start).count();
}

template <typename duration_t>
typename duration_t::rep timestamp() {
    using namespace std::chrono;
    auto now = high_resolution_clock::now();
    return duration_cast<duration_t>(now.time_since_epoch()).count();
}

template <typename K, typename V>
void mergeMapWith(std::map<K, V>& target, const std::map<K, V>& second) {
    for (auto&& item : second) {
        auto it = target.find(item.first);
        if (it != target.end()) {
            throw std::logic_error("Error: key: " + it->first + " is already in target map");
        }
        target.insert(item);
    }
}

template <typename T>
double avg(const std::vector<T>& vec) {
    return std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
}

template <typename T>
T max(const std::vector<T>& vec) {
    return *std::max_element(vec.begin(), vec.end());
}

template <typename T>
T min(const std::vector<T>& vec) {
    return *std::min_element(vec.begin(), vec.end());
}

template <typename T>
int64_t ms_to_mcs(T ms) {
    using namespace std::chrono;
    return duration_cast<microseconds>(duration<T, std::milli>(ms)).count();
}

} // namespace utils

#endif // OPENCV_GAPI_PIPELINE_MODELING_TOOL_UTILS_HPP
