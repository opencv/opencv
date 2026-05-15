#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>

static int g_error_count = 0;

static int errorCallback(int status, const char* func_name, const char* err_msg,
                         const char* file_name, int line, void*)
{
    ++g_error_count;
    std::cout << "\n[OpenCV error captured]" << std::endl;
    std::cout << "status: " << status << std::endl;
    std::cout << "function: " << func_name << std::endl;
    std::cout << "message: 1" << err_msg << std::endl;
    std::cout << "location: " << file_name << ":" << line << std::endl;
    return 0;
}

int main()
{
    cv::redirectError(errorCallback);

    if (!cv::ocl::haveOpenCL()) {
        std::cout << "No OpenCL device, cannot reproduce." << std::endl;
        return 0;
    }

    cv::ocl::setUseOpenCL(true);
    cv::ocl::Context context = cv::ocl::Context::getDefault();
    cv::ocl::Device device = cv::ocl::Device::getDefault();

    std::cout << "OpenCL enabled: " << (cv::ocl::useOpenCL() ? "YES" : "NO") << std::endl;
    std::cout << "OpenCL device: " << device.name() << std::endl;

    cv::Mat data = (cv::Mat_<float>(4, 3) <<
        1, 2, 3,
        2, 3, 4,
        3, 4, 5,
        4, 5, 6);

    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

    cv::Mat result_mat = pca.project(data);
    std::cout << "\nMat project OK, size: " << result_mat.size() << std::endl;

    g_error_count = 0;
    cv::UMat result_umat;
    pca.project(data, result_umat);

    std::cout << "UMat project returned, size: " << result_umat.size() << std::endl;
    if (g_error_count > 0) {
        std::cout << "Reproduced: OpenCL path reported an error, then OpenCV fell back to CPU." << std::endl;
        return 1;
    }

    std::cout << "No OpenCL error was captured." << std::endl;
    return 0;
}
