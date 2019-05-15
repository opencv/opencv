#error This is a shadow header file, which is not intended for processing by any compiler. \
       Only bindings parser should handle this file.

namespace cv { namespace dnn {

class CV_EXPORTS_W AsyncMat
{
public:
    //! Wait for Mat object readiness and return it.
    CV_WRAP Mat get();

    //! Wait for Mat object readiness.
    CV_WRAP void wait() const;

    /** @brief Wait for Mat object readiness specific amount of time.
     *  @param timeout Timeout in milliseconds
     *  @returns [std::future_status](https://en.cppreference.com/w/cpp/thread/future_status)
     */
    CV_WRAP AsyncMatStatus wait_for(std::chrono::milliseconds timeout) const;
};

}}
